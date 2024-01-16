from datetime import datetime

import chess.pgn
import torch
from accelerate import Accelerator
from torch import optim
from transformers import AutoTokenizer, GPTNeoXTokenizerFast, GPTNeoXForCausalLM

import soft_prompting.direct_soft_prompt
from chess_training.dataset_loading import AsyncChessBatchLoader, UnconditionalAsyncChessBatchLoader, \
    AsyncChessBatchLoaderWithOnes
from soft_prompting import training_and_testing, SoftPromptFactory, SnapshotPathCreator, try_create_snapshot
from language_training import autoregressive_baseline, AutoregressiveBaseline
from soft_prompting.data_logger import DataLogger

from enum import Enum


class SoftPromptParameterMode(Enum):
    """
    The mode to use for soft prompt parameters.
    """

    UNCONDITIONAL = 1
    """
    The soft prompt will not be given any conditional parameters; the soft prompt type must be a DirectSoftPrompt.
    """
    ELOS = 2
    """
    The soft prompt parameters will contain the Elos of the players in the game.
    """
    ELOS_PLUS_ONES = 3
    """
    The soft prompt parameters will contain the Elos of the players in the game, plus an additional 1.
    Intended for use in investigating optimizer behavior in linear versus nonlinear input models.
    """


def game_processor(game: chess.pgn.Game):
    if game.headers['Event'] != 'Rated Blitz game':
        return None
    # Convert the moves to a string and append it to the output batch.
    # Some games have zero moves! Skip those regardless of what the filter says.
    moves = ' '.join(move.uci() for move in game.mainline_moves())
    if len(moves) > 0:
        return (int(game.headers['WhiteElo']), int(game.headers['BlackElo'])), moves
    else:
        return None


def train_and_test_chess(chess_database_path: str, model_configurations: list[tuple[str, int]],
                         soft_prompt_token_counts: list[int],
                         soft_prompt_creator: SoftPromptFactory,
                         soft_prompt_parameter_mode: SoftPromptParameterMode = SoftPromptParameterMode.ELOS,
                         logging_prefix: str = "",
                         training_step_count: int = 512, batch_lanes_per_step: int = 32,
                         maximum_sample_length_in_tokens: int = 256, learning_rate: float = 1e-3,
                         weight_decay: float = 1e-4,
                         forward_test_generated_token_count: int = 128,
                         snapshot_path_creator: SnapshotPathCreator | None = None) -> None:
    """
    Trains soft prompts on the chess dataset. Trains a soft prompt for each model configuration for each
    soft prompt token count.

    :param chess_database_path: The path to the database to use.
                                Expected to be a compressed .zst file containing pgn-formatted games,
                                like those from lichess.
    :param model_configurations: A list of tuples containing the model size
                                 (as it appears in the pythia model name, like '70m'), plus the accumulation steps to
                                 use for each size.
    :param soft_prompt_token_counts: A list of soft prompt token counts to train.
    :param soft_prompt_creator: A function which creates a soft prompt for a training scenario.
    :param soft_prompt_parameter_mode: The mode to use for soft prompt parameters.
    :param logging_prefix: The prefix to prepend to the name of tensorboard log outputs.
    :param training_step_count: The number of training steps to use.
    :param batch_lanes_per_step: The number of batch lanes to use per optimization step. Must be a multiple of
                                 the accumulation step count defined in any configuration defined in
                                 model_configurations.
    :param maximum_sample_length_in_tokens: The maximum sample length in tokens pulled from the database.
    :param learning_rate: The learning rate to use.
    :param weight_decay: The weight decay to use.
    :param forward_test_generated_token_count: The number of tokens to generate when doing forward generation testing.
    :param snapshot_path_creator: Function which creates paths to save snapshots to.
                                  Takes
                                  If None, no snapshots will be saved.
                                  Snapshots will be saved as a tuple of (soft prompt state dict, metadata dict).
    """

    for model_size, accumulation_step_count in model_configurations:
        pythia_model_name = f'pythia-{model_size}-deduped'
        print(f'Preparing model {pythia_model_name}...')
        model: GPTNeoXForCausalLM = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/{pythia_model_name}")
        tokenizer: GPTNeoXTokenizerFast = AutoTokenizer.from_pretrained(f"EleutherAI/{pythia_model_name}")
        tokenizer.pad_token = tokenizer.eos_token
        parameter_count = sum(param.numel() for param in model.parameters() if param.requires_grad)

        # Note that training step counts refer to the number of optimization steps, not the number of batches.
        batch_size = batch_lanes_per_step // accumulation_step_count
        if batch_size * accumulation_step_count != batch_lanes_per_step:
            raise ValueError(f'Batch lanes per step ({batch_lanes_per_step}) must be a multiple of '
                             f'accumulation step count ({accumulation_step_count}).')
        accelerator = Accelerator(gradient_accumulation_steps=accumulation_step_count)

        for soft_prompt_token_count in soft_prompt_token_counts:
            # with (chess_loading.ChessBatchLoader(
            #         chess_database_path, tokenizer, batch_size,
            #         maximum_sample_length_in_tokens=maximum_sample_length_in_tokens,
            #         allow_game_filter=lambda game: game.headers['Event'] == 'Rated Blitz game') as batch_loader):
            batch_loader_parameters = {
                'path': chess_database_path,
                'game_processor': game_processor,
                'tokenizer': tokenizer,
                'batch_size': batch_size,
                'maximum_sample_length_in_tokens': maximum_sample_length_in_tokens,
                'num_processes_in_training': accelerator.num_processes,
                'process_index_in_training': accelerator.process_index}
            if soft_prompt_parameter_mode == SoftPromptParameterMode.UNCONDITIONAL:
                batch_loader = UnconditionalAsyncChessBatchLoader(**batch_loader_parameters)
            elif soft_prompt_parameter_mode == SoftPromptParameterMode.ELOS:
                batch_loader = AsyncChessBatchLoader(**batch_loader_parameters)
            else:  # soft_prompt_parameter_mode == SoftPromptParameterMode.ElosPlusOnes:
                batch_loader = AsyncChessBatchLoaderWithOnes(**batch_loader_parameters)
            with batch_loader:
                # Reusing the same batch loader for both. S'fine; won't be the same data.
                test_batch_loader = batch_loader

                logger = DataLogger(
                    f"../runs/{logging_prefix} BWElo chess {model_size}, {soft_prompt_token_count}, {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
                    accelerator)
                soft_prompt = soft_prompt_creator(soft_prompt_token_count, batch_loader.soft_prompt_parameters_size,
                                                  model.gpt_neox.embed_in.embedding_dim)
                if soft_prompt_parameter_mode == SoftPromptParameterMode.UNCONDITIONAL and \
                        not isinstance(soft_prompt, soft_prompting.DirectSoftPrompt):
                    raise ValueError(
                        f'Unconditional soft prompt parameter mode requires a DirectSoftPrompt, but got {soft_prompt}.')
                optimizer = optim.AdamW(soft_prompt.parameters(), lr=learning_rate, weight_decay=weight_decay)

                training_and_testing.train_and_test_soft_prompt(model, tokenizer, batch_loader, test_batch_loader,
                                                                soft_prompt,
                                                                0, training_step_count,
                                                                AutoregressiveBaseline(),
                                                                optimizer, accelerator, logger,
                                                                forward_test_generated_token_count)
                logger.close()
                try_create_snapshot(snapshot_path_creator, pythia_model_name, soft_prompt_token_count,
                                    maximum_sample_length_in_tokens, batch_lanes_per_step, accumulation_step_count,
                                    soft_prompt, training_step_count, learning_rate, weight_decay)

