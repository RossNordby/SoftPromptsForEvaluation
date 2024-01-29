from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import PrecisionType
from torch import optim
from transformers import AutoTokenizer, AutoModelForCausalLM

import soft_prompting.direct_soft_prompt
from pathfinding.async_pathfinding_loader import AsyncPathfindingLoader
from pathfinding.pathfinding_batch_loader import PathfindingBatchLoader
from pathfinding.pathfinding_batch_preparer import PathfindingBatchPreparer
from soft_prompting import training_and_testing, SoftPromptFactory
from soft_prompting.data_logger import DataLogger
from soft_prompting.training_callbacks import TrainingCallbacks


def train_and_test_pathfinding(board_width: int, board_height: int,
                               model_configurations: list[tuple[str, int]],
                               soft_prompt_token_counts: list[int],
                               soft_prompt_creator: SoftPromptFactory,
                               insert_spaces: bool,
                               insert_moves_section_separator: bool,
                               logging_prefix: str = "",
                               training_step_count: int = 512, batch_lanes_per_step: int = 32,
                               maximum_sample_length_in_tokens: int = 256, learning_rate: float = 1e-3,
                               weight_decay: float = 1e-4,
                               forward_test_generated_token_count: int = 128,
                               force_mixed_precision_mode: PrecisionType = None,
                               training_callbacks: TrainingCallbacks | None = None) -> None:
    """
    Trains soft prompts on the pathfinding dataset.
    Trains a soft prompt for each model configuration for each soft prompt token count.

    :param board_width: The width of the board to use.
    :param board_height: The height of the board to use.
    :param model_configurations: A list of tuples containing the model name and the number of accumulation steps to use.
                                 Note that this differs from the chess_training.train_and_test function; the
                                 full model name is expected, not just the size substring.
    :param soft_prompt_token_counts: A list of soft prompt token counts to train.
    :param soft_prompt_creator: A function which creates a soft prompt for a training scenario.
                                Note: Currently, language tasks only support DirectSoftPrompts.
    :param insert_spaces: Whether to insert spaces between the board slots and the moves.
                          May affect tokenization.
    :param insert_moves_section_separator: Whether to insert a line that says 'Moves:' between the board and moves.
    :param logging_prefix: The prefix to prepend to the name of tensorboard log outputs.
    :param training_step_count: The number of training steps to use.
    :param batch_lanes_per_step: The number of batch lanes to use per optimization step. Must be a multiple of
                                 the accumulation step count defined in any configuration defined in
                                 model_configurations.
    :param maximum_sample_length_in_tokens: The maximum sample length in tokens pulled from the database.
    :param learning_rate: The learning rate to use.
    :param weight_decay: The weight decay to use.
    :param forward_test_generated_token_count: The number of tokens to generate when doing forward generation testing.
    :param force_mixed_precision_mode: What kind of mixed precision training to force, if any.
                                       If None, will use the default specified by the accelerator.
    :param training_callbacks: Callbacks to call during training.
    """

    for model_name, accumulation_step_count in model_configurations:
        torch.manual_seed(5)

        print(f'Preparing model {model_name}...')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        embedding_width = model.get_input_embeddings().embedding_dim
        # Forcing a pad token id of 0; it works for the pythia and tinyllama models we're using.
        # This will NOT work universally.
        tokenizer.pad_token = 0

        # Note that training step counts refer to the number of optimization steps, not the number of batches.
        batch_size = batch_lanes_per_step // accumulation_step_count
        if batch_size * accumulation_step_count != batch_lanes_per_step:
            raise ValueError(f'Batch lanes per step ({batch_lanes_per_step}) must be a multiple of '
                             f'accumulation step count ({accumulation_step_count}).')
        accelerator = Accelerator(gradient_accumulation_steps=accumulation_step_count,
                                  mixed_precision=force_mixed_precision_mode)

        for soft_prompt_token_count in soft_prompt_token_counts:
            with AsyncPathfindingLoader(board_width, board_height, insert_spaces) as board_loader:
                batch_loader = PathfindingBatchLoader(board_loader, insert_moves_section_separator, tokenizer,
                                                      batch_size, maximum_sample_length_in_tokens,
                                                      num_processes=accelerator.num_processes,
                                                      process_index=accelerator.process_index)

                # Reusing the same batch loader for both. S'fine; won't be the same data.
                test_batch_loader = batch_loader

                logger_safe_model_name = model_name.replace('/', '_')
                logger = DataLogger(
                    f"runs/{logging_prefix} pathfinding {board_width}x{board_height} {logger_safe_model_name}, {soft_prompt_token_count}, {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
                    accelerator)
                soft_prompt = soft_prompt_creator(soft_prompt_token_count, batch_loader.soft_prompt_parameters_size,
                                                  embedding_width)
                if isinstance(soft_prompt, soft_prompting.DirectSoftPrompt):
                    raise ValueError(
                        f'Pathfinding requires a conditional soft prompt, but got {soft_prompt}.')
                optimizer = optim.AdamW(soft_prompt.parameters(), lr=learning_rate, weight_decay=weight_decay)

                training_and_testing.train_and_test_soft_prompt(model, model_name, tokenizer, batch_loader,
                                                                test_batch_loader, soft_prompt,
                                                                0, training_step_count,
                                                                PathfindingBatchPreparer(),
                                                                optimizer, accelerator, logger,
                                                                forward_test_generated_token_count,
                                                                training_callbacks=training_callbacks)

                logger.close()
