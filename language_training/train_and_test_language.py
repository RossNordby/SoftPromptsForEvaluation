from datetime import datetime
from typing import Iterable

import torch
from accelerate import Accelerator
from accelerate.utils import PrecisionType
from datasets import load_dataset
from torch import optim
from transformers import AutoTokenizer, AutoModelForCausalLM

import soft_prompting.direct_soft_prompt
from language_training.dataset_iterables import DatasetIterable
from language_training.persistent_batch_loader import PersistentBatchLoader
from soft_prompting import training_and_testing, SoftPromptFactory, TaskBatchDataPreparer
from soft_prompting.data_logger import DataLogger
from soft_prompting.training_callbacks import TrainingCallbacks


def train_and_test_language(model_configurations: list[tuple[str, int]],
                            soft_prompt_token_counts: list[int],
                            dataset_iterables: list[DatasetIterable],
                            soft_prompt_creator: SoftPromptFactory,
                            batch_data_preparer: TaskBatchDataPreparer,
                            maximum_soft_prompt_start_indices: int | None = None,
                            logging_prefix: str = "",
                            training_step_count: int = 512, batch_lanes_per_step: int = 32,
                            maximum_sample_length_in_tokens: int = 256, learning_rate: float = 1e-3,
                            weight_decay: float = 1e-4,
                            forward_test_generated_token_count: int = 128,
                            force_mixed_precision_mode: PrecisionType = None,
                            training_callbacks: TrainingCallbacks | None = None) -> None:
    """
    Trains soft prompts on the english subset of the redpajama v2 language dataset.
    Trains a soft prompt for each model configuration for each soft prompt token count.

    :param model_configurations: A list of tuples containing the model name and the number of accumulation steps to use.
                                 Note that this differs from the chess_training.train_and_test function; the
                                 full model name is expected, not just the size substring.
    :param soft_prompt_token_counts: A list of soft prompt token counts to train.
    :param dataset_iterables: A list of dataset iterables to use for training. An iterable's __next__ should return
                              a string containing the raw content of a sample, and __iter__ should return a fresh
                              iterator over the dataset.
    :param soft_prompt_creator: A function which creates a soft prompt for a training scenario.
                                Note: Currently, language tasks only support DirectSoftPrompts.
    :param batch_data_preparer: The data preparer to use for soft prompt training.
    :param maximum_soft_prompt_start_indices: The maximum index that soft prompts can be inserted at
                                              within a sample. If None, no maximum will be enforced.
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
        # All language tasks will use the redpajama dataset.
        # Reload the dataset for each model configuration
        torch.manual_seed(5)

        print(f'Preparing model {model_name}...')
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        embedding_width = model.get_input_embeddings().embedding_dim
        # Forcing a pad token id of 0; it works for the pythia and tinyllama models we're using.
        # This will NOT work universally.
        tokenizer.pad_token_id = 0

        # Note that training step counts refer to the number of optimization steps, not the number of batches.
        batch_size = batch_lanes_per_step // accumulation_step_count
        if batch_size * accumulation_step_count != batch_lanes_per_step:
            raise ValueError(f'Batch lanes per step ({batch_lanes_per_step}) must be a multiple of '
                             f'accumulation step count ({accumulation_step_count}).')
        accelerator = Accelerator(gradient_accumulation_steps=accumulation_step_count,
                                  mixed_precision=force_mixed_precision_mode)

        class DatasetWrapper:
            def __init__(self, dataset):
                self.dataset = dataset
                self.iterator = iter(dataset)

            def __iter__(self):
                return self

            def __next__(self):
                return None, tokenizer(next(self.iterator), return_tensors='pt')['input_ids'].squeeze()

        for soft_prompt_token_count in soft_prompt_token_counts:
            for dataset in dataset_iterables:
                iterator = DatasetWrapper(dataset)
                batch_loader = PersistentBatchLoader(iterator, batch_size, maximum_sample_length_in_tokens,
                                                     tokenizer.pad_token_id, 0,
                                                     num_processes=accelerator.num_processes,
                                                     process_index=accelerator.process_index)

                # Reusing the same batch loader for both. S'fine; won't be the same data.
                test_batch_loader = batch_loader

                logger_safe_model_name = model_name.replace('/', '_')
                logger = DataLogger(
                    f"runs/{logging_prefix} language {logger_safe_model_name}, {soft_prompt_token_count}, {dataset.name}, {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}",
                    accelerator)
                soft_prompt = soft_prompt_creator(soft_prompt_token_count, batch_loader.soft_prompt_parameters_size,
                                                  embedding_width)
                if not isinstance(soft_prompt, soft_prompting.DirectSoftPrompt):
                    raise ValueError(
                        f'Unconditional soft prompt parameter mode requires a DirectSoftPrompt, but got {soft_prompt}.'
                        f'Unlike chess training, no conditional modes exist for language training yet!')
                optimizer = optim.AdamW(soft_prompt.parameters(), lr=learning_rate, weight_decay=weight_decay)

                training_and_testing.train_and_test_soft_prompt(model, model_name, tokenizer, batch_loader,
                                                                test_batch_loader, soft_prompt,
                                                                maximum_soft_prompt_start_indices, training_step_count,
                                                                batch_data_preparer, optimizer, accelerator, logger,
                                                                forward_test_generated_token_count,
                                                                training_callbacks=training_callbacks)
                logger.close()
