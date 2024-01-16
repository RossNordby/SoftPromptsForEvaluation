from typing import Any

from torch import Tensor

from soft_prompting.task_batch_data_preparer import TaskBatchDataPreparer
from soft_prompting.prompted_embeddings_builder import build_prompted_embeddings_for_training
import soft_prompting.training_and_testing as training_and_testing
import soft_prompting.soft_prompts as soft_prompts


class TrivialRepetitionTest(TaskBatchDataPreparer):
    def __init__(self, string_to_repeat: str):
        self.string_to_repeat = string_to_repeat
        self.repeated_tokens = None

    def prepare_preparer(self, tokenizer, maximum_sample_length_in_tokens: int) -> None:
        # The string needs to be tokenized to use it, but __init__ didn't have access to a tokenizer.
        tokens_to_repeat = tokenizer(self.string_to_repeat, return_tensors='pt')['input_ids'].squeeze(0)
        repeat_count = maximum_sample_length_in_tokens // len(tokens_to_repeat) + 1
        self.repeated_tokens = tokens_to_repeat.repeat(repeat_count)

    def get_batch_data(self, samples: Tensor, soft_prompt_start_indices: Tensor,
                       soft_prompt: soft_prompts.SoftPrompt,
                       soft_prompt_parameters: Tensor | None,
                       task_metadata: Any,
                       ids_to_embeddings: training_and_testing.EmbedInputFunction,
                       end_of_text_token_id: int, pad_token_id: int) -> (
            Tensor, Tensor, training_and_testing.SoftPromptLossFunction | None):
        """
        Gets the data for the trivial repetition test.
        :param samples: The samples to use.
        :param soft_prompt_start_indices: The start indices for the soft prompts.
        :param soft_prompt: The soft prompt to use.
        :param soft_prompt_parameters: The parameters of the soft prompt, if any.
        :param task_metadata: The metadata for the task; should be None for this type.
        :param ids_to_embeddings: The function to use to convert IDs to embeddings.
        :param end_of_text_token_id: The id of the end-of-text token.
        :param pad_token_id: The id of the pad token.
        :return: The input embeddings, output labels, and a null loss function.
        """
        if self.repeated_tokens is None:
            raise ValueError('prepare_preparer must be called before get_data_for_batch.')
        input_embeddings, output_labels = build_prompted_embeddings_for_training(samples,
                                                                                 soft_prompt_start_indices,
                                                                                 soft_prompt,
                                                                                 soft_prompt_parameters,
                                                                                 ids_to_embeddings,
                                                                                 end_of_text_token_id,
                                                                                 pad_token_id)
        # The current dataset doesn't have any soft-prompt related effects in it,
        # so we have to insert the effect ourselves.
        self.repeated_tokens.to(output_labels.device)
        for j in range(samples.size(0)):
            # Note that we don't have to mask anything for the area before the soft prompt; it was already masked.
            labels_start = soft_prompt_start_indices[j] + soft_prompt.soft_prompt_token_count
            output_labels[j, labels_start:] = self.repeated_tokens[:output_labels.size(1) - labels_start]

        return input_embeddings, output_labels, None
