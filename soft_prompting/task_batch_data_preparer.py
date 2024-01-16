from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from soft_prompting import SoftPrompt
from soft_prompting.aliases import EmbedInputFunction, SoftPromptLossFunction


class TaskBatchDataPreparer(ABC):
    @abstractmethod
    def prepare_preparer(self, tokenizer, maximum_sample_length_in_tokens) -> None:
        """
        Prepares the preparer.
        :param tokenizer: The tokenizer to use.
        :param maximum_sample_length_in_tokens: The maximum length of a sample in tokens.
        """
        pass

    @abstractmethod
    def get_batch_data(self, samples: Tensor, soft_prompt_start_indices: Tensor,
                       soft_prompt: SoftPrompt,
                       soft_prompt_parameters: Tensor | None,
                       task_metadata: Any,
                       ids_to_embeddings: EmbedInputFunction,
                       end_of_text_token_id: int, pad_token_id: int) -> (
            Tensor, Tensor, SoftPromptLossFunction | None):
        """
        Gets the data for a batch.
        :param samples: The samples to use.
        :param soft_prompt_start_indices: The start indices for the soft prompts.
        :param soft_prompt: The soft prompt to use.
        :param soft_prompt_parameters: The parameters of the soft prompt, if any.
        :param task_metadata: The metadata for the task, if any.
        :param ids_to_embeddings: The function to use to convert IDs to embeddings.
        :param end_of_text_token_id: The id of the end-of-text token.
        :param pad_token_id: The id of the pad token.
        :return: The input embeddings, output labels, and null loss function.
        """
        pass
