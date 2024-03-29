from torch import Tensor
import soft_prompting.soft_prompts
from soft_prompting.task_batch_data_preparer import TaskBatchDataPreparer
from soft_prompting import training_and_testing
from soft_prompting.prompted_embeddings_builder import build_prompted_embeddings_for_training, \
    DEFAULT_OUTPUT_LABEL_MASK_ID


class PathfindingBatchPreparer(TaskBatchDataPreparer):
    def prepare_preparer(self, tokenizer, maximum_length_in_tokens: int) -> None:
        # No preparation needed.
        pass

    def get_batch_data(self, samples: Tensor, soft_prompt_start_indices: Tensor,
                       soft_prompt: soft_prompting.soft_prompts.SoftPrompt,
                       soft_prompt_parameters: Tensor | None,
                       move_tokens_start_indices: list[int],
                       ids_to_embeddings: training_and_testing.EmbedInputFunction,
                       end_of_text_token_id: int, pad_token_id: int) -> (
            Tensor, Tensor, training_and_testing.SoftPromptLossFunction | None):
        """
        Gets the data for the autoregressive baseline test.
        :param samples: The samples to use.
        :param soft_prompt_start_indices: The start indices for the soft prompts.
        :param soft_prompt: The soft prompt to use.
        :param soft_prompt_parameters: The parameters of the soft prompt, if any.
        :param move_tokens_start_indices: The start indices for the move tokens sequence in the samples tensor.
        :param ids_to_embeddings: The function to use to convert IDs to embeddings.
        :param end_of_text_token_id: The id of the end-of-text token.
        :param pad_token_id: The id of the pad token.
        :return: The input embeddings, output labels, and null loss function.
        """
        # Zero shift used for building prompts; the model's default loss performs shifting.
        input_embeddings, output_labels = (
            build_prompted_embeddings_for_training(samples,
                                                   soft_prompt_start_indices,
                                                   soft_prompt,
                                                   soft_prompt_parameters,
                                                   ids_to_embeddings, end_of_text_token_id, pad_token_id))

        # We'll be using the default loss function, but we need to mask out the loss from all non-move tokens.
        for i in range(samples.size(0)):
            output_labels[i, :move_tokens_start_indices[i]] = DEFAULT_OUTPUT_LABEL_MASK_ID

        return input_embeddings, output_labels, None
