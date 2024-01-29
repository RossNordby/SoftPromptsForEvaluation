from torch import Tensor
import soft_prompting.soft_prompts
from soft_prompting.task_batch_data_preparer import TaskBatchDataPreparer
from soft_prompting import training_and_testing
from soft_prompting.prompted_embeddings_builder import build_prompted_embeddings_for_training


class SkipTokens(TaskBatchDataPreparer):
    def __init__(self, skip_count: int):
        """
        :param skip_count: The number of tokens to skip forward into the future.
        0 is equivalent to the autoregressive baseline.
        """
        self.skip_count = skip_count

    def prepare_preparer(self, tokenizer, maximum_length_in_tokens: int) -> None:
        # No preparation needed.
        pass

    def get_batch_data(self, samples: Tensor, soft_prompt_start_indices: Tensor,
                       soft_prompt: soft_prompting.soft_prompts.SoftPrompt,
                       soft_prompt_parameters: Tensor | None,
                       task_metadata: None,
                       ids_to_embeddings: training_and_testing.EmbedInputFunction,
                       pad_token_id: int) -> (
            Tensor, Tensor, training_and_testing.SoftPromptLossFunction | None):
        input_embeddings, output_labels = (
            build_prompted_embeddings_for_training(samples,
                                                   soft_prompt_start_indices,
                                                   soft_prompt,
                                                   soft_prompt_parameters,
                                                   ids_to_embeddings, pad_token_id,
                                                   output_label_shift=self.skip_count))

        return input_embeddings, output_labels, None
