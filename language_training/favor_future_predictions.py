import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

from soft_prompting.task_batch_data_preparer import TaskBatchDataPreparer
from soft_prompting import soft_prompts, training_and_testing
from soft_prompting.prompted_embeddings_builder import build_prompted_embeddings_for_training


class FavorFutureTest(TaskBatchDataPreparer):
    def __init__(self, token_lookahead_count: int):
        self.token_lookahead_count = token_lookahead_count

    def compute_loss(self, logits, output_labels, soft_prompt, soft_prompt_start_indices):
        # Favoring future predictions means we want the model to focus on the quality of predictions N tokens ahead.
        # This isn't the same thing as shifting the labels N tokens ahead, because the model is still predicting
        # the next token. We just want it to output next tokens that make future tokens easier to predict.
        # Each token's responsibility is the loss for tokens from N tokens ahead to the end of the sequence.
        # token_loss(batch_index, token_index) =
        # loss(outputs[batch_index, token_index + N:], labels[batch_index, token_index + N:])
        # Summing across all tokens in a batch lane:
        # batch_lane_loss(batch_lane_index) =
        #   sum(token_loss(batch_lane_index, token_index)
        #   for token_index in range(soft_prompt_start_indices[batch_index] + soft_prompt.token_lookahead_count,
        #                            sequence_length - N))
        # And the total loss =
        #   sum(batch_loss(batch_index) for batch_index in range(batch_size))
        # The naive implementation is going to be very slow.
        # It's got nested loops and re-evaluates distribution-target losses for tokens over and over again.
        # Instead, format the logits to be of shape [batch_size * token_count, vocab_size] and the labels to be of
        # size [batch_size * token_count]. Then we can use the built-in CrossEntropyLoss with reduction='none'.
        batch_size = logits.size(0)
        token_count = logits.size(1)
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = output_labels.view(-1)
        loss_function = CrossEntropyLoss(reduction='none')
        token_losses_flat = loss_function(logits_flat, labels_flat.to(logits.device))
        token_losses = token_losses_flat.view(batch_size, token_count)
        # Create a tensor of the same size as token_losses containing the start of the post-prompt region
        # for each batch, broadcast.
        start_indices_broadcast: torch.Tensor = (
                soft_prompt_start_indices + soft_prompt.soft_prompt_token_count + self.token_lookahead_count).unsqueeze(
            1).expand(-1, token_count)
        # Create another tensor of the same size as token_losses containing the token indices for each slot.
        token_indices_matrix: torch.Tensor = torch.arange(token_count, device=logits.device).unsqueeze(0).expand(
            batch_size, -1)
        contributing_token_count = torch.clamp(token_indices_matrix - start_indices_broadcast, min=0)
        scaled_losses = token_losses * contributing_token_count.to(logits.device)
        loss = torch.sum(scaled_losses) / torch.sum(contributing_token_count)
        return loss

    def prepare_preparer(self, tokenizer, maximum_length_in_tokens: int) -> None:
        # No preparation needed.
        pass

    def get_batch_data(self, samples: Tensor, soft_prompt_start_indices: Tensor,
                       soft_prompt: soft_prompts.SoftPrompt,
                       soft_prompt_parameters: Tensor | None,
                       task_metadata: None,
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
        :return: The input embeddings, output labels, and loss function.
        """
        # Note that we use a nonzero output label shift. Our compute loss function doesn't do the shift internally
        # like the HF GPT_NeoX/llama's application of CrossEntropyLoss does.
        input_embeddings, output_labels = (
            build_prompted_embeddings_for_training(samples,
                                                   soft_prompt_start_indices,
                                                   soft_prompt,
                                                   soft_prompt_parameters,
                                                   ids_to_embeddings, end_of_text_token_id, pad_token_id,
                                                   output_label_shift=1))

        return input_embeddings, output_labels, self.compute_loss
