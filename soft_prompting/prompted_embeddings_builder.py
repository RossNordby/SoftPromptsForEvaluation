from typing import Callable
import torch

from soft_prompting.soft_prompts import SoftPrompt
from soft_prompting import utils


def build_prompted_embeddings(raw_embeddings: torch.Tensor, soft_prompt_start_indices: torch.Tensor | None,
                              soft_prompt: SoftPrompt,
                              soft_prompt_parameters: torch.Tensor | None) -> torch.Tensor:
    """
    Builds a batch of token embeddings with prompts inserted for training.
    :param raw_embeddings: Raw embeddings to insert the soft prompt into.
    :param soft_prompt_start_indices: Token indices at which to insert the soft prompt. If None, the soft prompt will be
                                      inserted at the beginning of the sequence. The caller should guarantee that the
                                      insertion indices are valid with respect to both the batch size and the number of
                                      tokens actually present within each batch lane.
                                      No validation to avoid inserting soft prompts into padding is performed.
    :param soft_prompt: Soft prompt to insert into the embeddings.
    :param soft_prompt_parameters: Parameters for the soft prompt, if any.
                                   Soft prompts which do not require parameters will ignore this.
    :return: Input embeddings with soft prompt tokens inserted.
    """
    batch_size = raw_embeddings.size(0)
    if (soft_prompt_start_indices is not None and batch_size != soft_prompt_start_indices.size(0)) or (
            soft_prompt_parameters is not None and batch_size != soft_prompt_parameters.size(0)):
        raise ValueError("Batch size doesn't match across inputs.")
    if soft_prompt_parameters is None and soft_prompt.input_parameter_count > 0:
        raise ValueError("Soft prompt requires tensor input but no parameters were provided.")
    # The output should be a set of embeddings across all batches which preserves the computational graph's
    # connection to the soft prompt parameters. Soft prompts that are directly optimized need no further effort,
    # but soft prompts which include a generating model will require a forward pass.
    soft_prompt_embeddings = soft_prompt.forward(
        soft_prompt_parameters if soft_prompt.input_parameter_count > 0 else batch_size)
    to_stack: list[torch.Tensor] = []
    # Note: if you knew the batches have equal size, or if you were willing to accept a bounded set of possible
    # prompt locations, this python loop could be punted entirely into a tensor-wide split+cat.
    # Could be worth it if we push the batch size.
    # The randomization allowed by per-lane start indices isn't that significant over the course of an epoch,
    # provided a sufficiently diverse dataset.
    if soft_prompt_start_indices is None:
        for i in range(batch_size):
            to_stack.append(torch.cat([soft_prompt_embeddings[i], raw_embeddings[i]], dim=0))
    else:
        for i in range(batch_size):
            assert soft_prompt_start_indices.size(0) == batch_size
            start_index = soft_prompt_start_indices[i]
            if start_index <= 0:
                combined = torch.cat([soft_prompt_embeddings[i], raw_embeddings[i]], dim=0)
            elif start_index >= raw_embeddings.size(1):
                combined = torch.cat([raw_embeddings[i], soft_prompt_embeddings[i]], dim=0)
            else:
                a = raw_embeddings[i][:start_index]
                b = raw_embeddings[i][start_index:]
                combined = torch.cat([a, soft_prompt_embeddings[i], b], dim=0)
            to_stack.append(combined)
    return torch.stack(to_stack)


DEFAULT_OUTPUT_LABEL_MASK_ID = -100


def build_prompted_embeddings_for_training(sample_token_ids: torch.Tensor,
                                           soft_prompt_start_indices: torch.Tensor | None,
                                           soft_prompt: SoftPrompt,
                                           soft_prompt_parameters: torch.Tensor | None,
                                           ids_to_embeddings: Callable[[torch.Tensor], torch.Tensor],
                                           end_of_text_token_id: int,
                                           pad_token_id: int,
                                           mask_before_soft_prompt: bool = True,
                                           output_label_mask_id: int = DEFAULT_OUTPUT_LABEL_MASK_ID,
                                           output_label_shift: int = 0) -> (torch.Tensor, torch.Tensor):
    """
    Builds a batch of token embeddings with prompts inserted for training, plus the expected output ids.
    :param sample_token_ids: Sample tokens to build embeddings for.
                             Dimensions of the tensor are interpreted as [batch_size, token_count].
                             Slots not occupied by a token are expected to be filled with padding tokens.
    :param soft_prompt_start_indices: Token indices at which to insert the soft prompt.
                                       If None, the soft prompt will be inserted at the beginning of the sequence.
                                       The caller should guarantee that the insertion indices are valid with respect to
                                       both the batch size and the number of tokens actually present within each
                                       batch lane. No validation to avoid inserting soft prompts into padding is
                                       performed.
    :param soft_prompt: Soft prompt to insert into the embeddings.
    :param soft_prompt_parameters: Parameters for the soft prompt, if any.
                                   Soft prompts which do not require parameters will ignore this.
    :param ids_to_embeddings: Function which converts token ids to embeddings.
    :param end_of_text_token_id: Token id of end-of-text tokens.
    :param pad_token_id: Token id of pad tokens.
    :param mask_before_soft_prompt: Whether to mask out the loss contributions for all tokens before the soft prompt.
                                    Defaults to True; makes no difference for autoregressive models.
    :param output_label_mask_id: Token id to use for output labels of soft prompt tokens, pad tokens,
                                 and end-of-text tokens.
    :param output_label_shift: Number of tokens to shift the output labels by. Defaults to 0.
    :return: Tuple containing the prompted embeddings and the expected output ids.
    """
    # For training, the input ids are the same as the output ids, except shifted by however much.
    if output_label_shift > 0:
        input_ids = sample_token_ids[:, :-output_label_shift]
        unprompted_output_ids = sample_token_ids[:, output_label_shift:]
    else:
        input_ids = sample_token_ids
        unprompted_output_ids = sample_token_ids
    # The unprompted ids will be modified in-place, so clone them.
    unprompted_output_ids = unprompted_output_ids.clone()
    raw_embeddings = ids_to_embeddings(input_ids)
    prompted_embeddings = build_prompted_embeddings(raw_embeddings, soft_prompt_start_indices, soft_prompt,
                                                    soft_prompt_parameters)
    # We'll want to mask out the loss contributions for all pad, eot, and soft prompt tokens.
    # First, find where the pad/eot tokens start.
    token_counts = utils.get_token_counts(input_ids, end_of_text_token_id, pad_token_id)
    # Mask out the pad/eot tokens for all lanes.
    batch_size = raw_embeddings.size(0)
    for i in range(batch_size):
        unprompted_output_ids[i, token_counts[i]:] = output_label_mask_id
        if mask_before_soft_prompt:
            # Mask out the soft prompt tokens before the soft prompt.
            unprompted_output_ids[i, :soft_prompt_start_indices[i].item()] = \
                output_label_mask_id
    # Expected output ids need to take into account the soft prompt tokens.
    masked_output_region_for_soft_prompt_token = torch.empty([soft_prompt.soft_prompt_token_count], dtype=torch.long,
                                                             device=sample_token_ids.device)
    masked_output_region_for_soft_prompt_token[:] = output_label_mask_id
    to_stack: list[torch.Tensor] = []
    if soft_prompt_start_indices is None:
        for i in range(batch_size):
            to_stack.append(torch.cat([masked_output_region_for_soft_prompt_token, unprompted_output_ids[i]], dim=0))
    else:
        for i in range(batch_size):
            start_index = soft_prompt_start_indices[i]
            if start_index <= 0:
                combined = torch.cat([masked_output_region_for_soft_prompt_token, unprompted_output_ids[i]], dim=0)
            elif start_index >= unprompted_output_ids.size(1):
                combined = torch.cat([unprompted_output_ids[i], masked_output_region_for_soft_prompt_token], dim=0)
            else:
                a = unprompted_output_ids[i][:start_index]
                b = unprompted_output_ids[i][start_index:]
                combined = torch.cat([a, masked_output_region_for_soft_prompt_token, b], dim=0)
            to_stack.append(combined)
    expected_output_ids = torch.stack(to_stack)
    return prompted_embeddings, expected_output_ids
