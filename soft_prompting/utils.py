import os

import torch
from torch import Tensor
from transformers import top_k_top_p_filtering

from soft_prompting import SnapshotPathCreator, SoftPrompt


def sample_token_from_logits(output_logits: torch.Tensor, top_k=0, top_p: float = 0.7) -> torch.Tensor:
    """
    Samples tokens from the given logits.
    :param output_logits: Logits to sample tokens from.
    :param top_k: Number of tokens to sample from for top-k sampling. If zero, top-k sampling is not used.
    :param top_p: Probability threshold for top-p sampling. If one or higher, top-p sampling is not used.
    """
    flattened_logits = output_logits.view(-1, output_logits.size(-1))
    warped_output_logits = top_k_top_p_filtering(flattened_logits, top_k=top_k, top_p=top_p)
    probabilities = torch.nn.functional.softmax(warped_output_logits, dim=-1)
    sampled = torch.multinomial(probabilities, num_samples=1)
    # Note that we change the view to match the original output_logits shape with the vocab dimension removed.
    # Between that and the earlier flattening, this function supports both single lane and batched logits.
    return sampled.view(output_logits.size()[0:-1])


def devices_match(a: torch.device, b: torch.device) -> bool:
    """
    Returns true if the given devices refer to the same device.
    This is distinct from a pure equality check because torch.device('cuda') == torch.device('cuda:0') is false.
    :param a: First device.
    :param b: Second device.
    :return: True if the given devices refer to the same device.
    """
    if a.type != b.type:
        return False
    if a.type == 'cuda':
        index_a = torch.cuda.current_device() if a.index is None else a.index
        index_b = torch.cuda.current_device() if b.index is None else b.index
        return index_a == index_b
    # CPU devices don't have indices.
    return True


def get_token_counts(input_samples: Tensor, end_of_text: int, padding: int) -> Tensor:
    """
    Returns the number of tokens for each lane of the input samples.
    Counts up until encountering the first end of text token or padding token.
    :param input_samples: Sample lanes to count.
                          Dimensions are expected to be [batch_size, sequence_length_in_tokens].
    :param end_of_text: Token ID of the end of text token.
    :param padding: Token ID of the padding token.
    """
    if input_samples.size(1) == 0:
        return torch.zeros(input_samples.size(0), dtype=torch.long, device=input_samples.device)
    mask = ((input_samples == end_of_text) | (input_samples == padding)).int()
    first_end_index = mask.argmax(dim=1)
    # If the input contains no eot or padding tokens, then argmax will return 0 for all lanes.
    # We need to set those lanes to the sequence length.
    no_mask_in_lane = ~mask.any(dim=1)
    first_end_index = first_end_index.masked_fill(no_mask_in_lane, input_samples.size(1))
    return first_end_index


def try_create_snapshot(snapshot_path_creator: SnapshotPathCreator | None, model_name: str,
                        soft_prompt_token_count: int,
                        maximum_sample_length_in_tokens: int, batch_lanes_per_step: int, accumulation_step_count: int,
                        soft_prompt: SoftPrompt, training_step_count: int, learning_rate: float, weight_decay: float):
    """
    Attempts to create a snapshot for the given soft prompt if the snapshot path creator is not None.
    :param snapshot_path_creator: Function which creates paths to save snapshots to.
    :param model_name: The name of the model used for training.
    :param soft_prompt_token_count: The number of tokens in the soft prompt.
    :param maximum_sample_length_in_tokens: The maximum sample length in tokens pulled from the dataset.
    :param batch_lanes_per_step: The number of batch lanes to use per optimization step.
    :param accumulation_step_count: The number of accumulation steps used in training.
    :param soft_prompt: The soft prompt to save.
    :param training_step_count: The number of training steps used in training.
    :param learning_rate: The learning rate used in training.
    :param weight_decay: The weight decay used in training.
    """
    if snapshot_path_creator is not None:
        snapshot_path = snapshot_path_creator(model_name, soft_prompt_token_count)
        os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
        torch.save((soft_prompt.state_dict(), {
            'model_name': model_name,
            'soft_prompt_type': soft_prompt.__class__.__name__,
            'soft_prompt_metadata': soft_prompt.get_metadata(),
            'training_metadata': {
                'training_step_index': training_step_count,
                'accumulation_step_count': accumulation_step_count,
                'batch_lanes_per_step': batch_lanes_per_step,
                'maximum_sample_length_in_tokens': maximum_sample_length_in_tokens,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay
            }
        }), snapshot_path)
