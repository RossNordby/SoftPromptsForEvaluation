import torch
from torch import Tensor
from transformers import top_k_top_p_filtering


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


def get_token_counts(input_samples: Tensor, padding_token_id: int) -> Tensor:
    """
    Returns the number of tokens for each lane of the input samples.
    Counts up until encountering the first end of text token or padding token.
    :param input_samples: Sample lanes to count.
                          Dimensions are expected to be [batch_size, sequence_length_in_tokens].
    :param padding_token_id: Token ID of the padding token.
    """
    if input_samples.size(1) == 0:
        return torch.zeros(input_samples.size(0), dtype=torch.long, device=input_samples.device)
    mask = (input_samples == padding_token_id).int()
    first_end_index = mask.argmax(dim=1)
    # If the input contains no padding tokens, then argmax will return 0 for all lanes.
    # We need to set those lanes to the sequence length.
    no_mask_in_lane = ~mask.any(dim=1)
    first_end_index = first_end_index.masked_fill(no_mask_in_lane, input_samples.size(1))
    return first_end_index


