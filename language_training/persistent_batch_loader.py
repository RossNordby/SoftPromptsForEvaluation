from typing import Generator, Iterator

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from soft_prompting.batch_loader import BatchLoader


class PersistentBatchLoader(BatchLoader):
    """
    Loads batches of data from a dataset by tracking individual lanes submitted by an underlying dataset.

    Attributes:
        pad_token_id (int): The ID of the padding token.

        lane_dataset (Generator[(str), None, None]): The dataset to pull lanes of data from.
        tokens_sampled_count (int):
            Number of tokens pulled from the BatchLoader across its lifespan.
            Includes tokens that were discarded because they were handled by another process; it is a global count
            if the underlying dataset is deterministic across all processes.
        lanes (list[tuple[torch.Tensor, torch.Tensor, int]]):
            Tracked batch lanes in the loader. Each lane is a tuple containing:
            - A tensor containing the soft prompt parameters for the lane.
            - A tensor containing the tokenized string for the lane.
            - The index of the next token to pull from the lane.
    """

    def __init__(self, lane_dataset: Iterator[tuple[Tensor | None, Tensor]],
                 batch_size: int, sample_length_in_tokens: int,
                 pad_token_id: int, soft_prompt_input_parameter_count: int,
                 discard_truncated_samples: bool = False, num_processes: int = 1, process_index: int = 0):
        """
        Creates a BatchLoader.
        :param lane_dataset: The dataset to pull data lanes from. Should return a tuple containing:
                             - A tensor containing the soft prompt parameters for the lane, or None if no soft prompt
                             parameters are needed.
                             - A tensor containing the tokenized string for the lane.
        :param batch_size: The number of samples to request across all parallel processes.
                           The returned number of samples may be smaller than requested if multiple processes are used.
        :param sample_length_in_tokens: The length of each sample in tokens.
                                        If insufficient data is available, the actual sample length may be shorter.
        :param pad_token_id: The ID of the padding token.
        :param soft_prompt_input_parameter_count: The number of parameters in the soft prompt.
        :param discard_truncated_samples: Whether to discard the content of samples beyond the target sample length.
                                          If true, only the first target_sample_length tokens of each sample will be
                                          returned and the rest will be discarded.
                                          Discarded tokens will not be used in future batches.
                                          If false, samples loaded from the underlying dataset will persist in the batch
                                          lane and can be consumed by future requests.
        :param num_processes: The number of processes to split the batch across.
        :param process_index: The index of this process.
        """
        super().__init__(batch_size, sample_length_in_tokens, num_processes, process_index)
        self.pad_token_id = pad_token_id
        self.soft_prompt_input_parameter_count = soft_prompt_input_parameter_count
        self.lane_dataset: Iterator[tuple[Tensor | None, Tensor]] = lane_dataset
        self.lanes: list[tuple[torch.Tensor, torch.Tensor, int] | None] = []
        self.tokens_sampled_count: int = 0
        self.discard_truncated_samples = discard_truncated_samples
        self.num_processes = num_processes
        self.process_index = process_index

    def __next__(self):
        """
        Requests samples from the batch loader.
        :return: A tuple containing (None, tensor of samples, None).
        """
        # Fill in any lanes that lack tokens.
        while len(self.lanes) < self.batch_size:
            self.lanes.append((*next(self.lane_dataset), 0))

        soft_parameters = None if self.soft_prompt_input_parameter_count == 0 else (
            torch.empty([self.batch_size, self.soft_prompt_input_parameter_count], dtype=torch.float32))
        target_samples = torch.empty([self.batch_size, self.sample_length_in_tokens], dtype=torch.long)
        maximum_observed_sample_length = 0
        for i in range(self.batch_size):
            lane = self.lanes[i]
            if lane is None or lane[2] == lane[1].size(0):
                # Already reached the end of this lane (or the lane was discarded); pull new data.
                lane = (*next(self.lane_dataset), 0)
            soft_prompt_parameters = lane[0]
            source_tokens = lane[1]
            start_index = lane[2]
            if soft_parameters is not None:
                assert (soft_prompt_parameters is not None and soft_prompt_parameters.size(0) ==
                        self.soft_prompt_input_parameter_count), \
                    "All lanes must have appropriate input parameters for the soft prompt."
                soft_parameters[i] = soft_prompt_parameters
            end_index = min(source_tokens.size(0), start_index + self.sample_length_in_tokens)
            token_count = end_index - start_index
            maximum_observed_sample_length = max(maximum_observed_sample_length, token_count)
            target_samples_lane = target_samples[i]
            target_samples_lane[:token_count] = source_tokens[start_index:end_index]
            target_samples_lane[token_count:] = self.pad_token_id
            self.lanes[i] = None if self.discard_truncated_samples \
                else (soft_prompt_parameters, source_tokens, end_index)

        # For some use cases, we may want to specify a large maximum sample length with the expectation
        # that almost all samples will be far smaller. If all batch lanes are shorter than the target sample length,
        # we can simply truncate the batch to the maximum observed sample length.
        target_samples = target_samples[:, :maximum_observed_sample_length]
        if self.num_processes > 1:
            # Simply drop the parts of the batch which are not needed by this process.
            # Yes, this is a bit wasteful, but doing *exactly* the same thing redundantly makes it a lot easier
            # to ensure deterministic behavior (if the underlying dataset is deterministic).
            lanes_per_process = self.batch_size // self.num_processes
            assert lanes_per_process * self.num_processes == self.batch_size, \
                "Batch size must be divisible by the number of processes."
            lane_start_index = lanes_per_process * self.process_index
            lane_end_index = lane_start_index + lanes_per_process
            return soft_parameters[lane_start_index:lane_end_index], target_samples[lane_start_index:lane_end_index]
        else:
            return None, target_samples, None

    def request_string_samples(self, tokenizer: PreTrainedTokenizerBase):
        """
        Requests samples from the batch loader in string format.
        :param tokenizer: The tokenizer to use to decode the samples.
        :return: A list of sample strings.
        """
        # Roundtripping tokenization is a bit doofy, but that's fine. Can fix it later! Or not!
        soft_prompt_parameters, samples = self.__next__()
        return [tokenizer.decode(sample) for sample in samples]

    @property
    def soft_prompt_parameters_size(self):
        return 0
