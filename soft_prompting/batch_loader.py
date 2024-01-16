from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor


class BatchLoader(ABC):

    def __init__(self, batch_size: int, maximum_sample_length_in_tokens: int, num_processes: int, process_index: int):
        """
        Initializes fields common to batch loaders.
        :param batch_size: The number of samples to request across all parallel processes.
                           The returned number of samples may be smaller than requested if multiple processes are used.
        :param maximum_sample_length_in_tokens: The maximum length of each sample in tokens.
                                                If insufficient data is available,
                                                the actual sample length may be shorter.
        :param num_processes: The number of processes to split the batch across.
        :param process_index: The index of this process.
        """
        if batch_size // num_processes == 0:
            raise ValueError(
                f"Batch size (currently {batch_size}) must be at least the number of processes "
                f"used in distributed execution (currently {num_processes}).")
        self.batch_size = batch_size
        self.sample_length_in_tokens = maximum_sample_length_in_tokens
        self.num_processes = num_processes
        self.process_index = process_index

    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self) -> tuple[Tensor, Tensor, Any]:
        """
        Requests samples from the batch loader.
        :return: A tuple containing (tensor of soft prompt parameters, tensor of samples, task metadata).
        """
        pass

    @property
    @abstractmethod
    def soft_prompt_parameters_size(self):
        """
        Gets the size of the input parameters tensor returned by this batch loader.
        For values greater than zero, the input tensor returned by __next__ will be of dimension
        [batch_size, input_parameters_size]. For values of zero, the soft prompt parameters will be None.
        """
        pass
