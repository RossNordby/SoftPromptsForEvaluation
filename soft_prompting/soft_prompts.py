import torch
from abc import ABC, abstractmethod


class SoftPrompt(torch.nn.Module, ABC):

    def __init__(self, soft_prompt_token_count: int):
        super().__init__()
        self.soft_prompt_token_count = soft_prompt_token_count

    @property
    @abstractmethod
    def input_parameter_count(self) -> int:
        """
        Gets the number of input parameters required by this soft prompt.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> dict:
        """
        Gets metadata about this soft prompt that is specific to the type.
        """
        pass
