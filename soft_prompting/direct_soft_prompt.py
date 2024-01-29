import torch

from soft_prompting import soft_prompts
from soft_prompting.soft_prompts import SoftPrompt


class DirectSoftPrompt(SoftPrompt):
    """
    A soft prompt which containing a directly optimized set of embeddings for some number of tokens.
    Direct soft prompts are not driven by externally supplied input parameters; there is no generating model.
    """

    def __init__(self, soft_prompt_token_count: int, embedding_size: int, use_zero_init: bool = False):
        """
        Creates a direct soft prompt.
        :param soft_prompt_token_count: Number of tokens in the soft prompt.
        :param embedding_size: Size of the embedding to create if random_init is True. Unused otherwise.
        :param use_zero_init: If True, the embeddings are initialized to zero. Otherwise, they are initialized randomly
        with a normal distribution with mean 0 and standard deviation 0.1.
        """
        super().__init__(soft_prompt_token_count)

        if use_zero_init:
            self.embedding = torch.nn.Parameter(torch.zeros([soft_prompt_token_count, embedding_size]))
        else:
            self.embedding = torch.nn.Parameter(
                torch.normal(torch.zeros([soft_prompt_token_count, embedding_size]), 0.1))

    def forward(self, batch_count):
        # Direct soft prompts do not involve any processing.
        # Just broadcast to the full width.
        return self.embedding.expand(batch_count, -1, -1)

    @property
    def input_parameter_count(self) -> int:
        return 0

    def get_metadata(self) -> dict:
        return {"soft_prompt_token_count": self.soft_prompt_token_count,
                "embedding_size": self.embedding.size(1)}


class DirectFactory:
    def __call__(self, soft_prompt_token_count: int, input_size: int, embedding_size: int) -> soft_prompts.SoftPrompt:
        return DirectSoftPrompt(soft_prompt_token_count, embedding_size)
