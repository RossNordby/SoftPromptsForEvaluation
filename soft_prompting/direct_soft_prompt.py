import torch

from soft_prompting import soft_prompts
from soft_prompting.soft_prompts import SoftPrompt


class DirectSoftPrompt(SoftPrompt):
    """
    A soft prompt which containing a directly optimized set of embeddings for some number of tokens.
    Direct soft prompts are not driven by externally supplied input parameters; there is no generating model.
    """

    def __init__(self, soft_prompt_token_count: int, embedding_size: int):
        """
        Creates a direct soft prompt.
        :param soft_prompt_token_count: Number of tokens in the soft prompt.
        :param embedding_size: Size of the embedding to create if random_init is True. Unused otherwise.
        """
        super().__init__(soft_prompt_token_count)

        # self.embedding = torch.nn.Parameter(
        #     torch.clamp(torch.normal(torch.empty([soft_prompt_token_count, embedding_size], requires_grad=True),
        #                              torch.fill(torch.zeros([soft_prompt_token_count, embedding_size]),
        #                                         .1)), -1, 1).detach())
        self.embedding = torch.nn.Parameter(torch.zeros([soft_prompt_token_count, embedding_size]))

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
