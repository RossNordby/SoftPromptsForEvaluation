import torch

from soft_prompting import soft_prompts
from soft_prompting.soft_prompts import SoftPrompt


class MLPLayer(torch.nn.Module):
    """
    A single layer in an MLP-style model.
    """

    def __init__(self, input_width: int, output_width: int, device: torch.device = None):
        super().__init__()
        self.linear = torch.nn.Linear(input_width, output_width, device=device)
        self.nonlinearity = torch.nn.Mish()

    def forward(self, x: torch.Tensor):
        return self.nonlinearity(self.linear(x))


class MLPSoftPrompt(SoftPrompt):
    """
    A soft prompt which is generated by a simple MLP model.
    """

    def __init__(self, soft_prompt_token_count: int, input_size: int, output_embedding_size: int,
                 hidden_layer_count: int, hidden_layer_width: int, device: torch.device = None):
        """
        Creates an MLP soft prompt.
        :param soft_prompt_token_count: Number of tokens in the soft prompt.
        :param output_embedding_size: Size of the embedding to create.
        :param hidden_layer_count: Number of hidden layers in the soft prompt generator.
        Doesn't include the input layer; the input layer is always present.
        :param hidden_layer_width: Width of the hidden layers in the soft prompt generator.
        :param device: Device to create the soft prompt on.
        """
        super().__init__(soft_prompt_token_count)
        self.output_embedding_size = output_embedding_size
        self.hidden_layer_width = hidden_layer_width
        self.input_layer = MLPLayer(input_size, soft_prompt_token_count * hidden_layer_width, device=device)
        # The parameters for each token are generated by a separate linear model.
        # Note that we also use a separate linear model for the output embedding.
        # Organizing as [layer_count, token_count] instead of [token_count, layer_count] on the assumption
        # that pytorch would sequentialize the latter. Worth testing.
        self.layers = torch.nn.ModuleList([torch.nn.ModuleList(
            [MLPLayer(hidden_layer_width, hidden_layer_width, device=device) for _ in
             range(soft_prompt_token_count)]) for _ in range(hidden_layer_count)])
        self.to_embedding = torch.nn.ModuleList(
            torch.nn.Linear(hidden_layer_width, output_embedding_size, device=device) for _ in
            range(soft_prompt_token_count))

    def forward(self, soft_prompt_parameters: torch.Tensor):
        # Residual soft prompts are generated by a resnet-style model.
        # The model takes a tensor of shape [batch_count, input_size] and returns a tensor of shape
        # [batch_count, soft_prompt_token_count, embedding_size].
        if self.soft_prompt_token_count == 0:
            return torch.empty([soft_prompt_parameters.size(0), 0, self.output_embedding_size], dtype=torch.float32,
                               device=soft_prompt_parameters.device)
        else:
            combined_tokens = self.input_layer(soft_prompt_parameters).view(-1, self.soft_prompt_token_count,
                                                                            self.hidden_layer_width)
            # Treat the execution of each token as independent to avoid pointless per-step stacking and slicing.
            tokens = [combined_tokens[:, i, :] for i in range(self.soft_prompt_token_count)]
            for layer in self.layers:
                for token_index in range(self.soft_prompt_token_count):
                    tokens[token_index] = layer[token_index](tokens[token_index])
            token_embeddings = [self.to_embedding[token_index](tokens[token_index]) for token_index in
                                range(self.soft_prompt_token_count)]
            return torch.stack(token_embeddings, dim=1)

    @property
    def input_parameter_count(self) -> int:
        return self.input_layer.linear.in_features

    def get_metadata(self) -> dict:
        return {"soft_prompt_token_count": self.soft_prompt_token_count,
                "input_size": self.input_parameter_count, "output_embedding_size": self.output_embedding_size,
                "hidden_layer_count": len(self.layers), "hidden_layer_width": self.hidden_layer_width}


class MLPFactory:
    def __init__(self, hidden_layer_count, hidden_width):
        self.hidden_layer_count = hidden_layer_count
        self.hidden_width = hidden_width

    def __call__(self, soft_prompt_token_count: int, input_size: int, embedding_size: int) -> soft_prompts.SoftPrompt:
        return MLPSoftPrompt(soft_prompt_token_count, input_size, embedding_size,
                             self.hidden_layer_count, self.hidden_width)
