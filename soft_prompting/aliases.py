from typing import TypeAlias, Callable

from torch import Tensor

from soft_prompting import SoftPrompt

SoftPromptFactory: TypeAlias = Callable[[int, int, int], SoftPrompt]
"""
A function which creates a soft prompt given the number of tokens in the soft prompt, 
the input size, and the embedding size.
"""

PathCreator: TypeAlias = Callable[[str, int, str | None], str]
"""
Creates a path from a given the model name (e.g. "EleutherAI/pythia-1.4b-deduped" or 
"meta-llama/Llama-2-7b-hf"), soft prompt token count, and dataset name (if any). Used for saving snapshots and results.
"""

TrainingCompleteCallback: TypeAlias = Callable[[str, int], str]
"""
Called by the training loop when a soft prompt's training is complete.
"""

SoftPromptLossFunction: TypeAlias = Callable[[Tensor, Tensor, SoftPrompt, Tensor], Tensor]
"""
A loss function of the form:
(logits, labels, soft prompt, soft prompt start indices) -> loss.
"""

EmbedInputFunction: TypeAlias = Callable[[Tensor], Tensor]
"""
A function that takes token ids and returns embeddings.
"""
