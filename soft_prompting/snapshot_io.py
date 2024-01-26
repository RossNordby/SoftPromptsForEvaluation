import os

import torch

from soft_prompting import SnapshotPathCreator, SoftPrompt


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
