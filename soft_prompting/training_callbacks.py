from abc import ABC, abstractmethod

from soft_prompting import SoftPrompt, SnapshotPathCreator
from soft_prompting.snapshot_io import try_create_snapshot


class TrainingCallbacks(ABC):
    """
    Callbacks for the training loop.
    """

    # In principle, there could be checkpoint callbacks for checkpointing or whatever else, but individual training
    # runs for soft prompts are pretty small, and I'm out of time!
    @abstractmethod
    def training_complete(self, model, model_name: str,
                          maximum_sample_length_in_tokens: int, batch_lanes_per_step: int,
                          accumulation_step_count: int,
                          soft_prompt: SoftPrompt, training_step_count: int, learning_rate: float,
                          weight_decay: float):
        """
        Called when a soft prompt's training is complete.
        :param model: The model used for training.
        :param model_name: The name of the model used for training.
        :param maximum_sample_length_in_tokens: The maximum sample length in tokens pulled from the dataset.
        :param batch_lanes_per_step: The number of batch lanes to use per optimization step.
        :param accumulation_step_count: The number of accumulation steps used in training.
        :param soft_prompt: The soft prompt to save.
        :param training_step_count: The number of training steps used in training.
        :param learning_rate: The learning rate used in training.
        :param weight_decay: The weight decay used in training.
        """
        pass


class SnapshottingCallbacks(TrainingCallbacks):
    """
    Callbacks for the training loop that save snapshots.
    """

    def __init__(self, snapshot_path_creator: SnapshotPathCreator):
        """
        :param snapshot_path_creator: Function which creates paths to save snapshots to.
                                      Snapshots will be saved as a tuple of (soft prompt state dict, metadata dict).
        """
        self.snapshot_path_creator = snapshot_path_creator

    def training_complete(self, model, model_name: str,
                          maximum_sample_length_in_tokens: int, batch_lanes_per_step: int,
                          accumulation_step_count: int,
                          soft_prompt: SoftPrompt, training_step_count: int, learning_rate: float,
                          weight_decay: float):
        try_create_snapshot(self.snapshot_path_creator, model_name, soft_prompt.soft_prompt_token_count,
                            maximum_sample_length_in_tokens, batch_lanes_per_step, accumulation_step_count,
                            soft_prompt, training_step_count, learning_rate, weight_decay)
