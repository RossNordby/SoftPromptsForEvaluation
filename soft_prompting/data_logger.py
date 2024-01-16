import torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter


class DataLogger:
    """
    Simple wrapper class for logging training data. Handles writing to tensorboard logs in the context of distributed
    training. Only the main process writes logs.
    """

    def __init__(self, log_path: str, accelerator: Accelerator):
        """
        Creates a TrainingLogger.
        :param log_path: The path to write logs to.
        :param accelerator: The accelerator to use to determine if this is the main process.
        """
        if accelerator.is_main_process:
            self.writer = SummaryWriter(log_path)
            self.accelerator = accelerator

    def add_scalar(self, tag: str, value: torch.Tensor, step: int):
        """
        Adds a scalar to the log if this is the main process.
        Requires that the scalar be represented by a tensor that can be gathered if execution is distributed.
        :param tag: The tag to add the scalar to.
        :param value: The value to add.
        :param step: The step to add the scalar at.
        """
        gathered = self.accelerator.gather(value)
        if self.writer is not None:
            gathered_mean_value = torch.mean(gathered)
            self.writer.add_scalar(tag, gathered_mean_value, step)

    def close(self):
        """
        Closes the logger.
        """
        if self.writer is not None:
            self.writer.close()
