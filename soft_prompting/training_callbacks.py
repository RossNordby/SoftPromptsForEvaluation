import os
from abc import ABC, abstractmethod

import soft_prompting
from soft_prompting import SoftPrompt, PathCreator
from soft_prompting.snapshot_io import try_create_snapshot


class TrainingCallbacks(ABC):
    """
    Callbacks for the training loop.
    """

    # In principle, there could be checkpoint callbacks for checkpointing or whatever else, but individual training
    # runs for soft prompts are pretty small, and I'm out of time!
    @abstractmethod
    def training_complete(self, model_name: str, model, tokenizer,
                          maximum_sample_length_in_tokens: int, batch_lanes_per_step: int,
                          accumulation_step_count: int,
                          soft_prompt: SoftPrompt, training_step_count: int, learning_rate: float,
                          weight_decay: float):
        """
        Called when a soft prompt's training is complete.
        :param model: The model used for training.
        :param model_name: The name of the model used for training.
        :param tokenizer: The tokenizer used for training.
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

    def __init__(self, snapshot_path_creator: PathCreator):
        """
        :param snapshot_path_creator: Function which creates paths to save snapshots to.
                                      Snapshots will be saved as a tuple of (soft prompt state dict, metadata dict).
        """
        self.snapshot_path_creator = snapshot_path_creator

    def training_complete(self, model_name: str, model, tokenizer,
                          maximum_sample_length_in_tokens: int, batch_lanes_per_step: int,
                          accumulation_step_count: int,
                          soft_prompt: SoftPrompt, training_step_count: int, learning_rate: float,
                          weight_decay: float):
        try_create_snapshot(self.snapshot_path_creator, model_name, soft_prompt.soft_prompt_token_count,
                            maximum_sample_length_in_tokens, batch_lanes_per_step, accumulation_step_count,
                            soft_prompt, training_step_count, learning_rate, weight_decay)


class ResultSavingCallbacks(TrainingCallbacks):
    """
    Callbacks for the training loop that save results.
    """

    def __init__(self, prompts: list[str], soft_prompt_parameters: list[tuple] | None, generated_token_count: int,
                 soft_prompt_at_end: bool, snapshot_path_creator: PathCreator, results_path_creator: PathCreator):
        """
        :param prompts: The prompts to use for generation.
        :param soft_prompt_parameters: The soft prompt parameters to use for generation. If None, no parameters will be
                                       used.
        :param generated_token_count: The number of tokens to generate.
        :param soft_prompt_at_end: Whether to put the soft prompt at the end of the prompt.
                                   If true, the soft prompt is placed immediately after the prompt tokens and before
                                   any generated tokens.
                                   If false, the soft prompt is inserted before any prompt.
        :param snapshot_path_creator: Function which creates paths to save snapshots to.
        :param results_path_creator: Function which creates paths to save result strings to.
        """
        self.prompts = prompts
        self.soft_prompt_parameters = soft_prompt_parameters
        self.generated_token_count = generated_token_count
        self.soft_prompt_at_end = soft_prompt_at_end
        self.snapshot_path_creator = snapshot_path_creator
        self.results_path_creator = results_path_creator

    def training_complete(self, model_name: str, model, tokenizer,
                          maximum_sample_length_in_tokens: int, batch_size: int,
                          accumulation_step_count: int,
                          soft_prompt: SoftPrompt, training_step_count: int, learning_rate: float,
                          weight_decay: float):
        result_strings = []
        # The generator supports inserting it anywhere, but the callbacks just choose 0 or max. None means max.
        soft_prompt_start_index = None if self.soft_prompt_at_end else 0
        for i in range(0, len(self.prompts), batch_size):
            start = i
            end = min(len(self.prompts), i + batch_size)
            effective_batch_size = end - start
            input_ids, output_ids = soft_prompting.training_and_testing.generate_from_prompts(
                self.prompts[start:end],
                None if self.soft_prompt_parameters else self.soft_prompt_parameters[start:end],
                soft_prompt_start_index, soft_prompt,
                model, tokenizer,
                effective_batch_size, self.generated_token_count)
            batch_result_strings = soft_prompting.training_and_testing.create_strings_from_prompted_generation(
                input_ids, output_ids, tokenizer, soft_prompt_start_index, '[SOFT PROMPT]')
            result_strings.extend(batch_result_strings)
        print(f'Prompted generation results for {model_name} with soft prompt token length'
              f' {soft_prompt.soft_prompt_token_count}:')

        formatted_results = [f"Conditions {i}:\n{t}\nSequence:\n{seq}" for i, (t, seq) in
                             enumerate(zip(self.soft_prompt_parameters, result_strings))]
        generated_string = '\n\n'.join(formatted_results)
        print(generated_string)

        if self.results_path_creator is not None:
            results_path = self.results_path_creator(model_name, soft_prompt.soft_prompt_token_count)
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, 'w', encoding='utf-8') as file:
                file.write(generated_string)
        try_create_snapshot(self.snapshot_path_creator, model_name, soft_prompt.soft_prompt_token_count,
                            maximum_sample_length_in_tokens, batch_size, accumulation_step_count,
                            soft_prompt, training_step_count, learning_rate, weight_decay)
