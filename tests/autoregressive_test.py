from datasets import load_dataset

from language_training import train_and_test_language, AutoregressiveBaseline
from soft_prompting import DirectFactory
from soft_prompting.training_callbacks import ResultSavingCallbacks
from tests.chat_detuning_test import append_loaded_prompts


def main():
    """
    Trains soft prompts on regular autoregressive prediction. In the absence of a distributional gap between
    the model's original training distribution and the soft prompt's training distribution,
    getting any benefit here would be strange. (See paper for details.)
    """

    model_configurations = [('EleutherAI/pythia-1b-deduped', 8),
                            ('EleutherAI/pythia-410m-deduped', 4),
                            ('EleutherAI/pythia-160m-deduped', 2),
                            ('EleutherAI/pythia-70m-deduped', 1),
                            ]

    # We'll use a mix of different prompts. Most will just be raw unstructured text from redpajama.
    prompts = ["The quick brown fox jumps",
               "The creature, low and squat, peered aside me; ",
               "Meow! Hiss! The real problem facing Novosibirsk in the wake of The Great Cat-astrophe of 1846",
               "Curiously, AI capabilities have not improved much since the",
               "I tried to get him to give me a cup of water but he just kept saying '",
               "Densely informative loss functions have an advantage in optimization landscapes that",
               "'Not a fan of that at all,' he said. 'I prefer cayenne pepper in my cereal.'",
               "Want to hear one weird trick to",
               "THE TEN BIGGEST REASONS WHY YOU STILL CAN'T",
               "I'm... darkly amused, maybe, but mostly sickened. My most-viewed video on youtube was about *",
               "If you look at the math, most babies spend at least 6 months in orbit around Earth before",
               "INTERNATIONAL CONFERENCE FOR DOG SHAPED COOKIES: Paper submission",
               "New reports indicate that no one likes webdev. 'It's just so', said one developer, '",
               ]

    append_loaded_prompts(128, 256, prompts)

    def snapshot_path_creator(model_name: str, soft_prompt_token_count: int):
        return f"snapshots/autoregressive/{model_name}-{soft_prompt_token_count}.pt"

    def results_path_creator(model_name: str, soft_prompt_token_count: int):
        return f"results/autoregressive/{model_name}-{soft_prompt_token_count}.txt"

    def train(soft_prompt_token_counts: list[int], training_step_count: int):
        train_and_test_language(
            model_configurations, soft_prompt_token_counts,
            DirectFactory(),
            batch_data_preparer=AutoregressiveBaseline(),
            # use_sample_dataset=True,
            maximum_soft_prompt_start_indices=0,
            logging_prefix=f"autoregressive",
            training_step_count=training_step_count,
            batch_lanes_per_step=32,
            maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
            forward_test_generated_token_count=32,
            training_callbacks=ResultSavingCallbacks(prompts, None, 128,
                                                     False, snapshot_path_creator, results_path_creator))

    train([0], 0)
    train([64, 16, 4, 1], 1024)


if __name__ == '__main__':
    main()
