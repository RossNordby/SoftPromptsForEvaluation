from language_training import train_and_test_language, TrivialRepetitionTest, dataset_iterables
from soft_prompting import DirectFactory
from soft_prompting.training_callbacks import ResultSavingCallbacks
from tests.tests_shared import append_loaded_prompts


def main():
    """
    Trains soft prompts to repeat a string over and over again.
    Repeating longer strings tends to require more prompting.
    """
    model_configurations = [
        ('EleutherAI/pythia-1b-deduped', 8),
        ('EleutherAI/pythia-410m-deduped', 4),
        ('EleutherAI/pythia-160m-deduped', 2),
        ('EleutherAI/pythia-70m-deduped', 1),
    ]

    repeated_strings = ['pee', 'meow', 'meowster',
                        'meowster jones', 'the great meowster jones', 'the great and terrible meowster jones']

    prompts = []
    append_loaded_prompts(128, 256, prompts)

    def snapshot_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: str):
        return f"snapshots/repetition/{dataset_name}/{model_name}-{soft_prompt_token_count}.pt"

    def results_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: str):
        return f"results/repetition/{dataset_name}/{model_name}-{soft_prompt_token_count}.txt"

    datasets = [
        dataset_iterables.RedPajamaV2DatasetIterable(),
    ]

    def train(soft_prompt_token_counts: list[int], training_step_count: int):
        for s in repeated_strings:
            train_and_test_language(
                model_configurations, soft_prompt_token_counts, datasets, DirectFactory(use_zero_init=True),
                batch_data_preparer=TrivialRepetitionTest(s),
                maximum_soft_prompt_start_indices=0,
                logging_prefix=f"repetition ({s})",
                training_step_count=training_step_count,
                batch_lanes_per_step=32,
                maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                forward_test_generated_token_count=32,
                training_callbacks=ResultSavingCallbacks(prompts, None, 128,
                                                         False, snapshot_path_creator, results_path_creator))

    # Collect baseline.
    train([0], 0)
    train([64, 16, 4, 1], 2048)


if __name__ == '__main__':
    main()
