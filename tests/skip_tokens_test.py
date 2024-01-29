from language_training import train_and_test_language, dataset_iterables
from language_training.skip_tokens import SkipTokens
from soft_prompting import DirectFactory
from soft_prompting.training_callbacks import ResultSavingCallbacks
from tests.tests_shared import append_loaded_prompts


def main():
    """
    Trains soft prompts to predict tokens N steps ahead.
    """
    model_configurations = [
        ('EleutherAI/pythia-410m-deduped', 4),
        ('EleutherAI/pythia-160m-deduped', 2),
        ('EleutherAI/pythia-70m-deduped', 1),
    ]

    skip_token_counts = [1, 2, 4, 8]

    prompts = []
    append_loaded_prompts(128, 256, prompts)

    datasets = [
        dataset_iterables.RedPajamaV2DatasetIterable(),
    ]

    def train(soft_prompt_token_counts: list[int], training_step_count: int):
        for skip_count in skip_token_counts:
            def snapshot_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: str):
                return f"snapshots/skip_{skip_count}_tokens/{dataset_name}/{model_name}-{soft_prompt_token_count}.pt"

            def results_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: str):
                return f"results/skip_{skip_count}_tokens/{dataset_name}/{model_name}-{soft_prompt_token_count}.txt"

            train_and_test_language(
                model_configurations, soft_prompt_token_counts, datasets, DirectFactory(),
                batch_data_preparer=SkipTokens(skip_count),
                logging_prefix=f"skip {skip_count} tokens",
                training_step_count=training_step_count,
                batch_lanes_per_step=32,
                maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                forward_test_generated_token_count=32,
                training_callbacks=ResultSavingCallbacks(prompts, None, 32, True, snapshot_path_creator,
                                                         results_path_creator))

    # Collect baseline.
    train([0], 0)

    # Collect full length data.
    train([1, 4, 16, 64], 4096)


if __name__ == '__main__':
    main()
