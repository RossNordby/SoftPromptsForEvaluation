from language_training import train_and_test_language
from language_training.skip_tokens import SkipTokens
from soft_prompting import DirectFactory


def main():
    """
    Trains soft prompts to predict tokens N steps ahead.
    """
    model_configurations = [('EleutherAI/pythia-70m-deduped', 4),
                            ('EleutherAI/pythia-160m-deduped', 8),
                            ('EleutherAI/pythia-410m-deduped', 16)]

    skip_token_counts = [1, 2, 4, 8]

    def train(soft_prompt_token_counts: list[int], training_step_count: int):
        for skip_count in skip_token_counts:
            train_and_test_language(
                model_configurations, soft_prompt_token_counts,
                DirectFactory(),
                batch_data_preparer=SkipTokens(skip_count),
                # use_sample_dataset=True,
                logging_prefix=f"skip {skip_count} tokens",
                training_step_count=training_step_count,
                batch_lanes_per_step=32,
                maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                forward_test_generated_token_count=32)

    # Collect baseline.
    train([0], 128)

    # Collect full length data.
    train([1, 4, 16, 64], 4096)


if __name__ == '__main__':
    main()
