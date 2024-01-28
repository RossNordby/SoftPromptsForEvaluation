from language_training import train_and_test_language, favor_future_predictions
from soft_prompting import DirectFactory


def main():
    """
    Trains soft prompts with a skewed loss function that favors predicting tokens that are further in the future.
    """
    model_configurations = [('EleutherAI/pythia-70m-deduped', 1),
                            ('EleutherAI/pythia-160m-deduped', 2),
                            ('EleutherAI/pythia-410m-deduped', 4),
                            ('EleutherAI/pythia-1b-deduped', 8)]
    # model_configurations = [('meta-llama/Llama-2-7b-hf', 16)]
    soft_prompt_token_counts = [1, 4, 16, 64, 256]

    train_and_test_language(
        model_configurations, soft_prompt_token_counts,
        DirectFactory(),
        batch_data_preparer=favor_future_predictions.FavorFutureTest(8),
        # use_sample_dataset=True,
        logging_prefix=f"favor future",
        training_step_count=128,
        batch_lanes_per_step=32,
        maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
        forward_test_generated_token_count=32)


if __name__ == '__main__':
    main()
