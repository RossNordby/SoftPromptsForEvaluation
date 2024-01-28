from language_training import train_and_test_language, AutoregressiveBaseline
from soft_prompting import DirectFactory


def main():
    """
    Trains soft prompts on regular autoregressive prediction. In the absence of a distributional gap between
    the model's original training distribution and the soft prompt's training distribution,
    getting any benefit here would be strange. (See paper for details.)

    The chat-tuned variant of TinyLlama should see some movement in loss, as the soft prompt is effectively
    undoing the chat tuning.
    """
    model_configurations = [
        ('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 8),
        ('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 8),
        ('EleutherAI/pythia-1b-deduped', 8),
        ('EleutherAI/pythia-410m-deduped', 4),
        ('EleutherAI/pythia-160m-deduped', 2),
        ('EleutherAI/pythia-70m-deduped', 1),
    ]

    def train(soft_prompt_token_counts: list[int], training_step_count: int):
        train_and_test_language(
            model_configurations, soft_prompt_token_counts,
            DirectFactory(),
            batch_data_preparer=AutoregressiveBaseline(),
            # use_sample_dataset=True,
            logging_prefix=f"autoregressive",
            training_step_count=training_step_count,
            batch_lanes_per_step=32,
            maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
            forward_test_generated_token_count=32)

    train([0], 0)
    train([1, 4, 16, 64], 1024)


if __name__ == '__main__':
    main()
