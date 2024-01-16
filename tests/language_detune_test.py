from language_training import train_and_test_language, AutoregressiveBaseline
from soft_prompting import DirectFactory


def test(model_configurations, soft_prompt_token_counts, logging_prefix: str, training_step_count: int):
    train_and_test_language(
        model_configurations, soft_prompt_token_counts,
        DirectFactory(),
        batch_data_preparer=AutoregressiveBaseline(),
        #use_sample_dataset=True,
        logging_prefix=logging_prefix,
        training_step_count=training_step_count,
        batch_lanes_per_step=32,
        maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
        forward_test_generated_token_count=32)


def main():
    """
    Trains soft prompts to revert the fine-tuning of a model. This is just the autoregressive baseline,
    but configured to target a chat-tuned model.
    """
    training_step_count = 4096
    model_configurations = [('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 16)]
    soft_prompt_token_counts = [0]
    test(model_configurations, soft_prompt_token_counts, f"chat detuning rawbaseline", training_step_count)

    model_configurations = [('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 16)]
    soft_prompt_token_counts = [0]
    test(model_configurations, soft_prompt_token_counts, f"chat detuning chatbaseline", training_step_count)

    model_configurations = [('TinyLlama/TinyLlama-1.1B-Chat-v1.0', 16)]
    soft_prompt_token_counts = [1, 2, 4, 8, 16]

    test(model_configurations, soft_prompt_token_counts, f"chat detuning", training_step_count)


if __name__ == '__main__':
    main()
