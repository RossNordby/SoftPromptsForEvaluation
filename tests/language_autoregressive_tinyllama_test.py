from language_training import train_and_test_language, AutoregressiveBaseline
from soft_prompting import DirectFactory


def main():
    """
    Trains soft prompts on regular autoregressive prediction.
    Checks for a distributional gap between the tinyllama's original training distribution and the soft prompt's
    training distribution.
    """
    model_configurations = [('TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T', 128)]
    # model_configurations = [('meta-llama/Llama-2-7b-hf', 16)]
    soft_prompt_token_counts = [0, 1, 4, 16]

    train_and_test_language(
        model_configurations, soft_prompt_token_counts,
        DirectFactory(),
        batch_data_preparer=AutoregressiveBaseline(),
        #use_sample_dataset=True,
        logging_prefix=f"autoregressive",
        training_step_count=1024,
        batch_lanes_per_step=256,
        maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
        forward_test_generated_token_count=32)


if __name__ == '__main__':
    main()
