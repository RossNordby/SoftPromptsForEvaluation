from language_training import train_and_test_language, TrivialRepetitionTest
from soft_prompting import DirectFactory


def main():
    """
    Trains soft prompts to repeat a string over and over again.
    Repeating longer strings tends to require more prompting.
    """
    model_configurations = [('EleutherAI/pythia-70m-deduped', 4)]
    soft_prompt_token_counts = [1, 2, 4, 8, 16, 64]

    repeated_strings = ['pee', 'meow', 'meowster',
                        'meowster jones', 'the great meowster jones', 'the great and terrible meowster jones']
    for s in repeated_strings:
        train_and_test_language(
            model_configurations, soft_prompt_token_counts,
            DirectFactory(),
            batch_data_preparer=TrivialRepetitionTest(s),
            # use_sample_dataset=True,
            logging_prefix=f"repetition ({s})",
            training_step_count=1024,
            batch_lanes_per_step=32,
            maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
            forward_test_generated_token_count=32)


if __name__ == '__main__':
    main()
