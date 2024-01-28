from chess_training import train_and_test_chess, SoftPromptParameterMode
from soft_prompting import DirectFactory
from exploratory_tests.test_shared import get_default_chess_database_path


def main():
    chess_database_path = get_default_chess_database_path()
    model_configurations = [('70m', 8), ('160m', 16), ('410m', 32)]
    soft_prompt_token_counts = [64, 128, 256, 512]
    train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                         DirectFactory(),
                         soft_prompt_parameter_mode=SoftPromptParameterMode.UNCONDITIONAL,
                         logging_prefix="direct x models unconditioned tokenscale (long)",
                         training_step_count=8192,
                         batch_lanes_per_step=32,
                         maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                         forward_test_generated_token_count=128)


if __name__ == '__main__':
    main()
