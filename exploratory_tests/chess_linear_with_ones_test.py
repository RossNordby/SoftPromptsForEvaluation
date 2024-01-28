from chess_training import train_and_test_chess, SoftPromptParameterMode
from soft_prompting import LinearFactory
from exploratory_tests.test_shared import get_default_chess_database_path, get_default_model_configurations


def main():
    chess_database_path = get_default_chess_database_path()
    model_configurations = get_default_model_configurations()
    soft_prompt_token_counts = [16, 64]
    train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                         LinearFactory(),
                         soft_prompt_parameter_mode=SoftPromptParameterMode.ELOS_PLUS_ONES,
                         logging_prefix="linear + 1 x models",
                         training_step_count=1024,
                         batch_lanes_per_step=32,
                         maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                         forward_test_generated_token_count=128)


if __name__ == '__main__':
    main()
