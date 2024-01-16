from chess_training import train_and_test_chess
from soft_prompting import MLPFactory
from tests.test_shared import get_default_chess_database_path


def main():
    chess_database_path = get_default_chess_database_path()
    model_configurations = [('1b', 4)]
    soft_prompt_token_counts = [64]
    prompt_configurations = [(0, 32), (1, 32), (4, 32), (8, 32),
                             (0, 128), (0, 256), (0, 512),
                             (1, 128), (1, 256), (1, 512)]

    for configuration in prompt_configurations:
        train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                             MLPFactory(*configuration),
                             logging_prefix=f"mlp scale {configuration}",
                             training_step_count=512,
                             batch_lanes_per_step=32,
                             maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                             forward_test_generated_token_count=128)


if __name__ == '__main__':
    main()
