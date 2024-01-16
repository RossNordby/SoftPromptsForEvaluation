from chess_training import train_and_test_chess
from soft_prompting import SharedResnetFactory
from tests.test_shared import get_default_chess_database_path


def main():
    chess_database_path = get_default_chess_database_path()
    model_configurations = [('70m', 4)]
    soft_prompt_token_counts = [64, 128, 256]
    resnet_configurations = [(16, 2, 32), (64, 2, 32), (128, 2, 32),  # Vary residual width
                             (16, 1, 32), (16, 4, 32), (16, 8, 32),  # Vary hidden layer count
                             (16, 2, 8), (16, 2, 128),  # Vary hidden width
                             (64, 1, 64), (64, 2, 64), (64, 4, 64),
                             (128, 1, 128), (128, 2, 128), (128, 4, 128)]

    for configuration in resnet_configurations:
        train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                             SharedResnetFactory(*configuration),
                             logging_prefix=f"shared resnet scale {configuration}",
                             training_step_count=512,
                             batch_lanes_per_step=32,
                             maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                             forward_test_generated_token_count=128)


if __name__ == '__main__':
    main()
