from chess_training import train_and_test_chess
from soft_prompting import ResnetFactory
from tests.test_shared import get_default_chess_database_path


def main():
    chess_database_path = get_default_chess_database_path()
    model_configurations = [('1b', 4)]
    soft_prompt_token_counts = [64]
    # Configurations are (learning rate, weight decay).
    learning_configurations = [(0.01, 0.5), (0.01, 0.1), (0.01, 0.01), (0.01, 0),
                               (0.005, 0.1), (0.005, 0.01), (0.005, 0.001), (0.005, 0),
                               (0.001, 0.1), (0.001, 0.01), (0.001, 0.001), (0.001, 0),
                               (0.0001, 0.01), (0.0001, 0.001), (0.0001, 0.0001), (0.0001, 0)]

    for learning_rate, weight_decay in learning_configurations:
        train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                             ResnetFactory(16, 2, 32),
                             logging_prefix=f"resnet lr {learning_rate}, wd {weight_decay}",
                             training_step_count=512, batch_lanes_per_step=32,
                             maximum_sample_length_in_tokens=256, learning_rate=learning_rate,
                             weight_decay=weight_decay,
                             forward_test_generated_token_count=128)


if __name__ == '__main__':
    main()
