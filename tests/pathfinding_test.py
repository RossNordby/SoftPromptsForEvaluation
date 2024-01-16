import torch

from pathfinding import train_and_test_pathfinding
from soft_prompting import MLPFactory


def main():
    """
    Trains soft prompts to perform a simple pathfinding task.
    """
    torch.manual_seed(5)

    model_configurations = [('EleutherAI/pythia-70m-deduped', 4),
                            ('EleutherAI/pythia-160m-deduped', 8),
                            ('EleutherAI/pythia-410m-deduped', 16)]

    board_configurations = [(8, 8)]
    use_spaces = [False, True]

    for use_space in use_spaces:
        soft_prompt_token_counts = [0]
        for board_width, board_height in board_configurations:
            train_and_test_pathfinding(
                board_width, board_height,
                model_configurations, soft_prompt_token_counts,
                MLPFactory(0, 128),
                insert_spaces=use_space,
                logging_prefix=f"pathfinding spaces-{use_space}",
                training_step_count=1,
                batch_lanes_per_step=32,
                maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                forward_test_generated_token_count=32)

        def snapshot_path_creator(model_name: str, soft_prompt_token_count: int):
            return f"../snapshots/pathfinding/{model_name}/spaces-{use_space}/soft-prompt-{soft_prompt_token_count}.pt"

        soft_prompt_token_counts = [1, 4, 16, 64]
        for board_width, board_height in board_configurations:
            train_and_test_pathfinding(
                board_width, board_height,
                model_configurations, soft_prompt_token_counts,
                MLPFactory(0, 128),
                insert_spaces=use_space,
                logging_prefix=f"pathfinding spaces-{use_space}",
                training_step_count=1,
                batch_lanes_per_step=32,
                maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                forward_test_generated_token_count=32, snapshot_path_creator=snapshot_path_creator)


if __name__ == '__main__':
    main()
