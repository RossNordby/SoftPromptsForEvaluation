import torch

from pathfinding import train_and_test_pathfinding
from soft_prompting import MLPFactory
from soft_prompting.training_callbacks import SnapshottingCallbacks


def main():
    """
    Trains soft prompts to perform a simple pathfinding task.
    """

    torch.manual_seed(5)

    model_configurations = [
        ('EleutherAI/pythia-410m-deduped', 16),
        ('EleutherAI/pythia-160m-deduped', 8),
        ('EleutherAI/pythia-70m-deduped', 4)]

    board_configurations = [(8, 8)]
    insert_spaces = [True, False]
    insert_move_section_separator = [True, False]

    for use_space in insert_spaces:
        for insert_separator in insert_move_section_separator:

            def snapshot_path_creator(model_name: str, soft_prompt_token_count: int):
                return f"snapshots/pathfinding/{model_name}/spaces-{use_space}/movesep-{insert_move_section_separator}/soft-prompt-{soft_prompt_token_count}.pt"

            def run_training(use_spaces, insert_move_separator, soft_prompt_token_counts, training_step_count,
                             path_creator):
                for board_width, board_height in board_configurations:
                    train_and_test_pathfinding(
                        board_width, board_height,
                        model_configurations, soft_prompt_token_counts,
                        MLPFactory(0, 128),
                        insert_spaces=use_spaces,
                        insert_moves_section_separator=insert_move_separator,
                        logging_prefix=f"pathfinding2 movesep-{insert_move_separator} spaces-{use_spaces}",
                        training_step_count=training_step_count,
                        batch_lanes_per_step=256,
                        maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                        forward_test_generated_token_count=32,
                        training_callbacks=None if path_creator is None else SnapshottingCallbacks(path_creator))

            run_training(use_space, insert_separator, [0], 128, None)

            run_training(use_space, insert_separator, [64, 16, 4, 1], 2048, snapshot_path_creator)


if __name__ == '__main__':
    main()
