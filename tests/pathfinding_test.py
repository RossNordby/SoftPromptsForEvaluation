import torch

from pathfinding import train_and_test_pathfinding, PathfindingBatchLoader
from pathfinding.pathfinding_dataset import PathfindingDataset
from soft_prompting import MLPFactory
from soft_prompting.training_callbacks import ResultSavingCallbacks


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

    # Inserting spaces and a "Moves: " indicator helps quite a bit, so we'll use them for the main result.
    insert_spaces = True
    insert_move_section_separator = True

    def run_training(soft_prompt_token_counts, training_step_count):
        for board_width, board_height in board_configurations:
            def snapshot_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: None):
                return f"snapshots/pathfinding/{model_name}/{soft_prompt_token_count}, {board_width}x{board_height}.pt"

            def results_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: None):
                return f"results/pathfinding/{model_name}/{soft_prompt_token_count}, {board_width}x{board_height}.txt"

            evaluation_dataset = PathfindingDataset(board_width, board_height, insert_spaces)
            prompts = []
            soft_prompt_parameters = []
            for _ in range(128):
                board, moves, extra_move_count, invalid_move_count = next(evaluation_dataset)
                board = PathfindingBatchLoader.append_move_section_separator(board, insert_move_section_separator)
                prompts.append(board)
                soft_prompt_parameters.append((extra_move_count, invalid_move_count))

            train_and_test_pathfinding(
                board_width, board_height,
                model_configurations, soft_prompt_token_counts,
                MLPFactory(0, 128),
                insert_spaces=insert_spaces,
                insert_moves_section_separator=insert_move_section_separator,
                # logging_prefix=f"",
                training_step_count=training_step_count,
                batch_lanes_per_step=256,
                maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                forward_test_generated_token_count=32,
                training_callbacks=ResultSavingCallbacks(prompts, soft_prompt_parameters, 32,
                                                         False, snapshot_path_creator, results_path_creator))

    # Create a baseline result with a model with no soft prompt.
    run_training([0], 0)
    run_training([64, 16, 4, 1], 2048)


if __name__ == '__main__':
    main()
