from chess_training import train_and_test_chess, game_processor
from chess_training.dataset_loading import ChessGameLoader, ChessBatchLoader, AsyncChessBatchLoader
from soft_prompting import MLPFactory
from soft_prompting.training_callbacks import ResultSavingCallbacks
from tests.test_shared import get_default_chess_database_path, get_default_model_configurations


def main():
    chess_database_path = 'chess_training/lichess_db_standard_rated_2023-11.pgn.zst'
    # Yes this is an entire 30 GB dataset for a few dozen evaluation games (at least with respect to this test).
    # Yes that's quite silly. Apologies.
    chess_test_database_path = 'chess_training/lichess_db_standard_rated_2023-12.pgn.zst'
    model_configurations = [
        ('410m', 16),
        ('160m', 8),
        ('70m', 4),
    ]

    evaluation_game_loader = ChessGameLoader(chess_test_database_path)
    evaluation_game_count = 16
    evaluation_prompts = []
    evaluation_conditions = []
    with evaluation_game_loader:
        while len(evaluation_prompts) < evaluation_game_count:
            game = game_processor(next(evaluation_game_loader))
            if game is not None:
                white_elo, black_elo, moves = game
                evaluation_prompts.append(moves)
                evaluation_conditions.append((white_elo, black_elo))

    def train(soft_prompt_token_counts: list[int], training_step_count: int):
        def snapshot_path_creator(model_name: str, soft_prompt_token_count: int):
            return f"snapshots/chess/{model_name}-{soft_prompt_token_count}.pt"

        def results_path_creator(model_name: str, soft_prompt_token_count: int):
            return f"results/chess/{model_name}-{soft_prompt_token_count}.txt"

        train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                             MLPFactory(0, 128),
                             # logging_prefix="",
                             training_step_count=training_step_count,
                             batch_lanes_per_step=256,
                             maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                             forward_test_generated_token_count=32,
                             training_callbacks=ResultSavingCallbacks(evaluation_prompts, evaluation_conditions, 32,
                                                                      False, snapshot_path_creator,
                                                                      results_path_creator))

    train([0], 128)
    train([1024, 512, 256, 64, 16, 4, 1], 16384)


if __name__ == '__main__':
    main()
