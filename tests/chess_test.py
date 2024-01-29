from chess_training import train_and_test_chess, game_processor
from chess_training.dataset_loading import ChessGameLoader, normalize_chess_elos
from soft_prompting import MLPFactory
from soft_prompting.training_callbacks import ResultSavingCallbacks


def main():
    chess_database_path = 'chess_training/lichess_db_standard_rated_2023-11.pgn.zst'
    # Yes this is an entire 30 GB dataset for a few dozen evaluation games (at least with respect to this test).
    # Yes that's quite silly. Apologies.
    chess_test_database_path = 'chess_training/lichess_db_standard_rated_2023-12.pgn.zst'

    evaluation_game_loader = ChessGameLoader(chess_test_database_path)
    evaluation_game_count = 128
    evaluation_prompts = []
    evaluation_conditions = []
    with evaluation_game_loader:
        while len(evaluation_prompts) < evaluation_game_count:
            game = game_processor(next(evaluation_game_loader))
            if game is not None:
                elos, moves = game
                evaluation_prompts.append(moves)
                evaluation_conditions.append(normalize_chess_elos(elos))

    def train(model_configurations, soft_prompt_token_counts: list[int], training_step_count: int):
        def snapshot_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: None):
            return f"snapshots/chess/{model_name}-{soft_prompt_token_count}.pt"

        def results_path_creator(model_name: str, soft_prompt_token_count: int, dataset_name: None):
            return f"results/chess/{model_name}-{soft_prompt_token_count}.txt"

        train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                             MLPFactory(0, 128),
                             # logging_prefix="",
                             training_step_count=training_step_count,
                             batch_lanes_per_step=32,
                             maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                             forward_test_generated_token_count=32,
                             training_callbacks=ResultSavingCallbacks(evaluation_prompts, evaluation_conditions, 32,
                                                                      False, snapshot_path_creator,
                                                                      results_path_creator))

    # The chess test will be using unusually large soft prompts, so we'll need to adjust the model configuration
    # depending on the soft prompt token count.
    small_prompt_model_configurations = [
        ('160m', 2),
        ('70m', 1),
    ]
    train(small_prompt_model_configurations, [0], 0)

    # This isn't enough to reach convergence, but I don't have the compute or time to do that.
    # The good news is that earlier tests on smaller models suggest that the relative position of loss curves
    # remains similar as the training step count increases.
    training_step_count = 4096
    large_prompt_model_configurations = [
        ('160m', 8),
        ('70m', 4),
    ]
    train(large_prompt_model_configurations, [1024, 512], training_step_count)

    medium_prompt_model_configurations = [
        ('160m', 4),
        ('70m', 2),
    ]
    train(medium_prompt_model_configurations, [256, 64], training_step_count)
    train(small_prompt_model_configurations, [16, 4, 1], training_step_count)



if __name__ == '__main__':
    main()
