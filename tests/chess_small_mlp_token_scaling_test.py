from chess_training import train_and_test_chess
from soft_prompting import MLPFactory
from tests.test_shared import get_default_chess_database_path


def snapshot_path_creator(model_size: str, soft_prompt_token_count: int) -> str:
    return f"small_mlp_very_long_{model_size}_{soft_prompt_token_count}.pt"


def main():
    """
    Test of the effect of token scaling on small MLP-derived soft prompts in chess.
    """
    chess_database_path = get_default_chess_database_path()
    model_configurations = [('70m', 16)]
    # soft_prompt_token_counts = [64, 128, 256, 512, 1024, 2048]
    soft_prompt_token_counts = [1024, 2048]

    train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                         MLPFactory(0, 128),
                         logging_prefix=f"small mlp tokenscale (very long)",
                         training_step_count=32768,
                         batch_lanes_per_step=32,
                         maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                         forward_test_generated_token_count=32,
                         snapshot_path_creator=snapshot_path_creator)


if __name__ == '__main__':
    main()
