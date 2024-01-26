import torch

import soft_prompting
from chess_training import train_and_test_chess
from soft_prompting import ResnetFactory
from soft_prompting.training_callbacks import SnapshottingCallbacks
from tests.test_shared import get_default_chess_database_path


def snapshot_path_creator(model_size: str, soft_prompt_token_count: int) -> str:
    return f"resnet_{model_size}_{soft_prompt_token_count}_testing_junk.pt"


def main():
    chess_database_path = get_default_chess_database_path()
    model_configurations = [('70m', 1)]
    soft_prompt_token_counts = [1]
    train_and_test_chess(chess_database_path, model_configurations, soft_prompt_token_counts,
                         ResnetFactory(16, 2, 32),
                         logging_prefix="resnet",
                         training_step_count=1,
                         batch_lanes_per_step=32,
                         maximum_sample_length_in_tokens=256, learning_rate=1e-3, weight_decay=1e-4,
                         forward_test_generated_token_count=128,
                         training_callbacks=SnapshottingCallbacks(snapshot_path_creator))

    result = torch.load("resnet_70m_1_testing_junk.pt")
    metadata = result[1]
    if metadata['soft_prompt_type'] == 'ResnetSoftPrompt':
        soft_prompt = soft_prompting.resnet_soft_prompt.ResnetSoftPrompt(**metadata['soft_prompt_metadata'])
        soft_prompt.load_state_dict(result[0])
        print(soft_prompt)


if __name__ == '__main__':
    main()
