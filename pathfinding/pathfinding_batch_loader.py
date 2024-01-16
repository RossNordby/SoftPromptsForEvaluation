from typing import Any

import torch
from torch import Tensor

from pathfinding.pathfinding_dataset import PathfindingDataset
from soft_prompting.batch_loader import BatchLoader
from soft_prompting.utils import get_token_counts


class PathfindingBatchLoader(BatchLoader):
    """
    Loads batches of pathfinding boards and their associated move data.
    Soft prompt parameters contain the number of extra steps and invalid steps.
    """

    def __init__(self, board_width: int, board_height: int, insert_spaces: bool, tokenizer, batch_size: int,
                 maximum_sample_length_in_tokens: int = 512,
                 num_processes: int = 1, process_index: int = 0):
        """
        Creates a PathfindingBatchLoader.
        :param board_width: The width of the boards to generate.
        :param board_height: The height of the boards to generate.
        :param insert_spaces: Whether to insert spaces between each board slot and move.
                              May affect tokenization.
        :param tokenizer: The tokenizer to use to tokenize the moves.
        :param batch_size: The number of games to load per batch.
        :param maximum_sample_length_in_tokens: The maximum length of each sample in tokens.
        """
        super().__init__(batch_size, maximum_sample_length_in_tokens, num_processes, process_index)
        self.board_dataset = PathfindingDataset(board_width, board_height, insert_spaces)
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def __next__(self) -> tuple[Tensor, Tensor, Any]:
        input_batch: list[tuple[int, int]] = []
        board_batch: list[str] = []
        moves_batch: list[str] = []
        while len(input_batch) < self.batch_size:
            board, moves, extra_move_count, invalid_move_count = next(self.board_dataset)
            input_batch.append((extra_move_count, invalid_move_count))
            board_batch.append(board + '\n')
            moves_batch.append(moves)
        tokenized_boards = self.tokenizer(board_batch, return_tensors='pt', padding=True, truncation=True,
                                          max_length=self.sample_length_in_tokens).input_ids
        tokenized_moves = self.tokenizer(moves_batch, return_tensors='pt', padding=True, truncation=True,
                                         max_length=self.sample_length_in_tokens).input_ids
        move_start_indices = get_token_counts(tokenized_boards, self.tokenizer.eos_token_id,
                                              self.tokenizer.pad_token_id)
        # Create a tensor containing the boards *and* moves by inserting the moves into the boards.
        combined_token_count = tokenized_boards.size(1) + tokenized_moves.size(1)
        if combined_token_count > self.sample_length_in_tokens:
            # If the combined token count is too long, truncate the moves.
            assert tokenized_boards.size(1) <= self.sample_length_in_tokens
            tokenized_moves = tokenized_moves[:, :self.sample_length_in_tokens - tokenized_boards.size(1)]
        combined_ids = torch.empty((len(input_batch), tokenized_boards.size(1) + tokenized_moves.size(1)),
                                   dtype=torch.long)
        combined_ids = torch.fill(combined_ids, self.tokenizer.pad_token_id)
        combined_ids[:, :tokenized_boards.size(1)] = tokenized_boards
        # Use the move_start_indices to insert moves into the combined input ids;
        # we can use fancy advanced indexing for this. I'm absolutely going to forget how this works.
        # The first index is the batch indices, simple enough.
        # The second index, for each batch lane, provides a row of indices to insert the moves at.
        # The tensor window pointed at by the indexing is [batch_size, move_tokens_count] which tokenized_moves
        # can be shoved into.
        combined_ids[torch.arange(len(input_batch)).unsqueeze(1), move_start_indices.unsqueeze(1) +
                                                                  torch.arange(
                                                                      tokenized_moves.size(1))] = tokenized_moves
        input_tensor = torch.tensor(input_batch, dtype=torch.float)
        return input_tensor, combined_ids, move_start_indices

    @property
    def soft_prompt_parameters_size(self):
        return 2
