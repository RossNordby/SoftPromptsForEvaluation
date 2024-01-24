from typing import Callable, Any

import torch
import zstandard as zstd
import chess.pgn
import io

from torch import Tensor

from pathfinding.pathfinding_dataset import PathfindingDataset
from soft_prompting.batch_loader import BatchLoader
import multiprocessing as mp
import queue as ye_olde_queue


# Instead of batchifying game generation, we'll just spawn some workers.
class AsyncPathfindingLoader:
    """
    Generates pathfinding boards asynchronously.
    """

    def __init__(self, board_width: int, board_height: int, insert_spaces: bool,
                 worker_count: int = 8, loader_queue_size: int = 2048):
        """
        Creates a AsyncPathfindingLoader.
        :param board_width: The width of the boards to generate.
        :param board_height: The height of the boards to generate.
        :param insert_spaces: Whether to insert spaces between each board slot and move.
        :param worker_count: The number of worker processes to use to generate boards.
        :param loader_queue_size: The size of the queue used to generate boards.
        """
        if loader_queue_size < worker_count * 2:
            raise ValueError("loader_queue_size should be significantly larger than the worker count.")
        self.board_width = board_width
        self.board_height = board_height
        self.insert_spaces = insert_spaces
        self.worker_count = worker_count
        self.loader_queue_size = loader_queue_size

    @staticmethod
    def board_generator_loop(board_width: int, board_height: int, insert_spaces: bool,
                             counter, worker_result_queue, stop_event: mp.Event):
        # We'll be using multiple workers; each worker's job is too small to warrant dispatching multithreaded work.
        torch.set_num_threads(1)
        dataset = PathfindingDataset(board_width, board_height, insert_spaces)
        while not stop_event.is_set():
            with counter.get_lock():
                index = counter.value
                counter.value += 1
            # We're using the request index as a random counter to maintain determinism.
            torch.manual_seed(index)
            dataset_result = next(dataset)
            worker_result_queue.put((dataset_result, index))

    @staticmethod
    def result_consumer_loop(worker_result_queue, result_queue, stop_event: mp.Event):
        """
        Asynchronous result consumer loop that takes results from the worker queue and puts them into the result queue.
        """
        pending_lanes = dict()
        next_lane = 0
        while not stop_event.is_set():
            result, index = worker_result_queue.get()
            pending_lanes[index] = result
            # Check if we can put any lanes into the result queue.
            if next_lane in pending_lanes:
                # Note that 'put' can block if the queue is full.
                # Like the board_generator_loop, the main process will flush the queue after setting the stop event.
                result_queue.put(pending_lanes[next_lane])
                del pending_lanes[next_lane]
                next_lane += 1

    def __enter__(self):
        self.manager = mp.Manager()
        self.counter = self.manager.Value('i', 0)
        self.worker_result_queue = self.manager.Queue(self.loader_queue_size)
        self.result_queue = self.manager.Queue(self.loader_queue_size)
        self.stop_event = mp.Event()
        self.board_generator_workers = [mp.Process(target=self.board_generator_loop,
                                                   args=(self.board_width, self.board_height, self.insert_spaces,
                                                         self.counter, self.worker_result_queue, self.stop_event),
                                                   daemon=True) for _ in range(self.worker_count)]
        self.result_consumer_process = mp.Process(target=self.result_consumer_loop,
                                                  args=(self.worker_result_queue, self.result_queue, self.stop_event),
                                                  daemon=True)
        for worker in self.board_generator_workers:
            worker.start()
        self.result_consumer_process.start()
        self.loaded_counter = 0
        return self

    def __iter__(self):
        return self

    def __next__(self):
        lane = self.result_queue.get()
        if lane is None:
            raise StopIteration
        return lane

    @staticmethod
    def flush_queue(queue):
        while True:
            try:
                queue.get(block=False)
            except ye_olde_queue.Empty:
                break

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_event.set()

        # Ensure that the workers can read the termination signal.
        self.flush_queue(self.worker_result_queue)
        self.flush_queue(self.result_queue)

        # Wait for the workers to terminate.
        for worker in self.board_generator_workers:
            worker.join()
        self.result_consumer_process.join()
        self.manager.shutdown()


class AsyncPathfindingBatchLoader(BatchLoader):
    """
    Loads batches of pathfinding training data.
    Outputs batches of games as a tuple:
    one containing lanes of [extra move count, invalid move count],
    another containing lanes of tokenized boards, and another containing lanes of move start indices.
    """

    def __init__(self, board_width, board_height, insert_spaces: bool, insert_move_section_separator: bool, tokenizer,
                 batch_size: int, maximum_sample_length_in_tokens: int = 512,
                 worker_count: int = 8, loader_queue_size: int = 2048,
                 num_processes_in_training: int = 1, process_index_in_training: int = 0):
        """
        Creates a AsyncPathfindingBatchLoader.
        :param board_width: The width of the boards to generate.
        :param board_height: The height of the boards to generate.
        :param insert_spaces: Whether to insert spaces between each board slot and move.
        :param insert_move_section_separator: Whether to insert a line that says 'Moves:' between the board and moves.
        :param tokenizer: The tokenizer to use to tokenize the moves.
        :param batch_size: The number of games to load per batch.
        :param maximum_sample_length_in_tokens: The maximum length of each sample in tokens.
        :param worker_count: The number of worker processes to use to parse and load games.
        :param loader_queue_size: The size of the queue used to load games.
        :param num_processes_in_training: The number of processes used in distributed training. This is used to split
        batches.
        :param process_index_in_training: The index of this process in distributed training.
        This is used to choose which part of the batch to use on this process.
        """
        super().__init__(batch_size, maximum_sample_length_in_tokens, num_processes_in_training,
                         process_index_in_training)
        self.tokenizer = tokenizer
        self.board_width = board_width
        self.board_height = board_height
        self.insert_spaces = insert_spaces
        self.insert_move_section_separator = insert_move_section_separator
        self.worker_count = worker_count
        self.loader_queue_size = loader_queue_size

    def __enter__(self):
        self.loader = AsyncPathfindingLoader(self.board_width, self.board_height, self.insert_spaces, self.worker_count,
                                             loader_queue_size=self.loader_queue_size)
        self.loader.__enter__()
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Any]:
        input_batch: list[tuple[int, int]] = []
        output_batch: list[str] = []
        while len(input_batch) < self.batch_size:
            elos, moves_string = next(self.loader)
            input_batch.append(elos)
            output_batch.append(moves_string)
        elo_tensor = torch.tensor(input_batch, dtype=torch.float)
        output_tensor = self.tokenizer(output_batch, padding=True, truncation=True,
                                       max_length=self.sample_length_in_tokens,
                                       return_tensors='pt').input_ids

        return elo_tensor, output_tensor, None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader.__exit__(exc_type, exc_val, exc_tb)

    @property
    def soft_prompt_parameters_size(self):
        return 2


class UnconditionalAsyncChessBatchLoader(AsyncChessBatchLoader):
    """
    Wraps an AsyncChessBatchLoader and lazily strips the soft prompt parameters out.
    Used for ablating the effect of Elo exposure during training.
    """

    def __init__(self, path: str, game_processor: Callable[[chess.pgn.Game], tuple[tuple[int, int], str]], tokenizer,
                 batch_size: int, maximum_sample_length_in_tokens: int = 512,
                 worker_count: int = 8, loader_queue_size: int = 2048,
                 num_processes_in_training: int = 1, process_index_in_training: int = 0):
        super().__init__(path, game_processor, tokenizer, batch_size, maximum_sample_length_in_tokens, worker_count,
                         loader_queue_size, num_processes_in_training, process_index_in_training)

    def __next__(self) -> tuple[None, Tensor]:
        return None, super().__next__()[1]

    def soft_prompt_parameters_size(self):
        return 0


class AsyncChessBatchLoaderWithOnes(AsyncChessBatchLoader):
    """
    Wraps an AsyncChessBatchLoader and lazily appends an extra constant soft prompt parameter of '1'.
    Used for investigating the impact of additional optimizable parameters, particularly in linear transforms
    that struggle to compete with MLP/ResNet.
    """

    def __init__(self, path: str, game_processor: Callable[[chess.pgn.Game], tuple[tuple[int, int], str]],
                 tokenizer,
                 batch_size: int, maximum_sample_length_in_tokens: int = 512,
                 worker_count: int = 8, loader_queue_size: int = 2048,
                 num_processes_in_training: int = 1, process_index_in_training: int = 0):
        super().__init__(path, game_processor, tokenizer, batch_size, maximum_sample_length_in_tokens, worker_count,
                         loader_queue_size, num_processes_in_training, process_index_in_training)

    def __next__(self) -> tuple[Tensor, Tensor]:
        soft_prompt_parameters, output_labels = super().__next__()
        soft_prompt_parameters = torch.cat((soft_prompt_parameters,
                                            torch.ones(soft_prompt_parameters.size(0), 1)), dim=1)
        return soft_prompt_parameters, output_labels

    @property
    def soft_prompt_parameters_size(self):
        return 3


# The following loaders are sequential. Careful; they're slow enough to matter.

class ChessGameLoader:
    """
    Loads chess games from a zstandard-compressed PGN file.
    """

    def __init__(self, path):
        """
        Creates a ChessGameLoader.
        :param path: The path to the zstandard-compressed database of games in PGN format.
        """
        self.path = path
        self.decompressor = zstd.ZstdDecompressor()

    def __enter__(self):
        self.compressed_pgn = open(self.path, 'rb')
        self.decompressed_stream = self.decompressor.stream_reader(self.compressed_pgn)
        self.text_stream = io.TextIOWrapper(self.decompressed_stream, encoding='utf-8')
        return self

    def __iter__(self):
        return self

    def __next__(self):
        return chess.pgn.read_game(self.text_stream)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.text_stream.close()
        self.decompressed_stream.close()
        self.compressed_pgn.close()


class ChessBatchLoader(BatchLoader):
    """
    Loads batches of chess games from a zstandard-compressed PGN file.
    Outputs batches of games as a tuple of two tensors:
    one containing lanes of [white Elo, black Elo], another containing lanes of tokenized moves in UCI format.
    """

    def __init__(self, path: str, tokenizer, batch_size: int, maximum_sample_length_in_tokens: int = 512,
                 allow_game_filter: Callable[[chess.pgn.Game], bool] = lambda _: True,
                 num_processes: int = 1, process_index: int = 0):
        """
        Creates a ChessBatchLoader.
        :param path: The path to the zstandard-compressed database of games in PGN format.
        :param tokenizer: The tokenizer to use to tokenize the moves.
        :param batch_size: The number of games to load per batch.
        :param maximum_sample_length_in_tokens: The maximum length of each sample in tokens.
        :param allow_game_filter: A function that determines whether a game should be included in a batch.
        """
        super().__init__(batch_size, maximum_sample_length_in_tokens, num_processes, process_index)
        self.tokenizer = tokenizer
        self.path = path
        self.allow_game_filter = allow_game_filter

    def __enter__(self):
        self.loader = ChessGameLoader(self.path)
        self.loader.__enter__()
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Any]:
        input_batch: list[tuple[int, int]] = []
        output_batch: list[str] = []
        while len(input_batch) < self.batch_size:
            game = next(self.loader)
            if game is None:
                raise StopIteration
            if self.allow_game_filter(game):
                # Convert the moves to a string and append it to the output batch.
                # Some games have zero moves! Skip those regardless of what the filter says.
                moves = ' '.join(move.uci() for move in game.mainline_moves())
                if len(moves) > 0:
                    input_batch.append((int(game.headers['WhiteElo']), int(game.headers['BlackElo'])))
                    output_batch.append(moves)
        input_tensor = torch.tensor(input_batch, dtype=torch.float)
        # Normalize elos a little bit.
        input_tensor = (input_tensor - 1000.0) / 2000.0
        output_tensor = self.tokenizer(output_batch, padding=True, truncation=True,
                                       max_length=self.sample_length_in_tokens,
                                       return_tensors='pt').input_ids
        return input_tensor, output_tensor, None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader.__exit__(exc_type, exc_val, exc_tb)

    @property
    def soft_prompt_parameters_size(self):
        return 2
