from typing import Callable, Any

import torch
import zstandard as zstd
import chess.pgn
import io

from torch import Tensor

from soft_prompting.batch_loader import BatchLoader
import multiprocessing as mp
import queue as ye_olde_queue


# Unfortunately, parsing pgn into games is slow enough that it's actually a bottleneck for training.
# To compensate, we use a multiprocess loader.
class AsyncChessGameLoader:
    """
    Loads chess games from a zstandard-compressed PGN file. Uses an asynchronous loader loop.
    Games are reported in the order they appear in the file.
    """

    def __init__(self, path, game_processor: Callable[[chess.pgn.Game], Any], worker_count: int = 8,
                 loader_queue_size: int = 2048):
        """
        Creates a ChessGameLoader.
        :param path: The path to the zstandard-compressed database of games in PGN format.
        :param game_processor: A function that processes a game and returns a result.
                               The function and its result must be pickleable.
        :param worker_count: The number of worker processes to use to parse games.
        :param loader_queue_size: The size of the queue used to load games.
        """
        if loader_queue_size < worker_count * 2:
            raise ValueError("loader_queue_size should be significantly larger than the worker count.")
        self.path = path
        self.game_processor = game_processor
        self.worker_count = worker_count
        self.loader_queue_size = loader_queue_size

    @staticmethod
    def read_until_end_of_game(text_stream: io.TextIOWrapper):
        """
        Reads the next game from the text stream.
        """
        lines = []
        for line in text_stream:
            lines.append(line)
            if line.startswith('1.'):
                break
        return ''.join(lines)

    @staticmethod
    def text_loader_loop(path, to_parse_queue, stop_event: mp.Event):
        """
        Asynchronous loader loop that loads games from the PGN file and puts them into the lane queue.
        """
        decompressor = zstd.ZstdDecompressor()
        game_count = 0
        with open(path, 'rb') as compressed_pgn, \
                decompressor.stream_reader(compressed_pgn) as decompressed_stream, \
                io.TextIOWrapper(decompressed_stream, encoding='utf-8') as text_stream:
            while not stop_event.is_set():
                # Read the next game. Note that we do not parse it within this process; parsing is slow.
                next_game_text = AsyncChessGameLoader.read_until_end_of_game(text_stream)
                # Note that 'put' can block if the queue is full.
                # There's no reason to load games much faster than they can be parsed.
                # This requires care, though: if the queue is full, we'll block here and never check the stop event.
                # So, the main process will flush the queue after setting the stop event.
                # (The only other thread that puts something in this queue is the main process's termination signal.)
                # game_count is provided so that, when the worker returns the results, we can stick them back in order.
                # We want the dataset to be deterministic!
                to_parse_queue.put((game_count, next_game_text))
                game_count += 1

    @staticmethod
    def parser_worker_loop(game_processor: Callable[[chess.pgn.Game], Any],
                           to_parse_queue,
                           worker_result_queue, ):
        while True:
            game_index, game_text = to_parse_queue.get()
            if game_index is None:
                # Terminate signal. The main process pushes worker_count None values to the queue to signal termination.
                break
            game = chess.pgn.read_game(io.StringIO(game_text))
            result = game_processor(game)
            worker_result_queue.put((game_index, result))

    @staticmethod
    def result_consumer_loop(worker_result_queue,
                             result_queue,
                             stop_event: mp.Event):
        """
        Asynchronous result consumer loop that takes results from the worker queue and puts them into the result queue.
        """
        pending_lanes = dict()
        next_lane = 0
        while not stop_event.is_set():
            game_index, result = worker_result_queue.get()
            if game_index is None:
                # Terminate signal. The main process pushes worker_count None values to the queue to signal termination.
                break
            pending_lanes[game_index] = result
            # Check if we can put any lanes into the result queue.
            if next_lane in pending_lanes:
                # Note that 'put' can block if the queue is full.
                # Like the text_loader_loop, the main process will flush the queue after setting the stop event.
                result = pending_lanes[next_lane]
                # A null result doesn't mean we need to terminate; the processor could have simply filtered the game.
                if result is not None:
                    result_queue.put(result)
                del pending_lanes[next_lane]
                next_lane += 1

    def __enter__(self):
        # We'll be managing the semaphores ourselves for some extra flexibility and less magic.
        self.manager = mp.Manager()

        self.to_parse_queue = self.manager.Queue(self.loader_queue_size)
        self.worker_result_queue = self.manager.Queue(self.loader_queue_size)
        self.result_queue = self.manager.Queue(self.loader_queue_size)
        self.stop_event = mp.Event()
        self.text_loader_process = mp.Process(target=self.text_loader_loop,
                                              args=(self.path, self.to_parse_queue, self.stop_event), daemon=True)
        self.parser_workers = [mp.Process(target=self.parser_worker_loop,
                                          args=(self.game_processor, self.to_parse_queue, self.worker_result_queue),
                                          daemon=True) for _ in range(self.worker_count)]
        self.result_consumer_process = mp.Process(target=self.result_consumer_loop,
                                                  args=(self.worker_result_queue, self.result_queue, self.stop_event),
                                                  daemon=True)
        self.text_loader_process.start()
        for worker in self.parser_workers:
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

        # Ensure that the text_loader_process can check the stop event. Only the text_loader_process pushes to the
        # to_parse_queue, so opening up any space is sufficient to let the worker check the event.
        self.flush_queue(self.to_parse_queue)
        # Push termination signals for the text loader.
        self.text_loader_process.join()

        # Similar to the above, ensure that the parser workers can read the termination signal.
        self.flush_queue(self.worker_result_queue)
        self.flush_queue(self.result_queue)

        # Push termination signals for the parser workers.
        for _ in range(self.worker_count):
            self.to_parse_queue.put((None, None))

        # Wait for the parser workers to terminate.
        for worker in self.parser_workers:
            worker.join()

        # Only remaining worker process is the result consumer. No new worker results are being added.
        # Ensure there's room in the result queue so the consumer can terminate.
        # Push termination signals for the result consumer.
        self.worker_result_queue.put((None, None))
        self.result_consumer_process.join()
        self.manager.shutdown()


class AsyncChessBatchLoader(BatchLoader):
    """
    Loads batches of chess games from a zstandard-compressed PGN file.
    Outputs batches of games as a tuple of two tensors:
    one containing lanes of [white Elo, black Elo], another containing lanes of tokenized moves in UCI format.
    """

    def __init__(self, path: str, game_processor: Callable[[chess.pgn.Game], tuple[tuple[int, int], str]], tokenizer,
                 batch_size: int, maximum_sample_length_in_tokens: int = 512,
                 worker_count: int = 8, loader_queue_size: int = 2048,
                 num_processes_in_training: int = 1, process_index_in_training: int = 0):
        """
        Creates a AsyncChessBatchLoader.
        :param path: The path to the zstandard-compressed database of games in PGN format.
        :param tokenizer: The tokenizer to use to tokenize the moves.
        :param batch_size: The number of games to load per batch.
        :param maximum_sample_length_in_tokens: The maximum length of each sample in tokens.
        :param game_processor: A pickleable function that processes a game into a batch lane.
        Takes a game and returns a tuple of ((white elo, black elo), moves_string).
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
        self.path = path
        self.game_processor = game_processor
        self.worker_count = worker_count
        self.loader_queue_size = loader_queue_size

    def __enter__(self):
        self.loader = AsyncChessGameLoader(self.path, self.game_processor, self.worker_count,
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
        # Normalize elos a little bit.
        elo_tensor = (elo_tensor - 1000.0) / 2000.0
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
