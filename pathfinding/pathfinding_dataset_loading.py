import torch

from pathfinding.pathfinding_dataset import PathfindingDataset
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
                             counter, lock, worker_result_queue, stop_event: mp.Event):
        # We'll be using multiple workers; each worker's job is too small to warrant dispatching multithreaded work.
        torch.set_num_threads(1)
        dataset = PathfindingDataset(board_width, board_height, insert_spaces)
        while not stop_event.is_set():
            with lock:
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
                # print(f'pending lane count: {len(pending_lanes)}')
                # print(f' count in worker queue: {worker_result_queue.qsize()}')
                # print(f' count in result queue: {result_queue.qsize()}')
                result_queue.put(pending_lanes[next_lane])
                del pending_lanes[next_lane]
                next_lane += 1

    def __enter__(self):
        self.manager = mp.Manager()
        self.counter = self.manager.Value('i', 0)
        self.lock = self.manager.Lock()
        self.worker_result_queue = self.manager.Queue(self.loader_queue_size)
        self.result_queue = self.manager.Queue(self.loader_queue_size)
        self.stop_event = mp.Event()
        self.board_generator_workers = [mp.Process(target=self.board_generator_loop,
                                                   args=(self.board_width, self.board_height, self.insert_spaces,
                                                         self.counter, self.lock, self.worker_result_queue,
                                                         self.stop_event),
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
