import hashlib
import time

from transformers import AutoTokenizer

from pathfinding import PathfindingBatchLoader
from pathfinding.pathfinding_dataset import PathfindingDataset
from pathfinding.async_pathfinding_loader import AsyncPathfindingLoader


def hash_tensor(tensor, truncation_bits=32):
    tensor = tensor.cpu()
    byte_representation = tensor.numpy().tobytes()
    hash_object = hashlib.sha256(byte_representation)
    hash_hex = hash_object.hexdigest()
    hash_int = int(hash_hex, 16)
    truncation_mask = (1 << truncation_bits) - 1
    return hash_int & truncation_mask

def main():
    """
    Tests the pathfinding batch loader performance or determinism.
    """

    board_loader = AsyncPathfindingLoader(8, 8, False, worker_count=4, loader_queue_size=2048)
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
    tokenizer.pad_token = tokenizer.eos_token
    board_loader = PathfindingDataset(8, 8, False)
    batch_loader = PathfindingBatchLoader(board_loader, True, tokenizer, 256)
    # with board_loader:
    hash = 0
    start = time.perf_counter()
    for i in range(100):
        input_conditions, tokens, move_start_indices = next(batch_loader)
        # hash = hash * 7919 + hash_tensor(input_conditions)
        # hash = hash * 4973 + hash_tensor(tokens)
        # hash = hash * 5197 + hash_tensor(move_start_indices)
        # hash = hash & ((1 << 32) - 1)
        # print(hash)
        # time.sleep(0.5)
        # print(next(batch_loader))
    end = time.perf_counter()
    print(f"Time taken: {end - start}")



if __name__ == '__main__':
    main()
