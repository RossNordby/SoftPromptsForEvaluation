import numpy
import torch


class PathfindingDataset:
    """
    Generates pathfinding boards and their associated move sequences.
    """
    BLOCKER_VALUE = 999999

    def __init__(self, board_width: int, board_height: int, insert_spaces: bool):
        self.width = board_width
        self.height = board_height
        self.insert_spaces = insert_spaces

    def __iter__(self):
        return self

    def get_minimal_neighbors(self, distances: torch.Tensor):
        """
        Gathers the 3x3 region of slots around each slot in the given tensor.
        Returns a tuple of the minimum value in each neighborhood and the index of that minimum value.
        """
        padded = torch.nn.functional.pad(distances, (1, 1, 1, 1), mode='constant',
                                         value=PathfindingDataset.BLOCKER_VALUE).unsqueeze(0)

        # Note clone; we'll be writing to this tensor, and unfold shares memory sometimes.
        slot_neighborhoods = torch.nn.functional.unfold(padded.to(torch.float), kernel_size=3).clone()
        # We've now got a tensor with width * height rows, each row being the neighborhood of a slot.
        # Set all even columns to 99999; ignores diagonal moves.
        # This also ignores the center slot, but that's fine; we can grab that from the original tensor.
        slot_neighborhoods[::2, :] = PathfindingDataset.BLOCKER_VALUE
        result = torch.min(slot_neighborhoods, dim=0)
        return (result.values.view(self.width, self.height).to(torch.int),
                result.indices.view(self.width, self.height))

    def propagate_distances_step(self, distances: torch.Tensor, blocker_mask: torch.Tensor):
        """
        Gather direct neighbors (left, right, up, down) in a 2D tensor.
        """
        minimal_neighbors = self.get_minimal_neighbors(distances)[0]
        new_distances = torch.min(minimal_neighbors + 1, distances)
        # Don't update blockers!
        blocked_distances = new_distances.masked_fill(blocker_mask, PathfindingDataset.BLOCKER_VALUE)

        return blocked_distances

    def get_board_optimal_moves(self, distances: torch.Tensor):
        """
        Returns a tensor of optimal moves for each slot in the given distances tensor.
        Values are 0 for left, 1 for right, 2 for down, 3 for up, -1 for stuck.
        """
        minimal_neighbors, minimal_neighbor_indices = self.get_minimal_neighbors(distances)
        maximum_valid_distance = distances.size(0) * distances.size(1)
        doomed_mask = distances > maximum_valid_distance
        up_mask = minimal_neighbor_indices == 1
        down_mask = minimal_neighbor_indices == 7
        right_mask = minimal_neighbor_indices == 5
        left_mask = minimal_neighbor_indices == 3
        result = torch.empty([self.width, self.height], dtype=torch.int)
        stuck_mask = minimal_neighbors > maximum_valid_distance
        result = torch.masked_fill(result, left_mask, 0)
        result = torch.masked_fill(result, right_mask, 1)
        result = torch.masked_fill(result, down_mask, 2)
        result = torch.masked_fill(result, up_mask, 3)
        result = torch.masked_fill(result, doomed_mask | stuck_mask, -1)
        return result

    def __next__(self):
        # A lot of this *could* be vectorized over a whole batch, but... it's not bottlenecking anything at the moment.
        distances_to_target = torch.empty([self.width, self.height], dtype=torch.int)
        torch.fill_(distances_to_target, PathfindingDataset.BLOCKER_VALUE)
        location_count = torch.randint(0, self.width * self.height * 3 // 4, [1]) + 2
        assert location_count <= self.width * self.height  # Just assuming we're using large enough boards.
        random_location_indices = torch.randperm(distances_to_target.numel())[:location_count]
        # Use the first location as the target.
        flat_board_distances = distances_to_target.view(-1)
        target_location_index = random_location_indices[0]
        flat_board_distances[target_location_index] = 0
        # Use the second location as the start.
        start_location_index = random_location_indices[1]
        # Use the rest as blockers.
        blocker_mask = torch.zeros_like(distances_to_target, dtype=torch.bool)
        flat_blocker_mask = blocker_mask.view(-1)
        flat_blocker_mask[random_location_indices[2:]] = True
        flat_board_distances[random_location_indices[2:]] = PathfindingDataset.BLOCKER_VALUE

        # print(distances_to_target)

        while True:
            # Rather than using a queue and BFS, we'll just do a series of tensorwide propagation steps.
            # (Didn't do a baseline for comparison; it's fast enough, shrug.)
            new_distances_to_target = self.propagate_distances_step(distances_to_target, blocker_mask)
            # If nothing changed, we're done.
            if torch.all(new_distances_to_target == distances_to_target):
                break
            distances_to_target = new_distances_to_target
            # print(distances_to_target)

        # The board's now set up for pathfinding.
        board_moves = self.get_board_optimal_moves(distances_to_target)
        flat_board_moves = board_moves.view(-1)
        location = start_location_index.item()
        # We output the 'extra' move count as a condition for the model, so we need to know the optimal.
        optimal_move_count = 0
        while location != target_location_index:
            move = flat_board_moves[location]
            if move < 0:
                # "U" for unsolvable.
                break
            if move == 0:
                location -= 1
            elif move == 1:
                location += 1
            elif move == 2:
                location += self.width
            elif move == 3:
                location -= self.width
            optimal_move_count += 1

        moves = []

        inverse_wiggle_propensity = torch.randint(2, 8, [1]).item()
        wall_hit_count = 0
        move_count = 0
        location = start_location_index.item()
        # debug_move_tracker = torch.zeros_like(distances_to_target, dtype=torch.int)
        # debug_move_tracker = torch.masked_fill(debug_move_tracker, blocker_mask, -1)
        while location != target_location_index:
            if torch.randint(0, inverse_wiggle_propensity, [1]) == 0:
                # Randomly choose a different move.
                # Note that random moves can be invalid; they might bump into a wall or try to go off the board.
                # This is something we want to capture as a part of training.
                move = torch.randint(0, 4, [1])
                x, y = location % self.width, location // self.width
                if move == 0:
                    moves.append('W')
                    x = max(0, x - 1)
                elif move == 1:
                    moves.append('E')
                    x = min(self.width - 1, x + 1)
                elif move == 2:
                    moves.append('S')
                    y = min(self.height - 1, y + 1)
                else:
                    moves.append('N')
                    y = max(0, y - 1)
                candidate_location = y * self.width + x
                if flat_blocker_mask[candidate_location]:
                    # Blocked.
                    candidate_location = location
                if candidate_location == location:
                    # Either hit the border or a blocker.
                    wall_hit_count += 1
                else:
                    location = candidate_location

            else:
                # Choose the optimal move.
                move = flat_board_moves[location]
                if move == 0:
                    moves.append('W')
                    location -= 1
                elif move == 1:
                    moves.append('E')
                    location += 1
                elif move == 2:
                    moves.append('S')
                    location += self.width
                elif move == 3:
                    moves.append('N')
                    location -= self.width
                else:
                    # "U" for unsolvable.
                    moves.append('U')
                    break
            move_count += 1
            # debug_move_tracker.view(-1)[location] = move_count
            # print(debug_move_tracker)
        else:
            # "D" for done.
            moves.append('D')
        join_string = ' ' if self.insert_spaces else ''
        move_string = join_string.join(moves)

        ascii_tensor = torch.where(blocker_mask, torch.full_like(blocker_mask, ord('X'), dtype=torch.uint8),
                                   torch.full_like(blocker_mask, ord('O'), dtype=torch.uint8))
        flat_ascii_tensor = ascii_tensor.view(-1)
        flat_ascii_tensor[start_location_index] = ord('A')
        flat_ascii_tensor[target_location_index] = ord('B')
        ascii_array = ascii_tensor.numpy()
        board_string = '\n'.join(join_string.join(map(chr, row)) for row in ascii_array)

        return board_string, move_string, move_count - optimal_move_count, wall_hit_count
