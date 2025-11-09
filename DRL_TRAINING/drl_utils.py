# DRL_TRAINING/drl_utils.py

import numpy as np
from numba import njit
from enum import Enum
from collections import deque
import copy

# Import GameResult from case_closed_game for convenience
from case_closed_game import GameResult

# --- Core Game Constants ---
GRID_HEIGHT = 18
GRID_WIDTH = 20
EMPTY = 0
AGENT_TRAIL = 1

# --- Direction Enums (used by all files) ---
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}
DIRS = {d.name: d.value for d in Direction}

# --- Numba Heuristic Functions ---
@njit
def _numba_wrap(x, y, w, h):
    return (x % w, y % h)

@njit
def _numba_corridor_penalty(board, x, y, w, h):
    empties = 0
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for i in range(4):
        dx, dy = local_dir_vectors[i]
        nx, ny = _numba_wrap(x + dx, y + dy, w, h)
        if board[ny, nx] == EMPTY:
            empties += 1
    
    if empties == 0: return -10.0 # Trapped (Scaled down for DRL)
    elif empties == 1: return -5.0   # Dead end
    elif empties == 2: return -0.5   # Corridor
    return 0.0

@njit
def _numba_voronoi(board, my_head_tuple, opp_head_tuple, w, h):
    my_head = np.array([my_head_tuple[0], my_head_tuple[1]], dtype=np.int16)
    opp_head = np.array([opp_head_tuple[0], opp_head_tuple[1]], dtype=np.int16)
    
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
    q = np.empty((w * h * 2, 4), dtype=np.int16)
    q_head, q_tail = 0, 0
    owner_grid = np.zeros((h, w), dtype=np.int16)
    dist_grid = np.full((h, w), 9999, dtype=np.int16)

    q[q_tail] = np.array([my_head[0], my_head[1], 1, 0], dtype=np.int16)
    dist_grid[my_head[1], my_head[0]] = 0
    owner_grid[my_head[1], my_head[0]] = 1
    q_tail += 1

    q[q_tail] = np.array([opp_head[0], opp_head[1], 2, 0], dtype=np.int16)
    dist_grid[opp_head[1], opp_head[0]] = 0
    if owner_grid[opp_head[1], opp_head[0]] == 0:
        owner_grid[opp_head[1], opp_head[0]] = 2
    q_tail += 1

    while q_head < q_tail:
        x, y, player, dist = q[q_head]
        q_head += 1
        if dist > dist_grid[y, x]: continue
        for i in range(4):
            dx, dy = local_dir_vectors[i]
            nx, ny = _numba_wrap(x + dx, y + dy, w, h)
            if board[ny, nx] == EMPTY:
                new_dist = dist + 1
                if new_dist < dist_grid[ny, nx]:
                    dist_grid[ny, nx] = new_dist
                    owner_grid[ny, nx] = player
                    q[q_tail] = np.array([nx, ny, player, new_dist], dtype=np.int16)
                    q_tail += 1
                elif new_dist == dist_grid[ny, nx] and owner_grid[ny, nx] != player:
                    owner_grid[ny, nx] = 0
    return np.sum(owner_grid == 1), np.sum(owner_grid == 2)

# --- Shared Python Helper Functions ---

def _torus_check(pos):
    return (pos[0] % GRID_WIDTH, pos[1] % GRID_HEIGHT)

def flood_fill(board, start_pos, my_id):
    visited = set()
    visited.add(start_pos)
    q = deque([start_pos])
    count = 0
    while q:
        x, y = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_pos = _torus_check((x + dx, y + dy))
            next_x, next_y = next_pos
            if next_pos not in visited and board[next_y, next_x] == EMPTY:
                visited.add(next_pos)
                q.append(next_pos)
                count += 1
    return count

def count_escape_routes(board, pos):
    safe_directions = 0
    for direction in Direction:
        dx, dy = direction.value
        next_pos = _torus_check((pos[0] + dx, pos[1] + dy))
        if board[next_pos[1], next_pos[0]] == EMPTY:
            safe_directions += 1
    return safe_directions

def get_possible_moves(agent_dir, boosts_remaining):
    valid_moves = []
    for direction in Direction:
        if direction != OPPOSITE_DIR.get(agent_dir):
            valid_moves.append((direction, False))
            if boosts_remaining > 0:
                valid_moves.append((direction, True))
    return valid_moves

def simulate_move(current_state, player_id, direction, use_boost):
    new_state = copy.deepcopy(current_state)
    
    if player_id == 1:
        my_trail = new_state["agent1_trail"]
        my_boosts_key = "agent1_boosts"
        my_dir_key = "my_direction"
    else:
        my_trail = new_state["agent2_trail"]
        my_boosts_key = "agent2_boosts"
        my_dir_key = "opp_direction"

    num_steps = 1
    if use_boost and new_state[my_boosts_key] > 0:
        num_steps = 2
        new_state[my_boosts_key] -= 1
    
    agent_alive = True
    current_head = my_trail[-1]
    
    for _ in range(num_steps):
        if not agent_alive: break
        dx, dy = direction.value
        next_head = _torus_check((current_head[0] + dx, current_head[1] + dy))
        if new_state["board"][next_head[1], next_head[0]] == AGENT_TRAIL:
            agent_alive = False
            break
        new_state["board"][next_head[1], next_head[0]] = AGENT_TRAIL
        my_trail.append(next_head)
        current_head = next_head
        
    if agent_alive:
        new_state[my_dir_key] = direction
    
    if "turn_count" in new_state:
        new_state["turn_count"] += 1
        
    return new_state, agent_alive

def calculate_center_distance(pos):
    return abs(pos[0] - GRID_WIDTH // 2) + abs(pos[1] - GRID_HEIGHT // 2)

def calculate_opponent_distance(my_head, opp_head):
    return abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])