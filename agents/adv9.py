import os
import time
import copy
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
from enum import Enum
import sys
from numba import njit # <-- Make sure numba is imported

# Set a higher recursion limit for deep minimax search
sys.setrecursionlimit(2500)

# --- Core Game Constants ---
class Direction(Enum):
    UP = (0, -1); DOWN = (0, 1); RIGHT = (1, 0); LEFT = (-1, 0)
DIRECTION_MAP = {d.name: d for d in Direction}
OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}
GRID_HEIGHT = 18; GRID_WIDTH = 20; EMPTY = 0; AGENT = 1; AGENT_TRAIL = 1
MAX_TURNS = 200 # Correct max turns from case_closed_game.py
MAX_SEARCH_DEPTH = 30 # Max depth for iterative deepening
MAX_TIME_MS = 3800  # 3.8 seconds max per move (4s limit)

# Helper function for torus wrapping
def _torus_check(pos):
    """Handles the wraparound logic for the toroidal grid."""
    x, y = pos
    return (x % GRID_WIDTH, y % GRID_HEIGHT)

# --- Transposition Table (Cache) ---
TT_CACHE = {}
START_TIME = 0.0

# --- Dynamic Weight Profiles ---
WEIGHT_PROFILES = {
    "STANDARD": {
        "space": 1.0, "center": 0.2, "pressure": 0.6, "trapping": 2.0, 
        "openness": 0.8, "my_corridor": 1.0, "opp_corridor": 1.0, "length": 0.1
    },
    "COUNTER_AGGRESSIVE": {
        "space": 1.5, "center": 0.1, "pressure": 0.5, "trapping": 3.0, 
        "openness": 1.0, "my_corridor": 1.5, "opp_corridor": 1.5, "length": 0.1
    },
    "PUNISH_DEFENSIVE": {
        "space": 2.0, "center": 1.0, "pressure": 0.1, "trapping": 0.5, 
        "openness": 1.0, "my_corridor": 1.0, "opp_corridor": 0.5, "length": 0.1
    },
    "AGGRESSIVE_KILL": {
        "space": 0.5, "center": 0.0, "pressure": 2.0, "trapping": 3.0, 
        "openness": 0.0, "my_corridor": 0.5, "opp_corridor": 5.0, "length": 0.0
    },
    "DESPERATE_SURVIVAL": {
        "space": 3.0, "center": 0.0, "pressure": 0.0, "trapping": 0.0, 
        "openness": 2.0, "my_corridor": 5.0, "opp_corridor": 0.0, "length": 0.0
    }
}
CURRENT_PROFILE = "STANDARD"

# --- Numba-Optimized Helper Functions ---
@njit(cache=False)
def _numba_wrap(x, y, w, h):
    return (x % w, y % h)

@njit(cache=False)
def _numba_count_routes(board, x, y, w, h):
    empties = 0
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
    for i in range(4):
        dx, dy = local_dir_vectors[i]
        nx, ny = _numba_wrap(x + dx, y + dy, w, h)
        if board[ny, nx] == EMPTY:
            empties += 1
    return empties

@njit(cache=False)
def _numba_corridor_penalty(board, x, y, w, h):
    empties = _numba_count_routes(board, x, y, w, h)
    if empties == 0: return -100000.0
    elif empties == 1: return -5000.0
    elif empties == 2: return -500.0 # Strong penalty for 2-wide
    elif empties == 3: return -1.0
    else: return 0.0

# @njit(cache=False)
# def _numba_reachable_area(board, start_x, start_y, w, h, cutoff):
#     q = np.empty((w * h, 2), dtype=np.int16)
#     q_head, q_tail = 0, 0
#     q[q_tail] = np.array([start_x, start_y], dtype=np.int16)
#     q_tail += 1
#     seen = np.zeros((h, w), dtype=np.bool_)
#     seen[start_y, start_x] = True
#     count = 0
#     local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
#     while q_head < q_tail and count < cutoff:
#         x, y = q[q_head]
#         q_head += 1
#         for i in range(4):
#             dx, dy = local_dir_vectors[i]
#             nx, ny = _numba_wrap(x + dx, y + dy, w, h)
#             if not seen[ny, nx] and board[ny, nx] == EMPTY:
#                 seen[ny, nx] = True
#                 q[q_tail] = np.array([nx, ny], dtype=np.int16)
#                 q_tail += 1
#                 count += 1
#     return count

# --- Heuristic Functions ---
# Add these functions to adv9.py, inside the "--- Heuristic Functions ---" section.
# (These are copied from ad3.py)

def calculate_voronoi_space(board, my_head, opp_head):
    my_space = 0; opp_space = 0
    q_my = deque([(my_head, 1)]); q_opp = deque([(opp_head, 1)])
    visited = {}; visited[my_head] = (1, 0); visited[opp_head] = (2, 0)
    are_separated = True
    while q_my or q_opp:
        if q_my:
            my_curr, my_dist = q_my.popleft()
            for direction in Direction:
                dx, dy = direction.value
                next_pos = _torus_check((my_curr[0] + dx, my_curr[1] + dy))
                if board[next_pos[1], next_pos[0]] == EMPTY:
                    if next_pos not in visited:
                        visited[next_pos] = (1, my_dist); my_space += 1
                        q_my.append((next_pos, my_dist + 1))
                    elif visited[next_pos][0] == 2 and my_dist < visited[next_pos][1]:
                        are_separated = False
        if q_opp:
            opp_curr, opp_dist = q_opp.popleft()
            for direction in Direction:
                dx, dy = direction.value
                next_pos = _torus_check((opp_curr[0] + dx, opp_curr[1] + dy))
                if board[next_pos[1], next_pos[0]] == EMPTY:
                    if next_pos not in visited:
                        visited[next_pos] = (2, opp_dist); opp_space += 1
                        q_opp.append((next_pos, opp_dist + 1))
                    elif visited[next_pos][0] == 1 and opp_dist == visited[next_pos][1]:
                        are_separated = False
    if are_separated and (my_space > 0 or opp_space > 0):
        my_space = flood_fill(board, my_head)
        opp_space = flood_fill(board, opp_head)
    return my_space, opp_space, are_separated

def flood_fill(board, start_pos):
    visited = set(); q = deque()
    if board[start_pos[1], start_pos[0]] == EMPTY:
        q.append(start_pos); visited.add(start_pos); count = 1
    else:
        q.append(start_pos); visited.add(start_pos); count = 0
    while q:
        x, y = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = _torus_check((x + dx, y + dy))
            next_pos = (next_x, next_y)
            if next_pos not in visited and board[next_y, next_x] == EMPTY:
                visited.add(next_pos); q.append(next_pos); count += 1
    return count

def calculate_center_distance(pos, w, h):
    return abs(pos[0] - w // 2) + abs(pos[1] - h // 2)

def calculate_center_bonus(my_head, opp_head, my_id, turn_count, w, h):
    my_dist = calculate_center_distance(my_head, w, h)
    opp_dist = calculate_center_distance(opp_head, w, h)
    phase_weight = max(0, 1.0 - turn_count / 100.0)
    if my_id == 1: my_dist -= 0.5
    else: my_dist += 0.5
    return (opp_dist - my_dist) * 2.0 * phase_weight

def calculate_opponent_distance(my_head, opp_head):
    return abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])

def calculate_pressure_score(my_head, opp_head, turn_count):
    distance = calculate_opponent_distance(my_head, opp_head)
    if turn_count < 12: return 0
    if 5 <= distance <= 10: return 5.0
    elif distance < 5: return -2.0
    elif distance > 15: return -3.0
    else: return 0

def calculate_escape_quality(my_routes, opp_routes):
    route_advantage = (my_routes - opp_routes) * 4.0
    if opp_routes == 0 and my_routes > 0: route_advantage += 100.0
    elif opp_routes == 1 and my_routes > 2: route_advantage += 50.0
    elif opp_routes <= 2 and my_routes > 3: route_advantage += 25.0
    if my_routes >= 4: route_advantage += 10.0
    elif my_routes == 3: route_advantage += 5.0
    return route_advantage

def detect_opponent_vulnerability(opp_space, total_space):
    if total_space == 0: return 0
    opp_percentage = opp_space / total_space
    if opp_percentage < 0.1: return 50.0
    elif opp_percentage < 0.2: return 25.0
    elif opp_percentage < 0.3: return 10.0
    return 0

def calculate_trapping_bonus(my_routes, opp_routes, my_space, opp_space):
    total_space = my_space + opp_space
    vulnerability = detect_opponent_vulnerability(opp_space, total_space)
    escape_quality = calculate_escape_quality(my_routes, opp_routes)
    return vulnerability + escape_quality

def calculate_openness_bonus(my_space, board_size):
    openness_ratio = my_space / board_size
    if openness_ratio > 0.35: return 15.0
    elif openness_ratio > 0.25: return 10.0
    elif openness_ratio > 0.15: return 0
    elif openness_ratio > 0.08: return -15.0
    else: return -40.0

# --- Fused Heuristic Evaluation Function (LAZY) ---
def evaluate_state(state: dict, weights: dict):
    """
    Hybrid evaluation function combining Voronoi space (from ad3)
    with adv9's heuristics and new survival logic.
    """
    board = state["board"]
    turn_count = state.get("turn_count", 0)
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    w, h = GRID_WIDTH, GRID_HEIGHT
    my_id = state["player_number"]
    my_len = len(state["agent1_trail"])
    opp_len = len(state["agent2_trail"])
    my_boosts = state["agent1_boosts"] # Get boosts for ad3 logic

    # --- 1. Absolute Priority: Endgame Length Battle ---
    if turn_count > MAX_TURNS - 20:
        return (my_len - opp_len) * 10000.0

    # --- 2. Fast "Lazy" Heuristics ---
    my_routes = _numba_count_routes(board, my_head[0], my_head[1], w, h)
    opp_routes = _numba_count_routes(board, opp_head[0], opp_head[1], w, h)

    # --- 3. "SURVIVAL LAYERS" (from 'ad3' logic) ---
    # Check absolute win/loss before running slow Voronoi
    if my_routes == 0: return -999999
    if opp_routes == 0 and my_routes > 0: return 999999

    # --- 4. Slow Heuristics (Run only if not obvious) ---
    # **** THIS IS THE NEW PART ****
    # Call the new Voronoi function instead of the old numba one
    my_space, opp_space, are_separated = calculate_voronoi_space(board, my_head, opp_head)
    
    # --- 5. (NEW) High-Priority Survival Logic from ad3 ---
    # This logic now uses the *reliable* Voronoi/flood_fill space count
    if my_routes == 1 and my_space < 15:
        return -800000 + my_space * 1000
    if my_routes == 2 and my_space < 10: # (Slightly stricter than ad3's 15)
        return -100000 + my_space * 100

    # --- 6. (NEW) Combined Strategic Scoring (using ad3's separated logic) ---
    openness = calculate_openness_bonus(my_space, w * h)
    
    if are_separated:
        # We are in separate parts of the board. Fight for space.
        space_score = my_space - opp_space
        final_score = (weights["space"] * space_score * 5.0 + 
                       weights["openness"] * openness * 2.0)
    else:
        # We are in a shared area. Fight for position and traps.
        space_score = my_space - opp_space
        center_bonus = calculate_center_bonus(my_head, opp_head, my_id, turn_count, w, h)
        pressure_score = calculate_pressure_score(my_head, opp_head, turn_count)
        
        # adv9's trapping bonus (now more reliable)
        trapping_bonus = calculate_trapping_bonus(my_routes, opp_routes, my_space, opp_space)
        
        # adv9's corridor penalties
        my_corridor_pen = _numba_corridor_penalty(board, my_head[0], my_head[1], w, h)
        opp_corridor_pen = _numba_corridor_penalty(board, opp_head[0], opp_head[1], w, h) * -1.0
        
        # ad3's boost preservation score
        boost_score = 5.0 * my_boosts # (Using ad3's weight)

        final_score = (
            weights["space"] * space_score +
            weights["center"] * center_bonus +
            weights["pressure"] * pressure_score +
            weights["trapping"] * trapping_bonus +
            weights["openness"] * openness +
            weights["my_corridor"] * my_corridor_pen +
            weights["opp_corridor"] * opp_corridor_pen +
            weights["length"] * (my_len - opp_len) +
            boost_score 
        )
    
    return final_score
# --- (FIX) Re-add get_possible_moves (from v8) ---
def get_possible_moves(agent_dir, boosts_remaining):
    valid_moves = []
    for direction in Direction:
        if direction != OPPOSITE_DIR.get(agent_dir):
            valid_moves.append((direction, False))
            if boosts_remaining > 0:
                valid_moves.append((direction, True))
    return valid_moves

# --- High-Speed Simulation & Search Logic (from v5) ---
def simulate_move_inplace(state, player_id, direction, use_boost):
    if player_id == 1:
        trail = state["agent1_trail"]; boosts_key = "agent1_boosts"; dir_key = "my_direction"
    else:
        trail = state["agent2_trail"]; boosts_key = "agent2_boosts"; dir_key = "opp_direction"

    current_agent_dir = state[dir_key]
    if direction == OPPOSITE_DIR.get(current_agent_dir):
        direction = current_agent_dir
    
    original_boosts = state[boosts_key]; original_direction = state[dir_key]
    undo_actions = []; num_steps = 1
    if use_boost and state[boosts_key] > 0:
        num_steps = 2; state[boosts_key] -= 1
        undo_actions.append(("boost", original_boosts, boosts_key))
    agent_alive = True
    for _ in range(num_steps):
        if not agent_alive: break
        current_head = trail[-1]; dx, dy = direction.value
        next_head = _torus_check((current_head[0] + dx, current_head[1] + dy))
        if state["board"][next_head[1], next_head[0]] == AGENT_TRAIL:
            agent_alive = False; break
        state["board"][next_head[1], next_head[0]] = AGENT_TRAIL
        trail.append(next_head)
        undo_actions.append(("move", next_head, player_id))
    if agent_alive:
        state[dir_key] = direction
    undo_actions.append(("direction", original_direction, dir_key))
    return agent_alive, undo_actions

def undo_moves(state, undo_actions):
    for action in reversed(undo_actions):
        if action[0] == "move":
            pos = action[1]; player_id = action[2]
            state["board"][pos[1], pos[0]] = EMPTY
            if player_id == 1: state["agent1_trail"].pop()
            else: state["agent2_trail"].pop()
        elif action[0] == "boost":
            state[action[2]] = action[1]
        elif action[0] == "direction":
            state[action[2]] = action[1]

def minimax_search(state, depth, is_max_turn, my_id, alpha, beta, weights):
    global TT_CACHE, START_TIME, MAX_TIME_MS
    if time.time() * 1000 > START_TIME + MAX_TIME_MS:
        return 0 
    state_hash = (
        tuple(state["agent1_trail"]), tuple(state["agent2_trail"]),
        state["agent1_boosts"], state["agent2_boosts"],
        state["my_direction"], state["opp_direction"]
    )
    cache_key = (state_hash, depth)
    if cache_key in TT_CACHE:
        return TT_CACHE[cache_key]
    if depth == 0:
        return evaluate_state(state, weights)

    max_id = 1; min_id = 2
    max_dir = state["my_direction"]; min_dir = state["opp_direction"]
    max_boosts = state["agent1_boosts"]; min_boosts = state["agent2_boosts"]
    value = -np.inf
    max_moves = get_possible_moves(max_dir, max_boosts)
    min_moves = get_possible_moves(min_dir, min_boosts)
    max_moves.sort(key=lambda m: m[1]) # Sort non-boosts first

    for my_dir, my_boost in max_moves:
        worst_case_score = np.inf
        my_survived, my_undo = simulate_move_inplace(state, max_id, my_dir, my_boost)
        for opp_dir, opp_boost in min_moves:
            opp_survived, opp_undo = simulate_move_inplace(state, min_id, opp_dir, opp_boost)
            state["turn_count"] += 1
            if not my_survived and not opp_survived: score = 0
            elif not my_survived: score = -999999
            elif not opp_survived: score = 999999
            else:
                score = minimax_search(state, depth - 1, True, my_id, alpha, beta, weights)
            state["turn_count"] -= 1
            undo_moves(state, opp_undo)
            worst_case_score = min(worst_case_score, score)
            if worst_case_score <= alpha:
                break 
        undo_moves(state, my_undo)
        value = max(value, worst_case_score)
        alpha = max(alpha, value)
        if alpha >= beta:
            break
    TT_CACHE[cache_key] = value
    return value

# --- Profile Selection (from v4) ---
def select_profile(state: dict):
    global CURRENT_PROFILE
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    w, h = GRID_WIDTH, GRID_HEIGHT
    
    my_routes = _numba_count_routes(state["board"], my_head[0], my_head[1], w, h)
    opp_routes = _numba_count_routes(state["board"], opp_head[0], opp_head[1], w, h)
    distance = calculate_opponent_distance(my_head, opp_head)
    
    if my_routes <= 2: # Proactive panic
        CURRENT_PROFILE = "DESPERATE_SURVIVAL"
    elif opp_routes <= 1:
        CURRENT_PROFILE = "AGGRESSIVE_KILL"
    elif distance < 5:
        CURRENT_PROFILE = "COUNTER_AGGRESSIVE"
    elif distance > 15:
        CURRENT_PROFILE = "PUNISH_DEFENSIVE"
    else:
        CURRENT_PROFILE = "STANDARD"
        
    return WEIGHT_PROFILES[CURRENT_PROFILE]

# --- Root-Level Search (get_best_move) (from v7) ---
def get_best_move(state, my_id):
    global CURRENT_OPPONENT_STRATEGY, START_TIME, TT_CACHE
    START_TIME = time.time() * 1000
    TT_CACHE.clear()

    turn_count = state.get("turn_count", 0)
    
    # --- (NEW) Select profile *before* search ---
    weights = select_profile(state)
    print(f"Turn {turn_count}: Selected Profile: {CURRENT_PROFILE}")
            
    my_dir = state["my_direction"]; my_boosts = state["agent1_boosts"]
    opp_dir = state["opp_direction"]; opp_boosts = state["agent2_boosts"]

    my_moves = get_possible_moves(my_dir, my_boosts)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    if not my_moves: return "UP" # Forfeit

    move_scores = {move: -np.inf for move in my_moves}

    best_move = (my_dir, False) 
    if (my_dir, False) not in my_moves:
        best_move = my_moves[0] 
    
    max_score = -np.inf
    ASPIRATION_WINDOW_SIZE = 50.0 

    for depth in range(2, MAX_SEARCH_DEPTH + 2, 2):
        if time.time() * 1000 > START_TIME + MAX_TIME_MS:
            print(f"Time limit. Using best from depth {depth-2}.")
            break

        print(f"--- Starting search for depth {depth} ---")
        current_best_move_for_depth = best_move
        current_max_score_for_depth = -np.inf

        if depth == 2:
            alpha, beta = -np.inf, np.inf
        else:
            alpha = max_score - ASPIRATION_WINDOW_SIZE
            beta = max_score + ASPIRATION_WINDOW_SIZE
        
        sorted_moves = sorted(my_moves, key=lambda m: move_scores[m], reverse=True)

        for move_dir, move_boost in sorted_moves:
            worst_case_score = np.inf
            my_survived, my_undo = simulate_move_inplace(state, 1, move_dir, move_boost)
            for opp_move_dir, opp_move_boost in opp_moves:
                opp_survived, opp_undo = simulate_move_inplace(state, 2, opp_move_dir, opp_move_boost)
                state["turn_count"] += 1
                if not my_survived and not opp_survived: score = 0
                elif not my_survived: score = -999999
                elif not opp_survived: score = 999999
                else:
                    score = minimax_search(state, depth - 1, True, my_id, alpha, beta, weights)
                state["turn_count"] -= 1
                undo_moves(state, opp_undo)
                worst_case_score = min(worst_case_score, score)
                if worst_case_score <= alpha: break
            undo_moves(state, my_undo)
            move_scores[(move_dir, move_boost)] = worst_case_score
            if worst_case_score > current_max_score_for_depth:
                current_max_score_for_depth = worst_case_score
                current_best_move_for_depth = (move_dir, move_boost)
            alpha = max(alpha, current_max_score_for_depth) 
            if time.time() * 1000 > START_TIME + MAX_TIME_MS: break
        
        if (depth > 2) and \
           (current_max_score_for_depth <= (max_score - ASPIRATION_WINDOW_SIZE) or \
            current_max_score_for_depth >= (max_score + ASPIRATION_WINDOW_SIZE)) and \
            (time.time() * 1000 < START_TIME + MAX_TIME_MS): 
            print(f"Aspiration window failed (score {current_max_score_for_depth}). Re-searching with full window...")
            current_max_score_for_depth = -np.inf
            alpha = -np.inf; beta = np.inf
            for move_dir, move_boost in sorted_moves:
                worst_case_score = np.inf
                my_survived, my_undo = simulate_move_inplace(state, 1, move_dir, move_boost)
                for opp_move_dir, opp_move_boost in opp_moves:
                    opp_survived, opp_undo = simulate_move_inplace(state, 2, opp_move_dir, opp_move_boost)
                    state["turn_count"] += 1
                    if not my_survived and not opp_survived: score = 0
                    elif not my_survived: score = -999999
                    elif not opp_survived: score = 999999
                    else:
                        score = minimax_search(state, depth - 1, True, my_id, alpha, beta, weights)
                    state["turn_count"] -= 1
                    undo_moves(state, opp_undo)
                    worst_case_score = min(worst_case_score, score)
                    if worst_case_score <= alpha: break
                undo_moves(state, my_undo)
                move_scores[(move_dir, move_boost)] = worst_case_score
                if worst_case_score > current_max_score_for_depth:
                    current_max_score_for_depth = worst_case_score
                    current_best_move_for_depth = (move_dir, move_boost)
                alpha = max(alpha, current_max_score_for_depth)
                if time.time() * 1000 > START_TIME + MAX_TIME_MS: break
        
        if time.time() * 1000 < START_TIME + MAX_TIME_MS:
            best_move = current_best_move_for_depth
            max_score = current_max_score_for_depth
            print(f"Completed depth {depth}. Best move: {best_move[0].name}, Score: {max_score:.1f}")
        else:
            print(f"Timed out during depth {depth}. Reverting to move from depth {depth-2}.")
            break

    final_dir, final_boost = best_move
    move_str = final_dir.name
    if final_boost:
        move_str += ":BOOST"
        
    print(f"Turn {turn_count}: Chosen move {move_str} with estimated score {max_score}")
    return move_str

# --- Flask Endpoints and Helper Functions ---
def decide_action(current_state, player_number):
    state_to_process = current_state 
    state_to_process["player_number"] = player_number
    
    if "board" in state_to_process and not isinstance(state_to_process["board"], np.ndarray):
        state_to_process["board"] = np.array(state_to_process["board"], dtype=np.int8)
    if "agent1_trail" in state_to_process and not isinstance(state_to_process["agent1_trail"], deque):
        state_to_process["agent1_trail"] = deque(tuple(p) for p in state_to_process["agent1_trail"])
    if "agent2_trail" in state_to_process and not isinstance(state_to_process["agent2_trail"], deque):
        state_to_process["agent2_trail"] = deque(tuple(p) for p in state_to_process["agent2_trail"])
    
    state_to_process["my_direction"] = get_current_direction(state_to_process["agent1_trail"], 1)
    state_to_process["opp_direction"] = get_current_direction(state_to_process["agent2_trail"], 2)
    
    if player_number == 2:
        state_to_process["agent1_trail"], state_to_process["agent2_trail"] = \
            state_to_process["agent2_trail"], state_to_process["agent1_trail"]
        state_to_process["agent1_boosts"], state_to_process["agent2_boosts"] = \
            state_to_process["agent2_boosts"], state_to_process["agent1_boosts"]
        state_to_process["my_direction"], state_to_process["opp_direction"] = \
            state_to_process["opp_direction"], state_to_process["my_direction"]
    
    move = get_best_move(state_to_process, player_number)
    return move

def get_current_direction(trail, player_id):
    if len(trail) < 2:
        return Direction.RIGHT if player_id == 1 else Direction.LEFT
    head = trail[-1]; prev = trail[-2]
    dx = head[0] - prev[0]; dy = head[1] - prev[1]
    if abs(dx) > 1: dx = -1 if dx > 0 else 1
    if abs(dy) > 1: dy = -1 if dy > 0 else 1
    if dx == 1: return Direction.RIGHT
    if dx == -1: return Direction.LEFT
    if dy == 1: return Direction.DOWN
    if dy == -1: return Direction.UP
    return Direction.RIGHT if player_id == 1 else Direction.LEFT

def _update_local_game_from_post(data: dict):
    with game_lock:
        GLOBAL_GAME_STATE.update(data)
        if "board" in data:
            GLOBAL_GAME_STATE["board"] = np.array(data["board"], dtype=np.int8)
        if "agent1_trail" in data:
            GLOBAL_GAME_STATE["agent1_trail"] = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME_STATE["agent2_trail"] = deque(tuple(p) for p in data["agent2_trail"])
        GLOBAL_GAME_STATE["my_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent1_trail"], 1)
        GLOBAL_GAME_STATE["opp_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent2_trail"], 2)

app = Flask(__name__)

GLOBAL_GAME_STATE = {
    "board": None, "agent1_trail": deque(), "agent2_trail": deque(),
    "agent1_boosts": 3, "agent2_boosts": 3, "turn_count": 0,
    "player_number": 1, "my_direction": Direction.RIGHT, "opp_direction": Direction.LEFT
}

PARTICIPANT = "GeminiAI_Agent"
AGENT_NAME = "CaseClosed_FinalBoss_v9_NumbaFix"
game_lock = Lock()



@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data: return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    with game_lock:
        current_state = copy.deepcopy(GLOBAL_GAME_STATE)
    move = decide_action(current_state, player_number)
    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    with game_lock:
        TT_CACHE.clear()
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    # --- (FIX) Numba JIT Warmup ---
    # Moved inside __name__ == "__main__" to guarantee constants are defined.
    print("Warming up Numba JIT... (this may take a moment)")
    try:
        dummy_board = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=np.int8)
        dummy_board[5, 5] = AGENT; dummy_board[10, 10] = AGENT
        _numba_wrap(1, 1, GRID_WIDTH, GRID_HEIGHT)
        _numba_count_routes(dummy_board, 1, 1, GRID_WIDTH, GRID_HEIGHT)
        # _numba_reachable_area(dummy_board, 1, 1, GRID_WIDTH, GRID_HEIGHT, 100)
        _numba_corridor_penalty(dummy_board, 1, 1, GRID_WIDTH, GRID_HEIGHT)
        print("Numba JIT compiled successfully.")
    except Exception as e:
        print(f"Numba warmup failed: {e}")
        
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)