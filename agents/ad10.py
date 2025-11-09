import os
import time
import copy
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
from enum import Enum
import sys

# Set a higher recursion limit for deep minimax search
sys.setrecursionlimit(2500)

# --- Core Game Constants (Unchanged) ---
class Direction(Enum):
    UP = (0, -1); DOWN = (0, 1); RIGHT = (1, 0); LEFT = (-1, 0)
DIRECTION_MAP = {d.name: d for d in Direction}
OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}
GRID_HEIGHT = 18; GRID_WIDTH = 20; EMPTY = 0; AGENT_TRAIL = 1

# --- Search Configuration ---
MAX_SEARCH_DEPTH = 30 
MAX_TIME_MS = 3800  # 3.8 seconds max per move (4s limit)

# --- Transposition Table (Cache) ---
TT_CACHE = {}
START_TIME = 0.0

# --- Flask API server setup (Unchanged) ---
app = Flask(__name__)

# --- Global State Management (Unchanged) ---
GLOBAL_GAME_STATE = {
    "board": None, "agent1_trail": deque(), "agent2_trail": deque(),
    "agent1_boosts": 3, "agent2_boosts": 3, "turn_count": 0,
    "player_number": 1, "my_direction": Direction.RIGHT, "opp_direction": Direction.LEFT
}
WEIGHT_PROFILES = {
    "standard": {
        "space": 1.0, "center": 0.5, "pressure": 0.2, "trapping": 1.0, 
        "openness": 0.5, "boost_preservation": 5.0
    },
    "counter_aggressive": {
        "space": 1.0, "center": 0.1, "pressure": 0.1, "trapping": 3.0, 
        "openness": 1.5, "boost_preservation": 5.0
    },
    "punish_defensive": {
        "space": 1.5, "center": 0.8, "pressure": 1.0, "trapping": 1.0, 
        "openness": 0.5, "boost_preservation": 5.0
    }
}
CURRENT_OPPONENT_STRATEGY = "standard"
game_lock = Lock()

# --- Agent Identity ---
PARTICIPANT = "GeminiAI_Agent"
AGENT_NAME = "CaseClosed_FinalBoss_v8_Fixed"

# --- All Heuristic Functions (Unchanged from v5) ---
def _torus_check(pos):
    x, y = pos
    normalized_x = x % GRID_WIDTH
    normalized_y = y % GRID_HEIGHT
    return (normalized_x, normalized_y)
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
def calculate_center_distance(pos):
    center_x, center_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
    return abs(pos[0] - center_x) + abs(pos[1] - center_y)
def calculate_center_bonus(my_head, opp_head, my_id, turn_count):
    my_dist = calculate_center_distance(my_head); opp_dist = calculate_center_distance(opp_head)
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
def count_escape_routes(board, pos):
    safe_directions = 0
    for direction in Direction:
        dx, dy = direction.value
        next_pos = _torus_check((pos[0] + dx, pos[1] + dy))
        if board[next_pos[1], next_pos[0]] == EMPTY:
            safe_directions += 1
    return safe_directions
def calculate_escape_quality(board, my_head, opp_head):
    my_routes = count_escape_routes(board, my_head); opp_routes = count_escape_routes(board, opp_head)
    route_advantage = (my_routes - opp_routes) * 4.0
    if opp_routes == 0 and my_routes > 0: route_advantage += 1000.0
    elif opp_routes == 1 and my_routes > 2: route_advantage += 50.0
    elif opp_routes <= 2 and my_routes > 3: route_advantage += 25.0
    if my_routes >= 4: route_advantage += 10.0
    elif my_routes == 3: route_advantage += 5.0
    if my_routes == 0: route_advantage -= 1000.0
    elif my_routes == 1: route_advantage -= 75.0
    return route_advantage
def calculate_trapping_bonus(board, my_space, opp_space, my_head, opp_head):
    return calculate_escape_quality(board, my_head, opp_head)
def calculate_openness_bonus(board, my_head, my_space):
    board_size = board.shape[0] * board.shape[1]; openness_ratio = my_space / board_size
    if openness_ratio > 0.35: return 15.0
    elif openness_ratio > 0.25: return 10.0
    elif openness_ratio > 0.15: return 0
    elif openness_ratio > 0.08: return -15.0
    else: return -40.0
def get_game_phase(turn_count):
    if turn_count < 20: return "early"
    elif turn_count < 60: return "mid"
    else: return "late"
def evaluate_state(state, my_id, weights):
    board = state["board"]; turn_count = state.get("turn_count", 0)
    my_head = state["agent1_trail"][-1]; opp_head = state["agent2_trail"][-1]
    my_boosts = state["agent1_boosts"]
    my_routes = count_escape_routes(board, my_head)
    opp_routes = count_escape_routes(board, opp_head)
    if my_routes == 0: return -999999
    if opp_routes == 0 and my_routes > 0: return 999999
    my_space, opp_space, are_separated = calculate_voronoi_space(board, my_head, opp_head)
    if my_routes == 1 and my_space < 15:
        return -800000 + my_space * 1000
    if are_separated:
        space_score = my_space - opp_space
        openness = calculate_openness_bonus(board, my_head, my_space)
        final_score = (weights["space"] * space_score * 5.0 + weights["openness"] * openness * 2.0)
    else:
        space_score = my_space - opp_space
        center_bonus = calculate_center_bonus(my_head, opp_head, my_id, turn_count)
        pressure_score = calculate_pressure_score(my_head, opp_head, turn_count)
        route_advantage = (my_routes - opp_routes) * 4.0
        openness = calculate_openness_bonus(board, my_head, my_space)
        boost_score = weights["boost_preservation"] * my_boosts
        final_score = (
            weights["space"] * space_score + weights["center"] * center_bonus +
            weights["pressure"] * pressure_score + weights["trapping"] * route_advantage +
            weights["openness"] * openness + boost_score
        )
    return final_score

# --- (FIX) THIS FUNCTION WAS MISSING ---
def get_possible_moves(agent_dir, boosts_remaining):
    valid_moves = []
    for direction in Direction:
        if direction != OPPOSITE_DIR.get(agent_dir):
            valid_moves.append((direction, False))
            if boosts_remaining > 0:
                valid_moves.append((direction, True))
    return valid_moves

# --- High-Speed Simulation & Search Logic (Unchanged) ---
def simulate_move_inplace(state, player_id, direction, use_boost):
    if player_id == 1:
        trail = state["agent1_trail"]; boosts_key = "agent1_boosts"; dir_key = "my_direction"
    else:
        trail = state["agent2_trail"]; boosts_key = "agent2_boosts"; dir_key = "opp_direction"
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
        return evaluate_state(state, my_id, weights)

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

# --- (CHANGED) Root-Level Search (get_best_move) ---
def get_best_move(state, my_id):
    global CURRENT_OPPONENT_STRATEGY, START_TIME, TT_CACHE
    START_TIME = time.time() * 1000
    TT_CACHE.clear()

    turn_count = state.get("turn_count", 0)
    
    # (Opponent modeling - unchanged)
    if turn_count > 10 and turn_count % 10 == 0:
        my_head = state["agent1_trail"][-1]; opp_head = state["agent2_trail"][-1]
        distance = calculate_opponent_distance(my_head, opp_head)
        if distance < 5: CURRENT_OPPONENT_STRATEGY = "counter_aggressive"
        elif distance > 15: CURRENT_OPPONENT_STRATEGY = "punish_defensive"
        elif calculate_center_distance(opp_head) < 4: CURRENT_OPPONENT_STRATEGY = "standard"
        else: CURRENT_OPPONENT_STRATEGY = "standard"
            
    my_dir = state["my_direction"]; my_boosts = state["agent1_boosts"]
    opp_dir = state["opp_direction"]; opp_boosts = state["agent2_boosts"]

    my_moves = get_possible_moves(my_dir, my_boosts)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    if not my_moves: # No moves possible
        return "UP" # Forfeit

    move_scores = {move: -np.inf for move in my_moves}

    best_move = (my_dir, False) 
    if (my_dir, False) not in my_moves:
        best_move = my_moves[0] 
    
    max_score = -np.inf
    selected_weights = WEIGHT_PROFILES[CURRENT_OPPONENT_STRATEGY].copy()
    phase = get_game_phase(state.get("turn_count", 0))
    if phase == "mid": selected_weights["trapping"] *= 1.5  
    elif phase == "late": selected_weights["trapping"] *= 3.0

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

        # --- (FIX) This loop is now IN-PLACE ---
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
                    score = minimax_search(state, depth - 1, True, my_id, alpha, beta, selected_weights)
                
                state["turn_count"] -= 1
                undo_moves(state, opp_undo)
                worst_case_score = min(worst_case_score, score)
                if worst_case_score <= alpha:
                   break
            
            undo_moves(state, my_undo)
            
            move_scores[(move_dir, move_boost)] = worst_case_score
            
            if worst_case_score > current_max_score_for_depth:
                current_max_score_for_depth = worst_case_score
                current_best_move_for_depth = (move_dir, move_boost)
            
            alpha = max(alpha, current_max_score_for_depth) 

            if time.time() * 1000 > START_TIME + MAX_TIME_MS:
                break
        
        # --- (FIX) Aspiration Window Re-search ---
        if (depth > 2) and \
           (current_max_score_for_depth <= (max_score - ASPIRATION_WINDOW_SIZE) or \
            current_max_score_for_depth >= (max_score + ASPIRATION_WINDOW_SIZE)) and \
            (time.time() * 1000 < START_TIME + MAX_TIME_MS): 
            
            print(f"Aspiration window failed (score {current_max_score_for_depth}). Re-searching with full window...")
            current_max_score_for_depth = -np.inf
            alpha = -np.inf
            beta = np.inf
            
            for move_dir, move_boost in sorted_moves:
                worst_case_score = np.inf
                my_survived, my_undo = simulate_move_inplace(state, 1, move_dir, move_boost)
                for opp_move_dir, opp_move_boost in opp_moves:
                    opp_survived, opp_undo = simulate_move_inplace(state, 2, opp_move_dir, opp_move_boost)
                    state["turn_count"] += 1 # Bugfix: was state_copy
                    if not my_survived and not opp_survived: score = 0
                    elif not my_survived: score = -999999
                    elif not opp_survived: score = 999999
                    else:
                        score = minimax_search(state, depth - 1, True, my_id, alpha, beta, selected_weights)
                    state["turn_count"] -= 1 # Bugfix: was state_copy
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
        
        # --- Update Best Move ---
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

# --- (CHANGED) Flask Endpoints and Helper Functions ---
def decide_action(current_state, player_number):
    """
    This function now receives a DEEP COPY of the state
    and can safely pass it to the in-place search.
    """
    state_to_process = current_state 
    state_to_process["player_number"] = player_number
    
    # Ensure types are correct (e.g., if just received from judge)
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
    
    # This state is now safe to be modified by the in-place search
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
        # Always convert to numpy/deque *immediately* on receive
        if "board" in data:
            GLOBAL_GAME_STATE["board"] = np.array(data["board"], dtype=np.int8)
        if "agent1_trail" in data:
            GLOBAL_GAME_STATE["agent1_trail"] = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME_STATE["agent2_trail"] = deque(tuple(p) for p in data["agent2_trail"])
        
        GLOBAL_GAME_STATE["my_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent1_trail"], 1)
        GLOBAL_GAME_STATE["opp_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent2_trail"], 2)

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
        # (FIX) We MUST deepcopy ONCE to get a clean state for in-place search
        current_state = copy.deepcopy(GLOBAL_GAME_STATE)
        
    # This 'current_state' is now safe to be modified in-place
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
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)