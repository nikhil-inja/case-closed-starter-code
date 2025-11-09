import os
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
from numba import njit
import sys

# Set a higher recursion limit for deep search
sys.setrecursionlimit(2000)

# ---- Game constants ----
EMPTY = 0
AGENT = 1
BOARD_HEIGHT = 18
BOARD_WIDTH = 20
MAX_TURNS = 200 # CRITICAL: This is 200 from case_closed_game.py
MOVE_TIME_LIMIT = 0.8 # Seconds. Be safe, judge timeout is ~1s

# Directions as (dx, dy) and names
DIRS = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
DIR_NAMES = list(DIRS.keys())
DIR_VECTORS = list(DIRS.values())
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

# --- Numba-Optimized Helper Functions ---

@njit(cache=True)
def _numba_wrap(x, y, w, h):
    """Numba-compatible torus wrap."""
    return (x % w, y % h)

@njit(cache=True)
def _numba_count_routes(board, x, y, w, h):
    """Numba-compiled function to count empty escape routes."""
    empties = 0
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0)) # UP, DOWN, LEFT, RIGHT
    
    for i in range(4):
        dx, dy = local_dir_vectors[i]
        nx, ny = _numba_wrap(x + dx, y + dy, w, h)
        if board[ny, nx] == EMPTY:
            empties += 1
    return empties

@njit(cache=True)
def _numba_corridor_penalty(board, x, y, w, h):
    """
    (FROM BSD) Penalize tight corridors / dead-ends.
    This is the key function we were missing.
    """
    empties = _numba_count_routes(board, x, y, w, h)
    
    # Penalties are large to dominate the score
    if empties == 0: return -100000.0 # Trapped
    elif empties == 1: return -5000.0   # Dead end
    elif empties == 2: return -50.0     # Corridor
    elif empties == 3: return -1.0      # Mildly constrained
    else: return 0.0                    # Open space

@njit(cache=True)
def _numba_reachable_area(board, start_x, start_y, w, h, cutoff):
    """
    Numba-compiled BFS flood-fill to find reachable area.
    """
    q = np.empty((w * h, 2), dtype=np.int16)
    q_head, q_tail = 0, 0
    q[q_tail] = np.array([start_x, start_y], dtype=np.int16)
    q_tail += 1
    
    seen = np.zeros((h, w), dtype=np.bool_)
    seen[start_y, start_x] = True
    count = 0
    
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0))
    
    while q_head < q_tail and count < cutoff:
        x, y = q[q_head]
        q_head += 1
        
        for i in range(4):
            dx, dy = local_dir_vectors[i]
            nx, ny = _numba_wrap(x + dx, y + dy, w, h)
            
            if not seen[ny, nx] and board[ny, nx] == EMPTY:
                seen[ny, nx] = True
                q[q_tail] = np.array([nx, ny], dtype=np.int16)
                q_tail += 1
                count += 1
                
    return count

# --- Search Optimization ---
TT_CACHE = {} # Transposition table (cache) for the search

# --- Game State Class (from bsd) ---
class GameState:
    """A fast, lightweight, and copyable game state for search."""
    def __init__(self, board, my_trail, opp_trail, my_len, opp_len, my_boosts, opp_boosts, turn, player_number):
        self.board = board
        self.my_trail = my_trail
        self.opp_trail = opp_trail
        self.my_len = my_len
        self.opp_len = opp_len
        self.my_boosts = my_boosts
        self.opp_boosts = opp_boosts
        self.turn = turn
        self.player_number = player_number
        self.my_head = my_trail[-1]
        self.opp_head = opp_trail[-1]
        self.w = BOARD_WIDTH
        self.h = BOARD_HEIGHT
    
    def get_hash(self):
        """Creates a unique, hashable key for the current game state."""
        return (self.board.tobytes(), self.my_head, self.opp_head, self.my_len, self.opp_len, self.my_boosts, self.opp_boosts)

    def is_safe(self, pos):
        return self.board[pos[1], pos[0]] == EMPTY

    def step(self, pos, w, h, direction):
        dx, dy = DIRS[direction]
        return _numba_wrap(pos[0] + dx, pos[1] + dy, w, h)

    def get_valid_moves(self, trail, current_dir):
        moves = []
        for d_name in DIR_NAMES:
            if d_name != OPPOSITE.get(current_dir):
                moves.append(d_name)
        return moves

    def simulate_step(self, my_move, opp_move):
        """
        Simulates one full turn given moves for both players (Player 1 moves first).
        """
        if self.player_number == 1:
            p1_move_str, p2_move_str = my_move, opp_move
            p1_trail, p2_trail = self.my_trail, self.opp_trail
            p1_boosts, p2_boosts = self.my_boosts, self.opp_boosts
            p1_len, p2_len = self.my_len, self.opp_len
        else:
            p1_move_str, p2_move_str = opp_move, my_move
            p1_trail, p2_trail = self.opp_trail, self.my_trail
            p1_boosts, p2_boosts = self.opp_boosts, self.my_boosts
            p1_len, p2_len = self.opp_len, self.my_len

        p1_dir, p1_use_boost = p1_move_str.split(":")
        p2_dir, p2_use_boost = p2_move_str.split(":")
        p1_use_boost = (p1_use_boost == "B") and (p1_boosts > 0)
        p2_use_boost = (p2_use_boost == "B") and (p2_boosts > 0)
        
        p1_steps = 2 if p1_use_boost else 1
        p2_steps = 2 if p2_use_boost else 1

        new_board = self.board.copy()
        new_p1_trail = deque(p1_trail)
        new_p2_trail = deque(p2_trail)
        new_p1_len = p1_len
        new_p2_len = p2_len
        new_p1_boosts = p1_boosts - 1 if p1_use_boost else p1_boosts
        new_p2_boosts = p2_boosts - 1 if p2_use_boost else p2_boosts

        p1_alive = True
        p2_alive = True
        p1_head = p1_trail[-1]
        p2_head = p2_trail[-1]

        # --- Simulate Player 1's Move ---
        for i in range(p1_steps):
            if not p1_alive: break
            p1_head = self.step(p1_head, self.w, self.h, p1_dir)
            
            if new_board[p1_head[1], p1_head[0]] == AGENT:
                p1_alive = False
                if p1_head == p2_head:
                    p2_alive = False
            if p1_alive:
                new_board[p1_head[1], p1_head[0]] = AGENT
                new_p1_trail.append(p1_head)
                new_p1_len += 1
            else:
                break

        # --- Simulate Player 2's Move ---
        for i in range(p2_steps):
            if not p2_alive: break
            p2_head = self.step(p2_head, self.w, self.h, p2_dir)
            
            if new_board[p2_head[1], p2_head[0]] == AGENT:
                p2_alive = False
                if p1_alive and p2_head == p1_head:
                    p1_alive = False
            if p2_alive:
                new_board[p2_head[1], p2_head[0]] = AGENT
                new_p2_trail.append(p2_head)
                new_p2_len += 1
            else:
                break

        # --- Return new state or terminal value ---
        my_alive = p1_alive if self.player_number == 1 else p2_alive
        opp_alive = p2_alive if self.player_number == 1 else p1_alive
        
        if not my_alive and not opp_alive: return 0
        if not my_alive: return -1e9
        if not opp_alive: return 1e9

        if self.player_number == 1:
            return GameState(new_board, new_p1_trail, new_p2_trail, new_p1_len, new_p2_len, new_p1_boosts, new_p2_boosts, self.turn + 1, self.player_number)
        else:
            return GameState(new_board, new_p2_trail, new_p1_trail, new_p2_len, new_p1_len, new_p2_boosts, new_p1_boosts, self.turn + 1, self.player_number)

# --- Heuristic Functions (Ported from 'ad') ---

def calculate_center_distance(pos, w, h):
    center_x, center_y = w // 2, h // 2
    return abs(pos[0] - center_x) + abs(pos[1] - center_y)

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

def get_game_phase(turn_count):
    if turn_count < 20: return "early"
    elif turn_count < 60: return "mid"
    else: return "late"

# --- NEW: Fused Heuristic Evaluation Function ---
def evaluate_state(state: GameState):
    """
    Hybrid evaluation function from 'ad', now powered by numba
    AND with the 'bsd' corridor penalty.
    """
    board = state.board
    turn_count = state.turn
    my_head = state.my_head
    opp_head = state.opp_head
    w, h = state.w, state.h
    my_id = state.player_number

    # --- 1. Numba-Powered Heuristics ---
    my_space = _numba_reachable_area(board, my_head[0], my_head[1], w, h, 400)
    opp_space = _numba_reachable_area(board, opp_head[0], opp_head[1], w, h, 400)
    my_routes = _numba_count_routes(board, my_head[0], my_head[1], w, h)
    opp_routes = _numba_count_routes(board, opp_head[0], opp_head[1], w, h)

    # --- 2. "SURVIVAL LAYERS" (The Winning Logic from 'ad') ---
    if my_space < 20:
        if my_routes == 0: return -999999
        elif my_routes == 1: return -800000 + my_space * 1000
        elif my_routes == 2: return -50000 + my_space * 500
        else: return my_space * 100 + my_routes * 5000
    
    # These hard-coded returns are the *most important* part of the 'ad' heuristic
    if my_routes <= 1:
        return -500000 + my_routes * 10000
    
    if my_routes == 2 and my_space < 15:
        return -100000 + my_space * 100
    
    # --- 3. End-Game Length Battle (from bsd) ---
    if state.turn > MAX_TURNS - 20:
        length_score = (state.my_len - state.opp_len)
        return length_score * 10000.0

    # --- 4. Core Strategic Metrics (from 'ad') ---
    space_score = my_space - opp_space
    center_bonus = calculate_center_bonus(my_head, opp_head, my_id, turn_count, w, h)
    pressure_score = calculate_pressure_score(my_head, opp_head, turn_count)
    trapping_bonus = calculate_trapping_bonus(my_routes, opp_routes, my_space, opp_space)
    openness = calculate_openness_bonus(my_space, w * h)
    
    # --- 5. (NEW) Corridor Penalties (from 'bsd') ---
    # This is the fix. We add the bsd corridor penalty to the ad heuristic.
    my_corridor_pen = _numba_corridor_penalty(board, my_head[0], my_head[1], w, h)
    # We also add the *aggressive* part: penalize the opponent for being in a corridor
    opp_corridor_pen = _numba_corridor_penalty(board, opp_head[0], opp_head[1], w, h) * -1.0
    
    # --- 6. Phase-Aware Weights (from 'ad') ---
    phase = get_game_phase(turn_count)
    if phase == "early":
        weights = {"space": 1.0, "center": 0.8, "pressure": 0.2, "trapping": 0.8, "openness": 0.5}
    elif phase == "mid":
        weights = {"space": 1.0, "center": 0.2, "pressure": 0.6, "trapping": 2.0, "openness": 0.8}
    else:
        weights = {"space": 1.0, "center": 0.1, "pressure": 0.4, "trapping": 3.5, "openness": 1.0}

    # --- 7. Final Combined Score ---
    final_score = (
        weights["space"] * space_score +
        weights["center"] * center_bonus +
        weights["pressure"] * pressure_score +
        weights["trapping"] * trapping_bonus +
        weights["openness"] * openness +
        my_corridor_pen +  # <-- ADDED
        opp_corridor_pen   # <-- ADDED
    )
    
    if opp_space == 0 and my_space > 0:
        return 999999
    
    return final_score


# --- Search Algorithm (from bsd) ---

class SearchTimeout(Exception):
    pass

def maximin_search(state, depth, alpha, beta, start_time, time_limit, my_current_dir, opp_current_dir):
    """
    Recursive Maximin search with Transposition Table caching.
    """
    
    if (time.time() - start_time) > time_limit:
        raise SearchTimeout()
    
    state_key = (state.get_hash(), my_current_dir, opp_current_dir)
    if state_key in TT_CACHE:
        cached_depth, cached_score = TT_CACHE[state_key]
        if cached_depth >= depth:
            return cached_score
            
    if depth == 0:
        return evaluate_state(state)

    my_valid_dirs = state.get_valid_moves(state.my_trail, my_current_dir)
    opp_valid_dirs = state.get_valid_moves(state.opp_trail, opp_current_dir)

    my_moves = []
    for d in my_valid_dirs:
        my_moves.append(f"{d}:NB")
        if state.my_boosts > 0:
            my_moves.append(f"{d}:B")
    
    opp_moves = []
    for d in opp_valid_dirs:
        opp_moves.append(f"{d}:NB")
        if state.opp_boosts > 0:
            opp_moves.append(f"{d}:B")

    if not my_moves: return -1e9 + state.turn
    if not opp_moves: return 1e9 - state.turn
            
    best_score = -np.inf
    
    for my_move in my_moves:
        worst_reply_score = np.inf
        
        for opp_move in opp_moves:
            sim_result = state.simulate_step(my_move, opp_move)
            
            score = 0
            if isinstance(sim_result, (int, float)):
                score = sim_result
            else:
                score = maximin_search(sim_result, depth - 1, alpha, beta, start_time, time_limit, my_move.split(":")[0], opp_move.split(":")[0])
            
            worst_reply_score = min(worst_reply_score, score)
            beta = min(beta, worst_reply_score)
            if beta <= alpha: break
        
        best_score = max(best_score, worst_reply_score)
        alpha = max(alpha, best_score)
        if beta <= alpha: break

    TT_CACHE[state_key] = (depth, best_score)
    return best_score

# --- Main Action Function (from bsd) ---

def get_move_sort_key(state, move, my_head, w, h):
    """
    Generates a 'quick score' for a move to be used for sorting.
    """
    move_dir, use_boost = move.split(":")
    
    next_pos = state.step(my_head, w, h, move_dir)
    if not state.is_safe(next_pos):
        return -1e10
    
    # Use the strong corridor penalty for sorting
    score = _numba_corridor_penalty(state.board, next_pos[0], next_pos[1], w, h)
    
    if use_boost == "B":
        next_pos_2 = state.step(next_pos, w, h, move_dir)
        if not state.is_safe(next_pos_2):
            return -1e10
        
        # Average the penalty
        score = (score + _numba_corridor_penalty(state.board, next_pos_2[0], next_pos_2[1], w, h)) / 2.0
        
    return score

def decide_action(state, player_number):
    
    start_time = time.time()
    global TT_CACHE
    TT_CACHE.clear()
    
    # --- 1. Parse State ---
    if player_number == 1:
        my_trail = deque(tuple(p) for p in state["agent1_trail"])
        opp_trail = deque(tuple(p) for p in state["agent2_trail"])
        my_len = int(state.get("agent1_length", len(my_trail)))
        opp_len = int(state.get("agent2_length", len(opp_trail)))
        my_boosts = int(state["agent1_boosts"])
        opp_boosts = int(state["agent2_boosts"])
    else:
        my_trail = deque(tuple(p) for p in state["agent2_trail"])
        opp_trail = deque(tuple(p) for p in state["agent1_trail"])
        my_len = int(state.get("agent2_length", len(my_trail)))
        opp_len = int(state.get("agent1_length", len(opp_trail)))
        my_boosts = int(state["agent2_boosts"])
        opp_boosts = int(state["agent1_boosts"])
        
    turn = int(state.get("turn_count", 0))

    if not my_trail or not opp_trail:
        return "RIGHT"

    board_list = state["board"]
    board_np = np.array(board_list, dtype=np.int8)
    w, h = BOARD_WIDTH, BOARD_HEIGHT
    my_dir = last_dir_from_trail(my_trail, w, h)
    opp_dir = last_dir_from_trail(opp_trail, w, h)

    # --- 2. Create Root GameState ---
    root_state = GameState(board_np, my_trail, opp_trail, my_len, opp_len, my_boosts, opp_boosts, turn, player_number)
    
    # --- 3. Get My Moves & Immediate Safety Check ---
    my_valid_dirs = root_state.get_valid_moves(my_trail, my_dir)
    safe_moves = []
    
    for d in my_valid_dirs:
        next_pos = root_state.step(root_state.my_head, w, h, d)
        if root_state.is_safe(next_pos):
            safe_moves.append(f"{d}:NB")
            if my_boosts > 0:
                next_pos_2 = root_state.step(next_pos, w, h, d)
                if root_state.is_safe(next_pos_2):
                    safe_moves.append(f"{d}:B")

    if not safe_moves:
        print(f"TURN {turn}: WARNING: All valid moves are unsafe. Searching for best loss...")
        safe_moves = []
        for d in my_valid_dirs:
            safe_moves.append(f"{d}:NB")
            if my_boosts > 0:
                safe_moves.append(f"{d}:B")
        if not safe_moves:
             return "UP"

    # --- 4. Iterative Deepening Search ---
    try:
        sorted_safe_moves = sorted(
            safe_moves, 
            key=lambda m: get_move_sort_key(root_state, m, root_state.my_head, w, h), 
            reverse=True # Higher scores (less penalty) first
        )
    except Exception as e:
        print(f"Error during move sort: {e}. Using unsorted list.")
        sorted_safe_moves = safe_moves

    best_overall_move = sorted_safe_moves[0]
    best_overall_score = -np.inf
    
    for depth in range(3, 20): 
        best_move_for_depth = best_overall_move
        best_score_for_depth = -np.inf
        
        if best_overall_move in sorted_safe_moves:
            sorted_safe_moves.insert(0, sorted_safe_moves.pop(sorted_safe_moves.index(best_overall_move)))

        try:
            for my_move in sorted_safe_moves:
                worst_reply_score = np.inf
                
                opp_valid_dirs = root_state.get_valid_moves(opp_trail, opp_dir)
                opp_moves = []
                for d in opp_valid_dirs:
                    opp_moves.append(f"{d}:NB")
                    if opp_boosts > 0:
                        opp_moves.append(f"{d}:B")
                
                if not opp_moves:
                    worst_reply_score = 1e9 - turn
                else:
                    for opp_move in opp_moves:
                        sim_result = root_state.simulate_step(my_move, opp_move)
                        score = 0
                        if isinstance(sim_result, (int, float)):
                            score = sim_result
                        else:
                            score = maximin_search(sim_result, depth - 1, -np.inf, np.inf, start_time, MOVE_TIME_LIMIT, my_move.split(":")[0], opp_move.split(":")[0])
                        worst_reply_score = min(worst_reply_score, score)

                if worst_reply_score > best_score_for_depth:
                    best_score_for_depth = worst_reply_score
                    best_move_for_depth = my_move

            best_overall_move = best_move_for_depth
            best_overall_score = best_score_for_depth
            
            if best_overall_score >= 1e9: break

        except SearchTimeout:
            break
        except Exception as e:
            print(f"!!! SEARCH ERROR at depth {depth}: {e}")
            break
    
    # --- 5. Format and Return ---
    final_move_dir, final_boost = best_overall_move.split(":")
    print(f"Turn {turn}: Chose {best_overall_move} (Score: {best_overall_score:.0f}) (Time: {(time.time() - start_time):.3f}s)")

    if final_boost == "B":
        return f"{final_move_dir}:BOOST"
    else:
        return final_move_dir

# ----------------------- Flask endpoints -----------------------

app = Flask(__name__)
PARTICIPANT = os.getenv("PARTICIPANT", "Gemini_Optimized")
AGENT_NAME = os.getenv("AGENT_NAME", "CaseClosed_Fusion_v2")
state_lock = Lock()
LAST = {
    "board": None, "agent1_trail": [], "agent2_trail": [],
    "agent1_length": 0, "agent2_length": 0, "agent1_alive": True,
    "agent2_alive": True, "agent1_boosts": 3, "agent2_boosts": 3, "turn_count": 0,
}

def last_dir_from_trail(trail, w, h):
    if len(trail) < 2: return "RIGHT"
    x2, y2 = trail[-1]; x1, y1 = trail[-2]
    dx, dy = x2 - x1, y2 - y1
    if abs(dx) > 1: dx = -np.sign(dx)
    if abs(dy) > 1: dy = -np.sign(dy)
    if dx == 1: return "RIGHT"
    if dx == -1: return "LEFT"
    if dy == 1: return "DOWN"
    if dy == -1: return "UP"
    return "RIGHT"

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data: return jsonify({"error": "no json body"}), 400
    with state_lock:
        for k in LAST.keys():
            if k in data: LAST[k] = data[k]
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    with state_lock:
        state_snapshot = {k: LAST[k] for k in LAST}
    move = decide_action(state_snapshot, player_number)
    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        with state_lock:
            for k in LAST.keys():
                if k in data: LAST[k] = data[k]
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    
    print("Warming up Numba JIT... (this may take a moment)")
    try:
        dummy_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
        dummy_board[5, 5] = AGENT; dummy_board[10, 10] = AGENT
        _numba_wrap(1, 1, BOARD_WIDTH, BOARD_HEIGHT)
        _numba_count_routes(dummy_board, 1, 1, BOARD_WIDTH, BOARD_HEIGHT)
        _numba_reachable_area(dummy_board, 1, 1, BOARD_WIDTH, BOARD_HEIGHT, 100)
        _numba_corridor_penalty(dummy_board, 1, 1, BOARD_WIDTH, BOARD_HEIGHT)
        print("Numba JIT compiled successfully.")
    except Exception as e:
        print(f"Numba warmup failed: {e}")
        
    app.run(host="0.0.0.0", port=port, debug=False)