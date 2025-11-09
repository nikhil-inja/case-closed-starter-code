import os
import time
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
from numba import njit

# ---- Game constants ----
EMPTY = 0
AGENT = 1
BOARD_HEIGHT = 18
BOARD_WIDTH = 20
MAX_TURNS = 500 # CRITICAL: This is 500 in judge_engine.py, not 200
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

# --- Numba-Optimized Helper Functions (Unchanged) ---
# These functions are top-level to be JIT-compiled correctly.

@njit # Enable cache for JIT
def _numba_wrap(x, y, w, h):
    """Numba-compatible torus wrap."""
    return (x % w, y % h)

@njit # Enable cache for JIT
def _numba_corridor_penalty(board, x, y, w, h):
    """
    Penalize tight corridors / dead-ends.
    Count empty neighbors; fewer implies riskier.
    """
    empties = 0
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0)) # UP, DOWN, LEFT, RIGHT
    
    for i in range(4):
        dx, dy = local_dir_vectors[i]
        nx, ny = _numba_wrap(x + dx, y + dy, w, h)
        if board[ny, nx] == EMPTY:
            empties += 1
    
    # Penalties are much higher to survive Minimax scoring
    if empties == 0: return -100000.0 # Trapped
    elif empties == 1: return -5000.0   # Dead end
    elif empties == 2: return -50.0     # Corridor
    elif empties == 3: return -1.0      # Mildly constrained
    else: return 0.0                    # Open space

@njit # Enable cache for JIT
def _numba_voronoi(board, my_head_tuple, opp_head_tuple, w, h):
    """
    Numba-compiled Voronoi partition (comparative flood fill).
    """
    my_head = np.array([my_head_tuple[0], my_head_tuple[1]], dtype=np.int16)
    opp_head = np.array([opp_head_tuple[0], opp_head_tuple[1]], dtype=np.int16)
    
    local_dir_vectors = (
        (0, -1), (0, 1), (-1, 0), (1, 0)
    )

    q = np.empty((w * h * 2, 4), dtype=np.int16) # x, y, player, dist
    q_head, q_tail = 0, 0

    # 0 = empty, 1 = me, 2 = opp
    owner_grid = np.zeros((h, w), dtype=np.int16)
    dist_grid = np.full((h, w), 9999, dtype=np.int16)

    # Add my head
    q[q_tail] = np.array([my_head[0], my_head[1], 1, 0], dtype=np.int16)
    dist_grid[my_head[1], my_head[0]] = 0
    owner_grid[my_head[1], my_head[0]] = 1
    q_tail += 1

    # Add opp head
    q[q_tail] = np.array([opp_head[0], opp_head[1], 2, 0], dtype=np.int16)
    dist_grid[opp_head[1], opp_head[0]] = 0
    if owner_grid[opp_head[1], opp_head[0]] == 0:
        owner_grid[opp_head[1], opp_head[0]] = 2
    q_tail += 1


    while q_head < q_tail:
        x, y, player, dist = q[q_head]
        q_head += 1

        if dist > dist_grid[y, x]:
            continue
        
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
                    # Contested tile, nobody owns it
                    owner_grid[ny, nx] = 0

    my_final_score = np.sum(owner_grid == 1)
    opp_final_score = np.sum(owner_grid == 2)

    return my_final_score, opp_final_score

# --- Search Optimization (NEW) ---

# Transposition table (cache) for the search
# We'll clear this at the start of each move
TT_CACHE = {} 

# --- Game State & Search ---

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
        """(NEW) Creates a unique, hashable key for the current game state."""
        # This is the fastest way to hash the board + key unique info
        return (
            self.board.tobytes(), 
            self.my_head, 
            self.opp_head, 
            self.my_len, 
            self.opp_len, 
            self.my_boosts, 
            self.opp_boosts
        )

    def copy(self):
        """Create a new GameState for a simulation step."""
        return GameState(
            self.board.copy(), # Critical: copy the numpy board
            self.my_trail,     # Will be replaced
            self.opp_trail,    # Will be replaced
            self.my_len,
            self.opp_len,
            self.my_boosts,
            self.opp_boosts,
            self.turn + 1,
            self.player_number
        )

    def is_safe(self, pos):
        return self.board[pos[1], pos[0]] == EMPTY

    def step(self, pos, w, h, direction):
        dx, dy = DIRS[direction]
        return _numba_wrap(pos[0] + dx, pos[1] + dy, w, h)

    def get_valid_moves(self, trail, current_dir):
        """Get all non-suicidal moves (no 180s)."""
        moves = []
        for d_name in DIR_NAMES:
            if d_name != OPPOSITE.get(current_dir):
                moves.append(d_name)
        return moves

    def simulate_step(self, my_move, opp_move):
        """
        Simulates one full turn given moves for both players.
        This is a SEQUENTIAL simulation, matching case_closed_game.py
        Player 1 always moves first.
        """
        
        # --- 1. Identify moves for P1 and P2 ---
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

        # --- 2. Parse moves and create new state ---
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

        # --- 3. Simulate Player 1's Move ---
        for i in range(p1_steps):
            if not p1_alive: break
            
            p1_head = self.step(p1_head, self.w, self.h, p1_dir)
            
            if new_board[p1_head[1], p1_head[0]] == AGENT:
                # Collision detected
                if p1_head == p2_head:
                    # Head-on with P2's *current* (non-moved) head
                    if p1_len > p2_len:
                        p2_alive = False
                    elif p2_len > p1_len:
                        p1_alive = False
                    else: # Equal length
                        p1_alive = False
                        p2_alive = False
                else:
                    # Hit a trail (or our own)
                    p1_alive = False
            
            if p1_alive:
                new_board[p1_head[1], p1_head[0]] = AGENT
                new_p1_trail.append(p1_head)
                new_p1_len += 1
            else:
                break # Stop moving if dead

        # --- 4. Simulate Player 2's Move ---
        for i in range(p2_steps):
            if not p2_alive: break
            
            p2_head = self.step(p2_head, self.w, self.h, p2_dir)
            
            if new_board[p2_head[1], p2_head[0]] == AGENT:
                # Collision detected
                if p1_alive and p2_head == p1_head:
                    # Head-on with P1's *new* head
                    if p2_len > p1_len:
                        p1_alive = False
                    elif p1_len > p2_len:
                        p2_alive = False
                    else: # Equal length
                        p1_alive = False
                        p2_alive = False
                else:
                    # Hit a trail (or P1's new trail, or own)
                    p2_alive = False
            
            if p2_alive:
                new_board[p2_head[1], p2_head[0]] = AGENT
                new_p2_trail.append(p2_head)
                new_p2_len += 1
            else:
                break # Stop moving if dead

        # --- 5. Return new state or terminal value ---
        my_alive = p1_alive if self.player_number == 1 else p2_alive
        opp_alive = p2_alive if self.player_number == 1 else p1_alive
        
        if not my_alive and not opp_alive:
            return 0 # Draw
        if not my_alive:
            return -1e9 # I lose
        if not opp_alive:
            return 1e9 # I win

        # Both alive, return the new state, swapping back if needed
        if self.player_number == 1:
            return GameState(new_board, new_p1_trail, new_p2_trail,
                             new_p1_len, new_p2_len, new_p1_boosts, new_p2_boosts,
                             self.turn + 1, self.player_number)
        else:
            return GameState(new_board, new_p2_trail, new_p1_trail,
                             new_p2_len, new_p1_len, new_p2_boosts, new_p1_boosts,
                             self.turn + 1, self.player_number)

# --- Heuristic & Search Functions ---

def get_move_sort_key(state, move, my_head, w, h):
    """
    (NEW) Generates a 'quick score' for a move to be used for sorting.
    We just want to avoid corridors.
    """
    move_dir, use_boost = move.split(":")
    
    # Simulate 1 step
    next_pos = state.step(my_head, w, h, move_dir)
    if not state.is_safe(next_pos):
        return -1e10 # Instant death, search this last
    
    score = _numba_corridor_penalty(state.board, next_pos[0], next_pos[1], w, h)
    
    if use_boost == "B":
        # Simulate 2nd step
        next_pos_2 = state.step(next_pos, w, h, move_dir)
        if not state.is_safe(next_pos_2):
            return -1e10 # Instant death on 2nd step
        
        # Add penalty for 2nd step, but weight 1st step more
        score = (score * 0.6) + (_numba_corridor_penalty(state.board, next_pos_2[0], next_pos_2[1], w, h) * 0.4)
        
    return score

def evaluate_state(state):
    """
    (UPDATED) Scores a non-terminal game state using an AGGRESSIVE heuristic.
    A positive score is good for me, negative is bad.
    """
    
    # --- 1. Terminal State Check (Fastest Check) ---
    my_moves = 0
    opp_moves = 0
    local_dir_vectors = ((0, -1), (0, 1), (-1, 0), (1, 0)) # UP, DOWN, LEFT, RIGHT
    
    for i in range(4):
        dx, dy = local_dir_vectors[i]
        my_n = _numba_wrap(state.my_head[0] + dx, state.my_head[1] + dy, state.w, state.h)
        opp_n = _numba_wrap(state.opp_head[0] + dx, state.opp_head[1] + dy, state.w, state.h)
        if state.board[my_n[1], my_n[0]] == EMPTY:
            my_moves += 1
        if state.board[opp_n[1], opp_n[0]] == EMPTY:
            opp_moves += 1

    if my_moves == 0 and opp_moves == 0: return 0 # Draw
    if my_moves == 0: return -1e9 + state.turn # Lose (prefer losing later)
    if opp_moves == 0: return 1e9 - state.turn # Win (prefer winning sooner)

    # --- 2. End-Game Length Battle ---
    # Check this early. If we are in the endgame, territory doesn't matter.
    if state.turn > (MAX_TURNS - 40): # Use a slightly larger endgame window
        length_score = (state.my_len - state.opp_len)
        return length_score * 10000.0 # Make this dominate all other heuristics

    # --- 3. Mid-Game Heuristics (Control & Aggression) ---
    
    # 3a. My Survival (Defensive): Heavily penalize *me* for being in a trap.
    my_corridor_penalty = _numba_corridor_penalty(
        state.board, state.my_head[0], state.my_head[1], state.w, state.h
    )
    
    # 3b. Opponent Survival (Aggressive): Heavily *reward* *forcing the opponent* into a trap.
    # We multiply by -1 because their penalty (a negative num) should be a bonus for us.
    opp_corridor_penalty = _numba_corridor_penalty(
        state.board, state.opp_head[0], state.opp_head[1], state.w, state.h
    ) * -1.0 # <-- This is the aggression switch

    # 3c. Territory Control (Voronoi)
    my_area, opp_area = _numba_voronoi(
        state.board, state.my_head, state.opp_head, state.w, state.h
    )

    # --- 4. Final Weighted Score ---
    # Priority:
    # 1. (Handled by 3a/3b): Don't die, and kill the opponent.
    #    The corridor penalties are so large (-5000, +5000) they will dominate.
    # 2. Maximize territory control.
    # 3. Conserve boosts and grow length.
    
    territory_score = (my_area - opp_area) * 100.0
    length_score = (state.my_len - state.opp_len) * 10.0 # Length is less important mid-game
    boost_score = (state.my_boosts - state.opp_boosts) * 50.0 # Boosts are a good tie-breaker
    
    # Combine all heuristics
    final_score = (
        territory_score 
        + length_score 
        + boost_score
        + my_corridor_penalty  # This is a large negative if I'm trapped
        + opp_corridor_penalty # This is a large positive if they are trapped
    )
    
    return final_score


class SearchTimeout(Exception):
    pass

def maximin_search(state, depth, alpha, beta, start_time, time_limit, my_current_dir, opp_current_dir):
    """
    (UPDATED) Recursive Maximin search with Transposition Table caching.
    """
    
    # --- 1. Time Limit Check ---
    if (time.time() - start_time) > time_limit:
        raise SearchTimeout()
    
    # --- 2. Transposition Table Check (NEW) ---
    # A "state" is defined by its board/positions AND the directions agents are locked into
    state_key = (state.get_hash(), my_current_dir, opp_current_dir)
    if state_key in TT_CACHE:
        cached_depth, cached_score = TT_CACHE[state_key]
        if cached_depth >= depth:
            return cached_score # We found a better-or-equal-depth score
            
    # --- 3. Base Case: Reached depth limit or terminal state ---
    if depth == 0:
        return evaluate_state(state) # Score this leaf node

    # --- 4. Get All Possible Moves ---
    my_valid_dirs = state.get_valid_moves(state.my_trail, my_current_dir)
    opp_valid_dirs = state.get_valid_moves(state.opp_trail, opp_current_dir)

    my_moves = []
    for d in my_valid_dirs:
        my_moves.append(f"{d}:NB") # No Boost
        if state.my_boosts > 0:
            my_moves.append(f"{d}:B") # Boost
    
    opp_moves = []
    for d in opp_valid_dirs:
        opp_moves.append(f"{d}:NB")
        if state.opp_boosts > 0:
            opp_moves.append(f"{d}:B")

    # Failsafe: If no moves, it's a terminal state
    if not my_moves: return -1e9 + state.turn # I lose
    if not opp_moves: return 1e9 - state.turn # I win
            
    # --- 5. "Maxi" node (My Turn) ---
    best_score = -np.inf
    
    # (Move ordering is handled at the root, in decide_action)
    
    for my_move in my_moves:
        
        # --- 6. "Min" node (Opponent's Reply) ---
        worst_reply_score = np.inf
        
        for opp_move in opp_moves:
            
            sim_result = state.simulate_step(my_move, opp_move)
            
            score = 0
            if isinstance(sim_result, (int, float)):
                # Game ended (win/loss/draw)
                score = sim_result
            else:
                # Game continues, recurse
                score = maximin_search(
                    sim_result, depth - 1, alpha, beta, 
                    start_time, time_limit, 
                    my_move.split(":")[0], opp_move.split(":")[0]
                )
            
            worst_reply_score = min(worst_reply_score, score)
            
            beta = min(beta, worst_reply_score)
            if beta <= alpha:
                break # Prune
        
        best_score = max(best_score, worst_reply_score)
        
        alpha = max(alpha, best_score)
        if beta <= alpha:
            break # Prune

    # --- 7. Store in Transposition Table (NEW) ---
    TT_CACHE[state_key] = (depth, best_score)
    return best_score


# --- Flask Server (Keep as-is from winner) ---

app = Flask(__name__)

PARTICIPANT = os.getenv("PARTICIPANT", "ParticipantX")
AGENT_NAME = os.getenv("AGENT_NAME", "CaseClosed-BestSuperDemon")

state_lock = Lock()
LAST = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
}

# --- Utility: Trail/Dir Helpers (Keep from winner) ---

def trail_head(trail):
    return trail[-1] if trail else None

def last_dir_from_trail(trail, w, h):
    """Infer current direction from last two positions, torus aware."""
    if len(trail) < 2:
        return "RIGHT"  # default opening direction
    x2, y2 = trail[-1]
    x1, y1 = trail[-2]
    dx, dy = x2 - x1, y2 - y1
    
    if abs(dx) > 1: dx = -np.sign(dx) # Wrap
    if abs(dy) > 1: dy = -np.sign(dy) # Wrap

    if dx == 1: return "RIGHT"
    if dx == -1: return "LEFT"
    if dy == 1: return "DOWN"
    if dy == -1: return "UP"
    return "RIGHT" # Should not happen

# --- NEW: Main Action Function ---

def decide_action(state, player_number):
    
    start_time = time.time()
    
    # --- 0. Clear the Transposition Table Cache (NEW) ---
    global TT_CACHE
    TT_CACHE.clear()
    
    # --- 1. Parse State (Unchanged) ---
    if player_number == 1:
        my_trail = deque(tuple(p) for p in state["agent1_trail"])
        opp_trail = deque(tuple(p) for p in state["agent2_trail"])
        my_len = int(state["agent1_length"])
        opp_len = int(state["agent2_length"])
        my_boosts = int(state["agent1_boosts"])
        opp_boosts = int(state["agent2_boosts"])
    else:
        my_trail = deque(tuple(p) for p in state["agent2_trail"])
        opp_trail = deque(tuple(p) for p in state["agent1_trail"])
        my_len = int(state["agent2_length"])
        opp_len = int(state["agent1_length"])
        my_boosts = int(state["agent2_boosts"])
        opp_boosts = int(state["agent1_boosts"])
        
    turn = int(state.get("turn_count", 0))

    if not my_trail or not opp_trail:
        return "RIGHT" # Opening move

    # Use numpy for the board
    board_list = state["board"]
    board_np = np.array(board_list, dtype=np.int8)
    
    w, h = BOARD_WIDTH, BOARD_HEIGHT
    my_dir = last_dir_from_trail(my_trail, w, h)
    opp_dir = last_dir_from_trail(opp_trail, w, h)

    # --- 2. Create Root GameState (Unchanged) ---
    root_state = GameState(
        board_np, my_trail, opp_trail, my_len, opp_len,
        my_boosts, opp_boosts, turn, player_number
    )
    
    # --- 3. Get My Moves & Immediate Safety Check ---
    my_valid_dirs = root_state.get_valid_moves(my_trail, my_dir)
    safe_moves = []
    
    for d in my_valid_dirs:
        next_pos = root_state.step(root_state.my_head, w, h, d)
        if root_state.is_safe(next_pos):
            safe_moves.append(f"{d}:NB")
        
            if my_boosts > 0:
                # Only check boost if 1st step is safe
                next_pos_2 = root_state.step(next_pos, w, h, d)
                if root_state.is_safe(next_pos_2):
                    safe_moves.append(f"{d}:B")
        elif my_boosts > 0:
             # Check if boost can "jump" an unsafe square
             # This is a critical edge case!
             next_pos_2 = root_state.step(next_pos, w, h, d)
             if root_state.is_safe(next_pos_2):
                # Is the *first* step safe? No.
                # Is the *second* step safe? Yes.
                # Can we jump? Only if the first step isn't our *own* head.
                # In this game, a boost moves 1, then 1. It doesn't jump.
                # The simulation handles this, but for the root-level
                # safety check, we must be careful.
                # Re-reading simulate_step: it checks step 1, *then* step 2.
                # So if step 1 is a wall, a boost is suicide.
                # My logic above is correct:  
                # - if root_state.is_safe(next_pos):
                #   - safe_moves.append(f"{d}:NB")
                #   - if my_boosts > 0 and root_state.is_safe(next_pos_2):
                #     - safe_moves.append(f"{d}:B")
                pass # Already handled

    if not safe_moves:
        print(f"TURN {turn}: ALERT: NO SAFE MOVES. FORCING {my_dir}.")
        # If no moves are "safe" (e.g., all lead to walls),
        # we must still provide a move. The search will find
        # the "best way to die" (e.g., a draw vs. a loss).
        # But if the list is empty, we must provide *something*.
        # We will let the search run on the *unsafe* moves.
        if not my_valid_dirs: # Completely trapped, 180 is only option
             return "UP" # Will be invalid, but we have to send something
        
        # We have valid (non-180) dirs, but all are "unsafe" (hit trail)
        # We must search these to find the best outcome.
        print(f"TURN {turn}: WARNING: All valid moves are unsafe. Searching for best loss...")
        safe_moves = [] # Re-populate with unsafe moves
        for d in my_valid_dirs:
            safe_moves.append(f"{d}:NB")
            if my_boosts > 0:
                safe_moves.append(f"{d}:B")

        if not safe_moves: # Should be impossible now
            return "UP" 

    # --- 4. Iterative Deepening Search on SAFE MOVES ---
    
    # --- 4a. Smart Move Ordering (NEW) ---
    # Sort moves by the 1-ply heuristic.
    # We want higher scores (e.g., 0.0) first, lower scores (e.g., -5000.0) last.
    try:
        sorted_safe_moves = sorted(
            safe_moves, 
            key=lambda m: get_move_sort_key(root_state, m, root_state.my_head, w, h), 
            reverse=True
        )
    except Exception as e:
        print(f"Error during move sort: {e}. Using unsorted list.")
        sorted_safe_moves = safe_moves

    best_overall_move = sorted_safe_moves[0] # Failsafe from the *sorted* list
    best_overall_score = -np.inf
    
    # Start at depth 3.
    for depth in range(3, 20): 
        
        best_move_for_depth = best_overall_move
        best_score_for_depth = -np.inf
        
        # --- 4b. Move Best-from-Last-Iter to Front (NEW) ---
        # This ensures we search the previously-best path first.
        if best_overall_move in sorted_safe_moves:
            sorted_safe_moves.insert(0, sorted_safe_moves.pop(sorted_safe_moves.index(best_overall_move)))

        try:
            for my_move in sorted_safe_moves: # <--- Iterate over safe_moves list
                
                # Find the opponent's best reply (worst outcome for me)
                worst_reply_score = np.inf
                
                opp_valid_dirs = root_state.get_valid_moves(opp_trail, opp_dir)
                opp_moves = []
                for d in opp_valid_dirs:
                    opp_moves.append(f"{d}:NB")
                    if opp_boosts > 0:
                        opp_moves.append(f"{d}:B")
                
                if not opp_moves:
                    worst_reply_score = 1e9 - turn # Opponent has no moves, I win
                else:
                    for opp_move in opp_moves:
                        sim_result = root_state.simulate_step(my_move, opp_move)
                        
                        score = 0
                        if isinstance(sim_result, (int, float)):
                            score = sim_result
                        else:
                            score = maximin_search(
                                sim_result, depth - 1, -np.inf, np.inf,
                                start_time, MOVE_TIME_LIMIT,
                                my_move.split(":")[0], opp_move.split(":")[0]
                            )
                        
                        worst_reply_score = min(worst_reply_score, score)

                if worst_reply_score > best_score_for_depth:
                    best_score_for_depth = worst_reply_score
                    best_move_for_depth = my_move

            # If we completed this depth without timeout, this is now our best-known move
            best_overall_move = best_move_for_depth
            best_overall_score = best_score_for_depth
            
            # If we found a forced win, stop searching
            if best_overall_score >= 1e9:
                # print(f"Found forced win at depth {depth}.")
                break

        except SearchTimeout:
            # Time's up! Return the best move from the *previous* completed depth.
            # print(f"Timeout at depth {depth}. Using best from depth {depth-1}.")
            break
        except Exception as e:
            print(f"!!! SEARCH ERROR at depth {depth}: {e}")
            print(f"My move: {my_move}, Opp moves: {opp_moves}")
            import traceback
            traceback.print_exc()
            break # Exit loop and use previous best
    
    # --- 5. Format and Return (Unchanged) ---
    final_move_dir, final_boost = best_overall_move.split(":")
    
    # print(f"Turn {turn}: Chose {best_overall_move} (Score: {best_overall_score:.0f}) (Time: {(time.time() - start_time):.3f}s)")

    if final_boost == "B":
        return f"{final_move_dir}:BOOST"
    else:
        return final_move_dir

# ----------------------- Flask endpoints (Unchanged) -----------------------

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    with state_lock:
        for k in LAST.keys():
            if k in data:
                LAST[k] = data[k]
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
                if k in data:
                    LAST[k] = data[k]
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    
    # Warm up numba JIT compilation before the first request
    print("Warming up Numba JIT... (this may take a moment)")
    try:
        # Create dummy data for warmup
        dummy_board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
        dummy_board[5, 5] = AGENT
        dummy_board[5, 6] = AGENT
        dummy_board[10, 10] = AGENT
        dummy_board[10, 9] = AGENT

        # Warmup functions
        _numba_wrap(1, 1, BOARD_WIDTH, BOARD_HEIGHT)
        _numba_corridor_penalty(dummy_board, 1, 1, BOARD_WIDTH, BOARD_HEIGHT)
        _numba_voronoi(dummy_board, (5, 6), (10, 9), BOARD_WIDTH, BOARD_HEIGHT)
        
        print("Numba JIT compiled successfully.")
    except Exception as e:
        print(f"Numba warmup failed (this is OK, will compile on first run): {e}")
        
    app.run(host="0.0.0.0", port=port, debug=False)