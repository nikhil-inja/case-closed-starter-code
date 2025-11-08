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
sys.setrecursionlimit(2000)

# --- Core Game Constants (from case_closed_game.py for reference) ---
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

DIRECTION_MAP = {d.name: d for d in Direction}
# Reverse map for movement simulation
OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}
GRID_HEIGHT = 18
GRID_WIDTH = 20
EMPTY = 0
AGENT_TRAIL = 1 # Represents any trail/wall
MAX_SEARCH_DEPTH = 6 # Max depth to search in Minimax (adjust based on timeout)
MAX_TIME_MS = 3800  # 3.8 seconds max per move (4s limit)

# Flask API server setup
app = Flask(__name__)

# Global state management
GLOBAL_GAME_STATE = {
    "board": None,
    "agent1_trail": deque(),
    "agent2_trail": deque(),
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
    "my_direction": Direction.RIGHT,
    "opp_direction": Direction.LEFT
}

game_lock = Lock()
START_TIME = 0.0

# Agent identity
PARTICIPANT = "GeminiAI_Agent"
AGENT_NAME = "CaseClosed_Minimax"

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

def _torus_check(pos):
    """Handles the wraparound logic for the toroidal grid."""
    x, y = pos
    normalized_x = x % GRID_WIDTH
    normalized_y = y % GRID_HEIGHT
    return (normalized_x, normalized_y)

def flood_fill(board, start_pos, my_id):
    """
    Flood-Fill heuristic: Calculates the size of the open, connected region
    accessible from the start_pos. This is the core evaluation function.
    """
    if board[start_pos[1], start_pos[0]] == AGENT_TRAIL:
        return 0

    q = deque([start_pos])
    visited = {start_pos}
    count = 0

    while q:
        x, y = q.popleft()
        count += 1
        
        # Check all 4 neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = _torus_check((x + dx, y + dy))
            next_pos = (next_x, next_y)
            
            # If the neighbor is open and hasn't been visited, add it to the queue
            if next_pos not in visited and board[next_y, next_x] == EMPTY:
                visited.add(next_pos)
                q.append(next_pos)
                
    return count

def evaluate_state(state, my_id):
    """
    Evaluates the board state by calculating the space advantage.
    Score = (My Space) - (Opponent Space)
    """
    board = state["board"]
    
    # Identify agent positions
    if my_id == 1:
        my_head = state["agent1_trail"][-1]
        opp_head = state["agent2_trail"][-1]
    else:
        my_head = state["agent2_trail"][-1]
        opp_head = state["agent1_trail"][-1]
        
    # Check if the agent is dead (in the simulation)
    if board[my_head[1], my_head[0]] == AGENT_TRAIL:
        return -999999  # Massive penalty for dying
    if board[opp_head[1], opp_head[0]] == AGENT_TRAIL:
        return 999999   # Massive reward for opponent dying
        
    my_space = flood_fill(board, my_head, my_id)
    opp_space = flood_fill(board, opp_head, my_id)
    
    # Core heuristic: maximize my space, minimize opponent's space
    score = my_space - opp_space
    
    # Tie-breaker: Manually check if opponent is trapped (space = 0)
    if opp_space == 0 and my_space > 0:
        return 999999
    
    return score

def get_possible_moves(agent_dir, boosts_remaining):
    """
    Generates all valid moves (Direction and Boost/No Boost) for a single agent,
    excluding the move opposite to the current direction.
    """
    valid_moves = []
    
    for direction in Direction:
        # Check for illegal move (opposite direction)
        if direction != OPPOSITE_DIR.get(agent_dir):
            valid_moves.append((direction, False)) # No boost
            if boosts_remaining > 0:
                valid_moves.append((direction, True)) # With boost
                
    return valid_moves

def simulate_move(current_state, player_id, direction, use_boost):
    """
    Simulates a single agent's move (1 or 2 steps if boosted) on a copy of the state.
    Returns the new state and whether the agent survived the move.
    """
    new_state = copy.deepcopy(current_state)
    
    # Pointers to the moving agent's data in the new state
    if player_id == 1:
        my_trail = new_state["agent1_trail"]
        my_boosts_key = "agent1_boosts"
        my_dir_key = "my_direction"
        opp_trail = new_state["agent2_trail"]
    else:
        my_trail = new_state["agent2_trail"]
        my_boosts_key = "agent2_boosts"
        my_dir_key = "opp_direction" # Note: we use 'opp_direction' to track P2's direction
        opp_trail = new_state["agent1_trail"]

    # Handle boost usage
    num_steps = 1
    if use_boost:
        if new_state[my_boosts_key] > 0:
            num_steps = 2
            new_state[my_boosts_key] -= 1
        else:
            # If agent requests boost but has none, it moves normally
            use_boost = False

    agent_alive = True
    current_head = my_trail[-1]
    
    for _ in range(num_steps):
        if not agent_alive: break

        dx, dy = direction.value
        next_head = _torus_check((current_head[0] + dx, current_head[1] + dy))
        
        # 1. Collision Check
        if new_state["board"][next_head[1], next_head[0]] == AGENT_TRAIL:
            agent_alive = False
            break

        # 2. Update state (no collision)
        # Mark the new position on the board
        new_state["board"][next_head[1], next_head[0]] = AGENT_TRAIL
        # Add new head to trail
        my_trail.append(next_head)
        current_head = next_head
        
        
    # Update direction for the next turn
    if agent_alive:
        new_state[my_dir_key] = direction
        
    return new_state, agent_alive

def minimax_search(state, depth, is_max_turn, my_id, alpha, beta):
    """
    Minimax with Alpha-Beta Pruning for simultaneous turn-based game.
    The top level must account for the opponent's best response to our move.
    """
    if time.time() * 1000 > START_TIME + MAX_TIME_MS or depth == 0:
        return evaluate_state(state, my_id)

    # Identify the two agents based on my_id
    if my_id == 1:
        max_id = 1
        min_id = 2
        max_dir = state["my_direction"]
        min_dir = state["opp_direction"]
        max_boosts = state["agent1_boosts"]
        min_boosts = state["agent2_boosts"]
    else:
        max_id = 2
        min_id = 1
        max_dir = state["opp_direction"]
        min_dir = state["my_direction"]
        max_boosts = state["agent2_boosts"]
        min_boosts = state["agent1_boosts"]

    # ------------------ MAXIMIZING PLAYER (My Agent) ------------------
    if is_max_turn:
        value = -np.inf
        
        # Max moves: All my possible directions (+ boost option)
        max_moves = get_possible_moves(max_dir, max_boosts)
        
        # Min moves: All opponent's possible directions (+ boost option)
        min_moves = get_possible_moves(min_dir, min_boosts)

        # For the MAX player (us), we are iterating over *our* possible moves (m)
        # and then calculating the *worst-case* scenario (opponent's best response, o)
        for my_dir, my_boost in max_moves:
            worst_case_score = np.inf # Assume opponent plays optimally against this move
            
            # Iterate over all possible opponent responses (simultaneous move)
            for opp_dir, opp_boost in min_moves:
                
                # 1. Simulate my move on a clean copy
                state_after_my_move, my_survived = simulate_move(state, max_id, my_dir, my_boost)
                
                # 2. Simulate opponent's move on the resulting state
                state_after_opp_move, opp_survived = simulate_move(state_after_my_move, min_id, opp_dir, opp_boost)
                
                # 3. Check for simultaneous crash (Draw/Loss) - only needed if the judge handles draw differently than two losses
                if not my_survived and not opp_survived:
                    score = 0 # Draw score
                elif not my_survived:
                    score = -999999 # Clear loss
                elif not opp_survived:
                    score = 999999 # Clear win
                else:
                    # Recursive call for the next turn (switched to MIN turn)
                    score = minimax_search(state_after_opp_move, depth - 1, False, my_id, alpha, beta)
                
                # Update the worst-case score (opponent minimizes my score)
                worst_case_score = min(worst_case_score, score)
                
                if worst_case_score <= alpha:
                    break # Beta cutoff (opponent won't let me get this move)
            
            value = max(value, worst_case_score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
                
        return value

    # ------------------ MINIMIZING PLAYER (Opponent's Future Turn) ------------------
    # The agent is calculating the MINIMIZING value for the opponent's turn.
    # We don't need this branch because we model simultaneous moves in the MAX turn only.
    # The depth 0 evaluation acts as the terminal node.
    else:
        # Since we use simultaneous search, the depth only decreases by 1
        # at the end of the full turn simulation. This branch should not be reached.
        return evaluate_state(state, my_id)

def get_best_move(state, my_id):
    """
    Main function to initiate the Minimax search and select the best move.
    """
    global START_TIME
    START_TIME = time.time() * 1000 # Time in milliseconds
    
    # Identify agent data
    if my_id == 1:
        my_dir = state["my_direction"]
        my_boosts = state["agent1_boosts"]
        opp_dir = state["opp_direction"]
        opp_boosts = state["agent2_boosts"]
    else:
        my_dir = state["opp_direction"]
        my_boosts = state["agent2_boosts"]
        opp_dir = state["my_direction"]
        opp_boosts = state["agent1_boosts"]

    # Max moves: All my possible directions (+ boost option)
    my_moves = get_possible_moves(my_dir, my_boosts)
    
    # Min moves: All opponent's possible directions (+ boost option)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    best_move = (my_dir, False) # Default to current direction
    max_score = -np.inf

    # Iterate over all my possible moves (m)
    for my_dir, my_boost in my_moves:
        
        # Calculate the worst-case score if the opponent plays optimally against this move
        worst_case_score = np.inf
        
        # Iterate over all possible opponent responses (o)
        for opp_dir, opp_boost in opp_moves:
            
            # 1. Simulate my move
            state_after_my_move, my_survived = simulate_move(state, my_id, my_dir, my_boost)
            
            # 2. Simulate opponent's move
            state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 3 - my_id, opp_dir, opp_boost)
            
            # 3. Collision/Terminal check
            if not my_survived and not opp_survived:
                score = 0 # Draw
            elif not my_survived:
                score = -999999
            elif not opp_survived:
                score = 999999
            else:
                # 4. Recursive call (Minimax) for remaining depth (MIN turn)
                # Note: We start in MAX turn (is_max_turn=True) but since we do the 
                # simulation in the loop, the first actual recursive call is for the 
                # next turn, so we call it with a slightly shallower depth.
                score = minimax_search(state_after_opp_move, MAX_SEARCH_DEPTH - 1, True, my_id, -np.inf, np.inf)

            # Update the worst-case score (opponent minimizes my score)
            worst_case_score = min(worst_case_score, score)

            # Simple pruning at the root
            if worst_case_score > max_score:
                break # Optimization: if we find a path better than the best so far, stop searching the opp moves
        
        # Check if this move is better than the best found so far
        if worst_case_score > max_score:
            max_score = worst_case_score
            best_move = (my_dir, my_boost)
            
        # Time check (for safety)
        if time.time() * 1000 > START_TIME + MAX_TIME_MS:
            break
            
    final_dir, final_boost = best_move
    
    # Format the move string
    move_str = final_dir.name
    if final_boost:
        move_str += ":BOOST"
        
    print(f"Turn {state.get('turn_count', 0)}: Chosen move {move_str} with estimated score {max_score}")
    return move_str


def _update_local_game_from_post(data: dict):
    """Updates the GLOBAL_GAME_STATE using the JSON posted by the judge."""
    with game_lock:
        GLOBAL_GAME_STATE.update(data)
        
        # Update board (convert list of lists to numpy array for fast lookups)
        if "board" in data:
            GLOBAL_GAME_STATE["board"] = np.array(data["board"], dtype=np.int8)

        # Update trails (convert list of lists/tuples to deque of tuples)
        if "agent1_trail" in data:
            GLOBAL_GAME_STATE["agent1_trail"] = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME_STATE["agent2_trail"] = deque(tuple(p) for p in data["agent2_trail"])

        # Determine current directions based on the last two trail segments
        # This is a critical step because the judge doesn't send the direction
        def get_current_direction(trail):
            if len(trail) < 2:
                # Use a default direction if trail is too short (shouldn't happen after turn 1)
                return Direction.RIGHT 
            
            head = trail[-1]
            prev = trail[-2]
            
            # Account for Torus wraparound when determining direction
            dx = head[0] - prev[0]
            dy = head[1] - prev[1]
            
            # Check for wraparound (e.g., 0 to 19 or 19 to 0)
            if abs(dx) > 1: dx = -1 if dx > 0 else 1
            if abs(dy) > 1: dy = -1 if dy > 0 else 1

            if dx == 1: return Direction.RIGHT
            if dx == -1: return Direction.LEFT
            if dy == 1: return Direction.DOWN
            if dy == -1: return Direction.UP
            return Direction.RIGHT # Default safe guess

        if len(GLOBAL_GAME_STATE["agent1_trail"]) >= 2:
            GLOBAL_GAME_STATE["my_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent1_trail"])
        if len(GLOBAL_GAME_STATE["agent2_trail"]) >= 2:
            GLOBAL_GAME_STATE["opp_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent2_trail"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick."""
    player_number = request.args.get("player_number", default=1, type=int)

    # Note: If the agent is Player 2, we must swap the 'my_direction' and 'opp_direction'
    # in the state dictionary *before* passing it to the solver.
    # The logic in _update_local_game_from_post ensures that agent1 always corresponds 
    # to 'my_direction' and agent2 to 'opp_direction' temporarily for easy swapping later.
    
    with game_lock:
        current_state = copy.deepcopy(GLOBAL_GAME_STATE)
        current_state["player_number"] = player_number

        # If we are Player 2, swap 'my_direction' and 'opp_direction' pointers for the solver
        if player_number == 2:
             # The solver always assumes my data is first (agent1 data in the state)
             # Swap 1's and 2's data in the local copy for the search function
             current_state["agent1_trail"], current_state["agent2_trail"] = current_state["agent2_trail"], current_state["agent1_trail"]
             current_state["agent1_boosts"], current_state["agent2_boosts"] = current_state["agent2_boosts"], current_state["agent1_boosts"]
             # The solver expects my direction in 'my_direction' and opp in 'opp_direction'
             current_state["my_direction"], current_state["opp_direction"] = current_state["opp_direction"], current_state["my_direction"]


    # ----------------- YOUR CODE GOES HERE -------------------
    # Minimax Search is initiated here
    move = get_best_move(current_state, player_number)
    # ----------------- END CODE HERE --------------------

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)