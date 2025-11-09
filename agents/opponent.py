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
DEBUG_EVAL = False  # Set to True to see evaluation details

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
    accessible from the start_pos. Only counts EMPTY cells, not the agent's trail.
    """
    visited = set()
    visited.add(start_pos)
    q = deque([start_pos])
    count = 0

    while q:
        x, y = q.popleft()
        
        # Check all 4 neighbors
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_x, next_y = _torus_check((x + dx, y + dy))
            next_pos = (next_x, next_y)
            
            # Only count and explore EMPTY cells
            if next_pos not in visited and board[next_y, next_x] == EMPTY:
                visited.add(next_pos)
                q.append(next_pos)
                count += 1  # Count AFTER confirming it's empty
                
    return count

def calculate_center_distance(pos):
    """Calculate Manhattan distance from position to board center."""
    center_x, center_y = GRID_WIDTH // 2, GRID_HEIGHT // 2
    return abs(pos[0] - center_x) + abs(pos[1] - center_y)

def calculate_center_bonus(my_head, opp_head, my_id, turn_count):
    """Reward being closer to center, especially early game."""
    my_dist = calculate_center_distance(my_head)
    opp_dist = calculate_center_distance(opp_head)
    
    # Early game: high weight, Late game: low weight
    phase_weight = max(0, 1.0 - turn_count / 100.0)
    
    # Player-specific preference to break symmetry
    if my_id == 1:
        my_dist -= 0.5  # Slight preference for upper-center
    else:
        my_dist += 0.5  # Slight preference for lower-center
    
    return (opp_dist - my_dist) * 2.0 * phase_weight

def calculate_opponent_distance(my_head, opp_head):
    """Calculate Manhattan distance between agents."""
    return abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])

def calculate_pressure_score(my_head, opp_head, turn_count):
    """Reward optimal distance to opponent based on game phase."""
    distance = calculate_opponent_distance(my_head, opp_head)
    
    # Early game: don't worry about distance
    if turn_count < 12:
        return 0
    
    # Mid-late game: optimal distance is 5-10 cells
    if 5 <= distance <= 10:
        return 5.0  # Good pressure position
    elif distance < 5:
        return -2.0  # Too close, risky
    elif distance > 15:
        return -3.0  # Too far, passive
    else:
        return 0

def count_escape_routes(board, pos):
    """Count number of safe directions from position."""
    safe_directions = 0
    for direction in Direction:
        dx, dy = direction.value
        next_pos = _torus_check((pos[0] + dx, pos[1] + dy))
        if board[next_pos[1], next_pos[0]] == EMPTY:
            safe_directions += 1
    return safe_directions

def calculate_escape_quality(board, my_head, opp_head):
    """Compare escape route quality between agents."""
    my_routes = count_escape_routes(board, my_head)
    opp_routes = count_escape_routes(board, opp_head)
    
    # Enhanced route advantage calculation
    route_advantage = (my_routes - opp_routes) * 4.0  # Increased from 3.0
    
    # Graduated bonuses
    if opp_routes == 0 and my_routes > 0:
        route_advantage += 100.0
    elif opp_routes == 1 and my_routes > 2:
        route_advantage += 50.0
    elif opp_routes <= 2 and my_routes > 3:
        route_advantage += 25.0
    
    # Self-preservation bonuses
    if my_routes >= 4:
        route_advantage += 10.0
    elif my_routes == 3:
        route_advantage += 5.0
    
    return route_advantage

def detect_opponent_vulnerability(opp_space, total_space):
    """Detect if opponent is in vulnerable position (limited space)."""
    if total_space == 0:
        return 0
    
    opp_percentage = opp_space / total_space
    
    if opp_percentage < 0.1:  # Opponent has <10% of space
        return 50.0  # Large bonus for dominant position
    elif opp_percentage < 0.2:  # Opponent has <20% of space
        return 25.0  # Moderate bonus
    elif opp_percentage < 0.3:
        return 10.0  # Small bonus
    
    return 0

def calculate_trapping_bonus(board, my_space, opp_space, my_head, opp_head):
    """Calculate bonus for trapping opportunities."""
    total_space = my_space + opp_space
    
    # Vulnerability bonus
    vulnerability = detect_opponent_vulnerability(opp_space, total_space)
    
    # Escape route disparity
    escape_quality = calculate_escape_quality(board, my_head, opp_head)
    
    return vulnerability + escape_quality

def calculate_openness_bonus(board, my_head, my_space):
    """
    Reward positions with access to large open areas.
    Penalize being trapped in small pockets.
    """
    board_size = board.shape[0] * board.shape[1]
    openness_ratio = my_space / board_size
    
    if openness_ratio > 0.35:
        return 15.0
    elif openness_ratio > 0.25:
        return 10.0
    elif openness_ratio > 0.15:
        return 0
    elif openness_ratio > 0.08:
        return -15.0
    else:
        return -40.0

def get_game_phase(turn_count):
    """Determine current game phase."""
    if turn_count < 20:
        return "early"
    elif turn_count < 60:
        return "mid"
    else:
        return "late"

def evaluate_state(state, my_id):
    """
    Hybrid evaluation function combining strategic depth with survival mechanisms.
    Prioritizes: survival, space control, center positioning, opponent pressure, trapping.
    """
    board = state["board"]
    turn_count = state.get("turn_count", 0)
    
    # After swap in send_move, state is normalized:
    # agent1 = current player, agent2 = opponent
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    
    # Calculate reachable space
    my_space = flood_fill(board, my_head, my_id)
    opp_space = flood_fill(board, opp_head, my_id)
    
    # SURVIVAL LAYER 1: Emergency mode when critically low on space
    if my_space < 20:
        my_routes = count_escape_routes(board, my_head)
        
        # Pure survival scoring
        if my_routes == 0:
            return -999999  # Already dead
        elif my_routes == 1:
            return -800000 + my_space * 1000  # Desperate situation
        elif my_routes == 2:
            return -50000 + my_space * 500   # Critical but survivable
        else:
            # Focus on maximizing space and routes
            return my_space * 100 + my_routes * 5000
    
    # Calculate all metrics
    my_routes = count_escape_routes(board, my_head)
    opp_routes = count_escape_routes(board, opp_head)
    
    # SURVIVAL LAYER 2: Absolute safety check - prevent dangerous positions
    if my_routes <= 1:
        # Massive penalty for dangerous positions
        return -500000 + my_routes * 10000
    
    if my_routes == 2 and my_space < 15:
        # Very dangerous - limited options in small space
        return -100000 + my_space * 100
    
    # Core strategic metrics
    space_score = my_space - opp_space
    center_bonus = calculate_center_bonus(my_head, opp_head, my_id, turn_count)
    pressure_score = calculate_pressure_score(my_head, opp_head, turn_count)
    trapping_bonus = calculate_trapping_bonus(board, my_space, opp_space, my_head, opp_head)
    openness = calculate_openness_bonus(board, my_head, my_space)
    
    # Enhanced phase-aware weights with survival priority
    phase = get_game_phase(turn_count)
    if phase == "early":
        # Early: space claiming with center control, moderate caution
        weights = {"space": 1.0, "center": 0.8, "pressure": 0.2, "trapping": 0.8, "openness": 0.5}
    elif phase == "mid":
        # Mid: balanced with increased trapping focus
        weights = {"space": 1.0, "center": 0.2, "pressure": 0.6, "trapping": 2.0, "openness": 0.8}
    else:
        # Late: maximize trapping and territory control
        weights = {"space": 1.0, "center": 0.1, "pressure": 0.4, "trapping": 3.5, "openness": 1.0}
    
    # Combine all factors
    final_score = (
        weights["space"] * space_score +
        weights["center"] * center_bonus +
        weights["pressure"] * pressure_score +
        weights["trapping"] * trapping_bonus +
        weights["openness"] * openness
    )
    
    # Win condition check
    if opp_space == 0 and my_space > 0:
        return 999999
    
    return final_score

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
    
    # CRITICAL FIX: Increment turn count for proper game phase evaluation in simulations
    if "turn_count" in new_state:
        new_state["turn_count"] += 1
        
    return new_state, agent_alive

def minimax_search(state, depth, is_max_turn, my_id, alpha, beta):
    """
    Minimax with Alpha-Beta Pruning for simultaneous turn-based game.
    The top level must account for the opponent's best response to our move.
    """
    if time.time() * 1000 > START_TIME + MAX_TIME_MS or depth == 0:
        return evaluate_state(state, my_id)

    # After swap in send_move, state is normalized:
    # agent1 = current player (max), agent2 = opponent (min)
    # my_direction = current player, opp_direction = opponent
    max_id = 1  # After normalization, 1 is always current player
    min_id = 2  # After normalization, 2 is always opponent
    max_dir = state["my_direction"]
    min_dir = state["opp_direction"]
    max_boosts = state["agent1_boosts"]
    min_boosts = state["agent2_boosts"]

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
    
    # After the swap in send_move, state is already normalized:
    # my_direction = current player's direction
    # opp_direction = opponent's direction
    # agent1_trail/boosts = current player's data (after swap)
    # agent2_trail/boosts = opponent's data (after swap)
    my_dir = state["my_direction"]
    my_boosts = state["agent1_boosts"]
    opp_dir = state["opp_direction"]
    opp_boosts = state["agent2_boosts"]

    # Max moves: All my possible directions (+ boost option)
    my_moves = get_possible_moves(my_dir, my_boosts)
    
    # Min moves: All opponent's possible directions (+ boost option)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    best_move = (my_dir, False) # Default to current direction
    max_score = -np.inf

    # Iterate over all my possible moves (m)
    # FIXED: Changed variable names to avoid shadowing my_dir and opp_dir
    for move_dir, move_boost in my_moves:
        
        # Calculate the worst-case score if the opponent plays optimally against this move
        worst_case_score = np.inf
        
        # Iterate over all possible opponent responses (o)
        for opp_move_dir, opp_move_boost in opp_moves:
            
            # 1. Simulate my move (after normalization: always use 1 for current player)
            state_after_my_move, my_survived = simulate_move(state, 1, move_dir, move_boost)
            
            # 2. Simulate opponent's move (after normalization: always use 2 for opponent)
            state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 2, opp_move_dir, opp_move_boost)
            
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
            best_move = (move_dir, move_boost)
            
        # Time check (for safety)
        if time.time() * 1000 > START_TIME + MAX_TIME_MS:
            break
            
    final_dir, final_boost = best_move
    
    # Format the move string
    move_str = final_dir.name
    if final_boost:
        move_str += ":BOOST"
    
    # Debug output
    if DEBUG_EVAL:
        print(f"[Turn {state.get('turn_count', 0)}] Move: {final_dir.name}, Score: {max_score:.1f}")
        
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
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)