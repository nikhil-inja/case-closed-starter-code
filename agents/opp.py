"""
Aggressive Minimax Challenger - Tests agent.py with competitive AI
Uses minimax with different strategic priorities to create challenging games.
"""

import os
from flask import Flask, request, jsonify
from collections import deque
import numpy as np
import copy
import time
from enum import Enum

app = Flask(__name__)

# Basic identity
PARTICIPANT = os.getenv("PARTICIPANT", "Challenger_Participant")
AGENT_NAME = os.getenv("AGENT_NAME", "AggressiveChallenger")

# Direction Enum (matching game engine)
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)

# Direction mappings
OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}

# Grid constants
GRID_HEIGHT = 18
GRID_WIDTH = 20
EMPTY = 0
AGENT_TRAIL = 1

# Search parameters - Now matches agent.py for competitive play
MAX_SEARCH_DEPTH = 6  # Increased from 5 (now matches agent.py)
MAX_TIME_MS = 3800
START_TIME = 0

# Global game state
GLOBAL_GAME_STATE = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "my_direction": Direction.RIGHT,
    "opp_direction": Direction.RIGHT,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
}

import threading
game_lock = threading.Lock()


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Receive game state from judge."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Generate move using minimax."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        current_state = copy.deepcopy(GLOBAL_GAME_STATE)
        current_state["player_number"] = player_number
        
        if player_number == 2:
            current_state["agent1_trail"], current_state["agent2_trail"] = \
                current_state["agent2_trail"], current_state["agent1_trail"]
            current_state["agent1_boosts"], current_state["agent2_boosts"] = \
                current_state["agent2_boosts"], current_state["agent1_boosts"]
            current_state["my_direction"], current_state["opp_direction"] = \
                current_state["opp_direction"], current_state["my_direction"]
    
    move = get_best_move(current_state, player_number)
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Handle game end."""
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


def _update_local_game_from_post(data: dict):
    """Update local game state."""
    with game_lock:
        GLOBAL_GAME_STATE.update(data)
        
        if "board" in data:
            GLOBAL_GAME_STATE["board"] = np.array(data["board"], dtype=np.int8)
        
        if "agent1_trail" in data:
            trail1 = [tuple(p) if isinstance(p, list) else p for p in data["agent1_trail"]]
            GLOBAL_GAME_STATE["agent1_trail"] = deque(trail1, maxlen=500)
        
        if "agent2_trail" in data:
            trail2 = [tuple(p) if isinstance(p, list) else p for p in data["agent2_trail"]]
            GLOBAL_GAME_STATE["agent2_trail"] = deque(trail2, maxlen=500)
        
        # Infer directions from trails
        if len(GLOBAL_GAME_STATE["agent1_trail"]) >= 2:
            GLOBAL_GAME_STATE["my_direction"] = _infer_direction(
                GLOBAL_GAME_STATE["agent1_trail"][-2],
                GLOBAL_GAME_STATE["agent1_trail"][-1]
            )
        
        if len(GLOBAL_GAME_STATE["agent2_trail"]) >= 2:
            GLOBAL_GAME_STATE["opp_direction"] = _infer_direction(
                GLOBAL_GAME_STATE["agent2_trail"][-2],
                GLOBAL_GAME_STATE["agent2_trail"][-1]
            )


def _infer_direction(prev, curr):
    """Infer direction from two positions."""
    dx = curr[0] - prev[0]
    dy = curr[1] - prev[1]
    
    if abs(dx) > 1: dx = -1 if dx > 0 else 1
    if abs(dy) > 1: dy = -1 if dy > 0 else 1
    
    for direction in Direction:
        if direction.value == (dx, dy):
            return direction
    return Direction.RIGHT


def _torus_check(pos):
    """Handle torus wrapping."""
    return (pos[0] % GRID_WIDTH, pos[1] % GRID_HEIGHT)


def flood_fill(board, start_pos, my_id):
    """Calculate reachable space using BFS. Only counts EMPTY cells."""
    
    visited = set()
    visited.add(start_pos)
    q = deque([start_pos])
    count = 0
    
    while q:
        x, y = q.popleft()
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_pos = _torus_check((x + dx, y + dy))
            next_x, next_y = next_pos
            
            # Only count EMPTY cells
            if next_pos not in visited and board[next_y, next_x] == EMPTY:
                visited.add(next_pos)
                q.append(next_pos)
                count += 1  # Count AFTER confirming it's empty
    
    return count


def calculate_aggression_bonus(my_head, opp_head, my_space, opp_space, turn_count):
    """
    Aggressive evaluation - balanced with safety awareness.
    """
    distance = abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])
    
    if turn_count < 10:
        return 0  # Still racing for position
    elif turn_count < 30:
        # Early-mid: Measured aggression
        if 5 <= distance <= 10:
            return 12.0  # Good pressure distance
        elif 3 <= distance < 5:
            return 5.0   # Close but acceptable
        elif distance < 3:
            return -25.0  # Too close - very dangerous!
        else:
            return -3.0  # Bit passive
    else:
        # Late: Careful pursuit
        if 4 <= distance <= 8:
            return 15.0  # Optimal trap distance
        elif distance < 4:
            return -20.0  # Risk of collision
        elif distance <= 12:
            return 3.0
        else:
            return -8.0  # Too far in endgame


def count_escape_routes(board, pos):
    """Count safe directions from position."""
    safe_dirs = 0
    for direction in Direction:
        dx, dy = direction.value
        next_pos = _torus_check((pos[0] + dx, pos[1] + dy))
        if board[next_pos[1], next_pos[0]] == EMPTY:
            safe_dirs += 1
    return safe_dirs


def calculate_openness_bonus(board, my_head, my_space):
    """
    Reward positions with access to large open areas.
    Penalize being trapped in small pockets.
    """
    board_size = board.shape[0] * board.shape[1]
    openness_ratio = my_space / board_size
    
    if openness_ratio > 0.35:
        return 15.0  # Excellent - control large area
    elif openness_ratio > 0.25:
        return 10.0  # Good open space
    elif openness_ratio > 0.15:
        return 0     # Acceptable
    elif openness_ratio > 0.08:
        return -15.0  # Getting cramped
    else:
        return -40.0  # Dangerously confined


def evaluate_state_aggressive(state, my_id):
    """
    Aggressive evaluation function - different weights than agent.py.
    Prioritizes: aggressive positioning, space control, trapping.
    """
    board = state["board"]
    turn_count = state.get("turn_count", 0)
    
    # After swap in send_move, state is normalized:
    # agent1 = current player, agent2 = opponent
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    
    # Calculate space
    my_space = flood_fill(board, my_head, my_id)
    opp_space = flood_fill(board, opp_head, my_id)
    
    # Emergency survival mode when critically low on space
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
    
    # Core metric
    space_score = my_space - opp_space
    
    # Aggression bonus (different from agent.py)
    aggression = calculate_aggression_bonus(my_head, opp_head, my_space, opp_space, turn_count)
    
    # Escape route analysis
    my_routes = count_escape_routes(board, my_head)
    opp_routes = count_escape_routes(board, opp_head)
    
    # CRITICAL: Absolute safety check - prevent self-destruction
    if my_routes <= 1:
        # Massive penalty for dangerous positions
        # Slightly prefer 1 route over 0 to encourage escape attempts
        return -500000 + my_routes * 10000
        
    if my_routes == 2 and my_space < 15:
        # Very dangerous - limited options in small space
        return -100000 + my_space * 100
    
    # Base route comparison
    route_advantage = (my_routes - opp_routes) * 5.0  # Increased from 4.0
    
    # Bonus scaling based on opponent's routes
    if opp_routes == 0 and my_routes > 0:
        route_advantage += 100.0  # Opponent trapped - go for kill
    elif opp_routes == 1 and my_routes > 2:
        route_advantage += 50.0   # Opponent nearly trapped
    elif opp_routes <= 2 and my_routes > 3:
        route_advantage += 25.0   # Strong advantage
    
    # Self-preservation bonuses
    if my_routes >= 4:
        route_advantage += 10.0   # Safe position bonus
    elif my_routes == 3:
        route_advantage += 5.0    # Adequate safety
    
    # Space dominance bonus
    total_space = my_space + opp_space
    if total_space > 0 and opp_space / total_space < 0.15:
        space_dominance = 60.0  # Very high reward for dominating
    elif total_space > 0 and opp_space / total_space < 0.25:
        space_dominance = 30.0
    else:
        space_dominance = 0
    
    # Phase-specific weights - SURVIVAL-FOCUSED
    if turn_count < 15:
        # Early: Space claiming with caution
        weights = {"space": 1.0, "aggression": 0.4, "routes": 1.5, "dominance": 0.3}
    elif turn_count < 50:
        # Mid: Balanced with high safety priority
        weights = {"space": 1.0, "aggression": 0.7, "routes": 2.5, "dominance": 1.5}
    else:
        # Late: Careful trapping with self-preservation
        weights = {"space": 0.9, "aggression": 0.4, "routes": 3.0, "dominance": 3.0}
    
    # Calculate openness bonus
    openness = calculate_openness_bonus(board, my_head, my_space)
    
    final_score = (
        weights["space"] * space_score +
        weights["aggression"] * aggression +
        weights["routes"] * route_advantage +
        weights["dominance"] * space_dominance +
        openness  # Always included regardless of phase
    )
    
    # Win/loss detection
    if opp_space == 0 and my_space > 0:
        return 999999
    
    return final_score


def get_possible_moves(agent_dir, boosts_remaining):
    """Generate valid moves."""
    valid_moves = []
    
    for direction in Direction:
        if direction != OPPOSITE_DIR.get(agent_dir):
            valid_moves.append((direction, False))
            if boosts_remaining > 0:
                valid_moves.append((direction, True))
    
    return valid_moves


def simulate_move(current_state, player_id, direction, use_boost):
    """Simulate a move."""
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
    if use_boost:
        if new_state[my_boosts_key] > 0:
            num_steps = 2
            new_state[my_boosts_key] -= 1
        else:
            use_boost = False
    
    agent_alive = True
    current_head = my_trail[-1]
    
    for _ in range(num_steps):
        if not agent_alive:
            break
        
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


def minimax_search(state, depth, is_max_turn, my_id, alpha, beta):
    """Minimax with alpha-beta pruning."""
    if time.time() * 1000 > START_TIME + MAX_TIME_MS or depth == 0:
        return evaluate_state_aggressive(state, my_id)
    
    # After swap in send_move, state is normalized:
    # agent1 = current player (max), agent2 = opponent (min)
    # my_direction = current player, opp_direction = opponent
    max_id, min_id = 1, 2  # After normalization, 1 is always current player
    max_dir = state["my_direction"]
    min_dir = state["opp_direction"]
    max_boosts = state["agent1_boosts"]
    min_boosts = state["agent2_boosts"]
    
    if is_max_turn:
        value = -np.inf
        max_moves = get_possible_moves(max_dir, max_boosts)
        min_moves = get_possible_moves(min_dir, min_boosts)
        
        for my_dir, my_boost in max_moves:
            worst_case_score = np.inf
            
            for opp_dir, opp_boost in min_moves:
                state_after_my_move, my_survived = simulate_move(state, max_id, my_dir, my_boost)
                state_after_opp_move, opp_survived = simulate_move(state_after_my_move, min_id, opp_dir, opp_boost)
                
                if not my_survived and not opp_survived:
                    score = 0
                elif not my_survived:
                    score = -999999
                elif not opp_survived:
                    score = 999999
                else:
                    score = minimax_search(state_after_opp_move, depth - 1, False, my_id, alpha, beta)
                
                worst_case_score = min(worst_case_score, score)
                
                if worst_case_score <= alpha:
                    break
            
            value = max(value, worst_case_score)
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        
        return value
    else:
        return evaluate_state_aggressive(state, my_id)


def get_best_move(state, my_id):
    """Main function to select best move using minimax."""
    global START_TIME
    START_TIME = time.time() * 1000
    
    # After the swap in send_move, state is already normalized:
    # my_direction = current player's direction
    # opp_direction = opponent's direction
    # agent1_trail/boosts = current player's data (after swap)
    # agent2_trail/boosts = opponent's data (after swap)
    my_dir = state["my_direction"]
    my_boosts = state["agent1_boosts"]
    opp_dir = state["opp_direction"]
    opp_boosts = state["agent2_boosts"]
    
    my_moves = get_possible_moves(my_dir, my_boosts)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    best_move = (my_dir, False)
    max_score = -np.inf
    
    for move_dir, move_boost in my_moves:
        worst_case_score = np.inf
        
        for opp_move_dir, opp_move_boost in opp_moves:
            # After normalization: agent1 = current player (1), agent2 = opponent (2)
            state_after_my_move, my_survived = simulate_move(state, 1, move_dir, move_boost)
            state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 2, opp_move_dir, opp_move_boost)
            
            if not my_survived and not opp_survived:
                score = 0
            elif not my_survived:
                score = -999999
            elif not opp_survived:
                score = 999999
            else:
                score = minimax_search(state_after_opp_move, MAX_SEARCH_DEPTH - 1, True, my_id, -np.inf, np.inf)
            
            worst_case_score = min(worst_case_score, score)
            
            if worst_case_score > max_score:
                break
        
        if worst_case_score > max_score:
            max_score = worst_case_score
            best_move = (move_dir, move_boost)
        
        if time.time() * 1000 > START_TIME + MAX_TIME_MS:
            break
    
    final_dir, final_boost = best_move
    
    move_str = final_dir.name
    if final_boost:
        move_str += ":BOOST"
    
    print(f"Turn {state.get('turn_count', 0)}: Challenger chose {move_str} with score {max_score:.1f}")
    return move_str


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    print("Strategy: Aggressive Minimax with trapping focus")
    app.run(host="0.0.0.0", port=port, debug=False)
