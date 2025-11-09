# agent_drl.py

import os
import time
import copy
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
from enum import Enum
import sys

# --- DRL Imports ---
import torch
import torch.nn as nn
from numba import njit

# --- Core Game Constants ---
class Direction(Enum):
    UP = (0, -1)
    DOWN = (0, 1)
    RIGHT = (1, 0)
    LEFT = (-1, 0)

OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}
GRID_HEIGHT = 18
GRID_WIDTH = 20
EMPTY = 0
AGENT_TRAIL = 1

# --- Flask API server setup ---
app = Flask(__name__)
game_lock = Lock()

# --- Agent identity ---
PARTICIPANT = "GeminiAI_Agent"
AGENT_NAME = "CaseClosed_DRL"

# --- Global State ---
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

# --- Action Mapping (from tron_env.py) ---
ACTION_MAP = {
    0: (Direction.UP, False),
    1: (Direction.DOWN, False),
    2: (Direction.LEFT, False),
    3: (Direction.RIGHT, False),
    4: (Direction.UP, True),
    5: (Direction.DOWN, True),
    6: (Direction.LEFT, True),
    7: (Direction.RIGHT, True),
}

# --- Numba Heuristic Functions (from drl_utils.py) ---
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
    
    if empties == 0: return -10.0
    elif empties == 1: return -5.0
    elif empties == 2: return -0.5
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

# --- Python Helper Functions (from drl_utils.py) ---
def calculate_center_distance(pos):
    return abs(pos[0] - GRID_WIDTH // 2) + abs(pos[1] - GRID_HEIGHT // 2)

def calculate_opponent_distance(my_head, opp_head):
    return abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])

# --- DRL Model Definition (from model.py) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # Simple MLP architecture matching the saved checkpoint
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

# --- MODEL LOADING ---
print("Loading DRL model...")
# CRITICAL: Load model onto CPU as required by the challenge
device = torch.device("cpu")
state_size = 12
action_size = 8
model = QNetwork(state_size, action_size).to(device)

# --- IMPORTANT ---
# Make sure your trained model file is in the SAME directory as this agent
# and is named "tron_dd_dqn_model_ckpt.pth" (or change the name here)
MODEL_PATH = "DRL_TRAINING/tron_dd_dqn_model_ckpt.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"!!! ERROR: FAILED TO LOAD MODEL FROM {MODEL_PATH} !!!")
    print(e)
    # The agent will still run, but will take random actions.
    pass

# --- State Vector Creation (from tron_env.py) ---
def _get_state_vector(state_dict):
    """
    Converts the judge's state dictionary into the 12-feature
    heuristic state vector our DRL model was trained on.
    """
    # State is *already normalized* by send_move, so "agent1" is always us.
    my_head = state_dict["agent1_trail"][-1]
    opp_head = state_dict["agent2_trail"][-1]
    board_np = state_dict["board"]
    
    my_corridor = _numba_corridor_penalty(board_np, my_head[0], my_head[1], GRID_WIDTH, GRID_HEIGHT)
    opp_corridor = _numba_corridor_penalty(board_np, opp_head[0], opp_head[1], GRID_WIDTH, GRID_HEIGHT)
    my_area, opp_area = _numba_voronoi(board_np, my_head, opp_head, GRID_WIDTH, GRID_HEIGHT)
    my_center_dist = calculate_center_distance(my_head)
    opp_center_dist = calculate_center_distance(opp_head)
    opp_dist = calculate_opponent_distance(my_head, opp_head)
    
    state_vector = [
        len(state_dict["agent1_trail"]) / 100.0,
        len(state_dict["agent2_trail"]) / 100.0,
        state_dict["agent1_boosts"] / 3.0,
        state_dict["agent2_boosts"] / 3.0,
        state_dict["turn_count"] / 200.0,
        my_corridor / 10.0,
        opp_corridor / 10.0,
        (my_area - opp_area) / 360.0,
        my_center_dist / 18.0,
        opp_center_dist / 18.0,
        opp_dist / 36.0,
        1.0 # Bias term
    ]
    
    return np.array(state_vector, dtype=np.float32)

# --- NEW Brain of the Agent ---
def get_best_move(state):
    """
    Gets the best move by feeding the current state into the
    loaded DRL model.
    """
    try:
        # 1. Convert state dict to the 12-feature vector
        state_vector = _get_state_vector(state)
        
        # 2. Convert to a PyTorch tensor
        state_tensor = torch.Tensor(state_vector).to(device).unsqueeze(0)
        
        # 3. Get Q-values from the model (inference)
        with torch.no_grad():
            q_values = model(state_tensor)
            
        # 4. Get best action (greedy-only, no epsilon)
        action_idx = q_values.argmax().item()
        
        # 5. Convert action_idx (0-7) to the judge's move string
        move = ACTION_MAP[action_idx]
        final_dir, final_boost = move
        
        # 6. Safety Check: If agent picks 180, override
        if final_dir == OPPOSITE_DIR.get(state["my_direction"]):
            # Model made a mistake, pick any non-180 move
            for d in Direction:
                if d != OPPOSITE_DIR.get(state["my_direction"]):
                    final_dir = d
                    final_boost = False
                    break
        
        move_str = final_dir.name
        if final_boost and state["agent1_boosts"] > 0:
            move_str += ":BOOST"
            
        return move_str
        
    except Exception as e:
        print(f"!!! ERROR IN get_best_move: {e} !!!")
        # Failsafe: return a non-boosted version of the current direction
        return state["my_direction"].name


# --- Flask Endpoints (from agent_dynamic.py) ---

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

def _update_local_game_from_post(data: dict):
    with game_lock:
        GLOBAL_GAME_STATE.update(data)
        
        if "board" in data:
            GLOBAL_GAME_STATE["board"] = np.array(data["board"], dtype=np.int8)

        if "agent1_trail" in data:
            GLOBAL_GAME_STATE["agent1_trail"] = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME_STATE["agent2_trail"] = deque(tuple(p) for p in data["agent2_trail"])

        def get_current_direction(trail):
            if len(trail) < 2: return Direction.RIGHT 
            head = trail[-1]; prev = trail[-2]
            dx, dy = head[0] - prev[0], head[1] - prev[1]
            if abs(dx) > 1: dx = -1 if dx > 0 else 1
            if abs(dy) > 1: dy = -1 if dy > 0 else 1
            if dx == 1: return Direction.RIGHT
            if dx == -1: return Direction.LEFT
            if dy == 1: return Direction.DOWN
            if dy == -1: return Direction.UP
            return Direction.RIGHT

        if len(GLOBAL_GAME_STATE["agent1_trail"]) >= 2:
            GLOBAL_GAME_STATE["my_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent1_trail"])
        if len(GLOBAL_GAME_STATE["agent2_trail"]) >= 2:
            GLOBAL_GAME_STATE["opp_direction"] = get_current_direction(GLOBAL_GAME_STATE["agent2_trail"])

@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        current_state = copy.deepcopy(GLOBAL_GAME_STATE)
        current_state["player_number"] = player_number

        if player_number == 2:
             current_state["agent1_trail"], current_state["agent2_trail"] = current_state["agent2_trail"], current_state["agent1_trail"]
             current_state["agent1_boosts"], current_state["agent2_boosts"] = current_state["agent2_boosts"], current_state["agent1_boosts"]
             current_state["my_direction"], current_state["opp_direction"] = current_state["opp_direction"], current_state["my_direction"]

    move = get_best_move(current_state)
    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
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