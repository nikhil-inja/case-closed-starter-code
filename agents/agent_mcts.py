import os
import time
import copy
import math
import random
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque
import numpy as np
from enum import Enum
import sys

# --- Import the actual game engine ---
# CRITICAL: This file requires 'case_closed_game.py' to be in the same directory.
try:
    from case_closed_game import Game, Direction, GameResult
except ImportError:
    print("FATAL ERROR: 'case_closed_game.py' not found.")
    print("Please copy 'case_closed_game.py' into the same directory as 'agent_mcts.py'")
    sys.exit(1)


# Set a higher recursion limit for deep search
sys.setrecursionlimit(2000)

# --- Core Game Constants ---
GRID_HEIGHT = 18
GRID_WIDTH = 20
EMPTY = 0
AGENT_TRAIL = 1
MAX_TIME_MS = 3800  # 3.8 seconds max per move

# Reverse map for movement simulation
OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}

# --- Flask API server setup ---
app = Flask(__name__)
game_lock = Lock()
START_TIME = 0.0

# Agent identity
PARTICIPANT = "GeminiAI_Agent"
AGENT_NAME = "CaseClosed_MCTS"

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

# --- MCTS Node Class ---
class MCTSNode:
    def __init__(self, game_state: Game, parent=None, move=None):
        """
        Initializes a node in the MCTS tree.
        :param game_state: A 'Game' object from case_closed_game.py
        :param parent: The parent MCTSNode
        :param move: The (direction, boost) tuple that led to this state
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move  # The move that *led* to this node
        
        self.children = []
        self.wins = 0
        self.visits = 0
        
        # Get all legal moves for *us* (Agent 1 in this state)
        self.untried_moves = self._get_legal_moves(self.game_state.agent1)
        random.shuffle(self.untried_moves)

    def _get_legal_moves(self, agent):
        """Helper to get all valid (non-180) moves for an agent."""
        moves = []
        for direction in Direction:
            if direction != OPPOSITE_DIR.get(agent.direction):
                moves.append((direction, False)) # No boost
                if agent.boosts_remaining > 0:
                    moves.append((direction, True)) # With boost
        return moves

    def select_child(self) -> 'MCTSNode':
        """
        Selects the best child node using the UCT (UCB1) formula.
        This balances exploitation (high win rate) and exploration (low visits).
        """
        # UCT = (wins/visits) + C * sqrt(log(parent_visits) / visits)
        C_param = 1.41  # A common value for the exploration constant (sqrt(2))
        best_score = -float('inf')
        best_child = None
        
        for child in self.children:
            if child.visits == 0:
                # If a child has not been visited, it's the best choice
                return child
            
            exploit_score = child.wins / child.visits
            explore_score = C_param * math.sqrt(math.log(self.visits) / child.visits)
            score = exploit_score + explore_score
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child

    def expand(self) -> 'MCTSNode':
        """
        Creates one new child node from an untried move.
        This simulates *our* move and a *random opponent move*.
        """
        # 1. Pop one of *our* untried moves
        my_move_dir, my_move_boost = self.untried_moves.pop()
        
        # 2. Get a random legal move for the opponent
        opp_legal_moves = self._get_legal_moves(self.game_state.agent2)
        if not opp_legal_moves:
            # Opponent has no moves, this is a winning path
            opp_move_dir, opp_move_boost = Direction.UP, False # Doesn't matter
        else:
            opp_move_dir, opp_move_boost = random.choice(opp_legal_moves)
        
        # 3. Simulate the full step
        new_game_state = copy.deepcopy(self.game_state)
        new_game_state.step(my_move_dir, opp_move_dir, my_move_boost, opp_move_boost)
        
        # 4. Create the new child node
        child = MCTSNode(new_game_state, parent=self, move=(my_move_dir, my_move_boost))
        self.children.append(child)
        return child

    def simulate_rollout(self) -> int:
        """
        Simulates a random game (rollout) from the current state.
        Returns +1 for an Agent 1 (our) win, -1 for a loss, 0 for a draw.
        """
        sim_game = copy.deepcopy(self.game_state)
        
        # Run for a max of 150 more turns or until game ends
        for _ in range(150): 
            
            # Get random legal moves for both players
            my_legal_moves = self._get_legal_moves(sim_game.agent1)
            opp_legal_moves = self._get_legal_moves(sim_game.agent2)

            # If no moves, use a default (agent will die)
            my_dir = random.choice(my_legal_moves)[0] if my_legal_moves else Direction.UP
            opp_dir = random.choice(opp_legal_moves)[0] if opp_legal_moves else Direction.UP
            
            # Simulate the step (no boosts in rollouts for speed)
            result = sim_game.step(my_dir, opp_dir, False, False)
            
            if result is not None:
                # Game has ended
                if result == GameResult.AGENT1_WIN:
                    return 1  # We won
                elif result == GameResult.AGENT2_WIN:
                    return -1 # We lost
                elif result == GameResult.DRAW:
                    return 0  # Draw
        
        # If the game timed out (150 steps), call it a draw
        return 0

    def backpropagate(self, result: int):
        """Updates wins and visits all the way up the tree."""
        self.visits += 1
        self.wins += result
        if self.parent:
            # We add -result because the parent is the *other* player's turn
            # (In simultaneous-move MCTS, we just propagate the same result up)
            self.parent.backpropagate(result)

# --- Main Agent Logic ---

def get_best_move(state, my_id):
    """
    Main function to initiate the MCTS search.
    """
    global START_TIME
    START_TIME = time.time() * 1000 # Time in milliseconds
    
    # 1. Reconstruct the Game object from the state dict
    # This is the most critical step for MCTS
    root_game = Game()
    
    # After normalization in 'send_move', we are always Agent 1
    root_game.board.grid = np.array(state["board"]).tolist() # Convert numpy back to list
    
    root_game.agent1.trail = deque(tuple(p) for p in state["agent1_trail"])
    root_game.agent1.boosts_remaining = state["agent1_boosts"]
    root_game.agent1.direction = state["my_direction"]
    root_game.agent1.length = len(root_game.agent1.trail) # Recalculate length
    root_game.agent1.alive = True # Assume we are alive if move is requested
    
    root_game.agent2.trail = deque(tuple(p) for p in state["agent2_trail"])
    root_game.agent2.boosts_remaining = state["agent2_boosts"]
    root_game.agent2.direction = state["opp_direction"]
    root_game.agent2.length = len(root_game.agent2.trail)
    root_game.agent2.alive = True
    
    root_game.turns = state["turn_count"]

    # 2. Create the root node
    root_node = MCTSNode(game_state=root_game)

    # 3. Run MCTS loop until time is up
    # We use 3.8s, matching agent_dynamic.py
    time_limit_seconds = MAX_TIME_MS / 1000.0
    start_time_seconds = time.time()
    num_sims = 0

    while (time.time() - start_time_seconds) < time_limit_seconds:
        
        # 1. Selection
        node = root_node
        while node.untried_moves == [] and node.children != []:
            node = node.select_child()
        
        # 2. Expansion
        if node.untried_moves != []:
            node = node.expand()
        
        # 3. Simulation
        result = node.simulate_rollout()
        
        # 4. Backpropagation
        node.backpropagate(result)
        num_sims += 1

    # 4. Select the best move
    # Choose the child with the most *visits*, as it's the most robust choice.
    if not root_node.children:
        # We had no time to even expand, or no legal moves
        return state["my_direction"].name # Failsafe
        
    best_move_node = max(root_node.children, key=lambda c: c.visits)
    
    final_dir, final_boost = best_move_node.move
    
    move_str = final_dir.name
    if final_boost:
        move_str += ":BOOST"
    
    print(f"Turn {state.get('turn_count', 0)}: MCTS ran {num_sims} simulations.")
    print(f"Chose move {move_str} (Visits: {best_move_node.visits}, WinRate: {best_move_node.wins/best_move_node.visits:.2f})")
    return move_str


# --- Flask Endpoints (Boilerplate) ---

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

    move = get_best_move(current_state, player_number)
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