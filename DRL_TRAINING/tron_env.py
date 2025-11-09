# DRL_TRAINING/tron_env.py

import numpy as np
import random
from collections import deque
import time

# --- Core Game Logic ---
from case_closed_game import Game, GameResult #

# --- Import our new Bot "League" ---
from minimax_opponents import get_hybrid_move, get_aggressive_move, get_voronoi_move

# --- Import all constants, enums, and helper functions ---
from drl_utils import (
    GRID_HEIGHT, GRID_WIDTH, EMPTY, AGENT_TRAIL,
    Direction, OPPOSITE_DIR,
    _numba_wrap, _numba_corridor_penalty, _numba_voronoi,
    calculate_center_distance, calculate_opponent_distance
)


# --- The Main Environment Class ---

class TronEnv:
    """
    A DRL-compatible environment for the Case Closed game.
    It wraps the core game logic and provides a gym-like interface.
    """
    def __init__(self, opponent_move_funcs):
        self.game = Game()
        self.opponent_move_funcs = opponent_move_funcs # List of functions
        self.opponent_func = random.choice(self.opponent_move_funcs)

        # 4 directions + 4 boost directions = 8 possible actions
        self.action_space_n = 8 
        
        # This is our Heuristic State Vector
        self.state_space_n = 12 # 12 features
        
        self.action_map = {
            0: (Direction.UP, False),
            1: (Direction.DOWN, False),
            2: (Direction.LEFT, False),
            3: (Direction.RIGHT, False),
            4: (Direction.UP, True),
            5: (Direction.DOWN, True),
            6: (Direction.LEFT, True),
            7: (Direction.RIGHT, True),
        }
        self.OPPOSITE_DIR_ENUM = {
            Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
        }

    def reset(self):
        """Resets the game and returns the initial state vector."""
        self.game.reset()
        # Select a new opponent for this episode
        self.opponent_func = random.choice(self.opponent_move_funcs)
        return self._get_state_vector(player_num=1)

    def _get_state_vector(self, player_num):
        """
        Calculates the 12-feature Heuristic State Vector.
        This is the "state" our DRL agent will see.
        """
        if player_num == 1:
            my_agent, opp_agent = self.game.agent1, self.game.agent2
        else:
            my_agent, opp_agent = self.game.agent2, self.game.agent1
            
        my_head = my_agent.get_trail_positions()[-1]
        opp_head = opp_agent.get_trail_positions()[-1]
        
        # Convert board to numpy for Numba
        board_np = np.array(self.game.board.grid, dtype=np.int8)
        
        # 1. Corridor Penalties
        my_corridor = _numba_corridor_penalty(board_np, my_head[0], my_head[1], GRID_WIDTH, GRID_HEIGHT)
        opp_corridor = _numba_corridor_penalty(board_np, opp_head[0], opp_head[1], GRID_WIDTH, GRID_HEIGHT)
        
        # 2. Voronoi Areas
        my_area, opp_area = _numba_voronoi(board_np, my_head, opp_head, GRID_WIDTH, GRID_HEIGHT)
        
        # 3. Distances
        my_center_dist = calculate_center_distance(my_head)
        opp_center_dist = calculate_center_distance(opp_head)
        opp_dist = calculate_opponent_distance(my_head, opp_head)
        
        # Normalize features to be roughly between -1 and 1 (helps training)
        state_vector = [
            my_agent.length / 100.0,
            opp_agent.length / 100.0,
            my_agent.boosts_remaining / 3.0,
            opp_agent.boosts_remaining / 3.0,
            self.game.turns / 200.0,
            my_corridor / 10.0,
            opp_corridor / 10.0,
            (my_area - opp_area) / 360.0,
            my_center_dist / 18.0,
            opp_center_dist / 18.0,
            opp_dist / 36.0,
            1.0 # Bias term
        ]
        
        return np.array(state_vector, dtype=np.float32)

    def _get_opponent_move(self):
        """
        Gets a move from the current opponent_func.
        """
        # Create the state object the Minimax agent expects
        # We are Player 1 (agent1), so the opponent is Player 2 (agent2)
        # The Minimax funcs expect a state normalized for *them*
        # So we swap agent1 and agent2 data
        state_dict = {
            "board": np.array(self.game.board.grid, dtype=np.int8),
            "agent1_trail": deque(self.game.agent2.get_trail_positions()), # Opponent is P1
            "agent2_trail": deque(self.game.agent1.get_trail_positions()), # We are P2
            "agent1_boosts": self.game.agent2.boosts_remaining,
            "agent2_boosts": self.game.agent1.boosts_remaining,
            "turn_count": self.game.turns,
            "my_direction": self.game.agent2.direction,
            "opp_direction": self.game.agent1.direction,
        }
        
        # We pass player_id=1 to the bot, since its state is
        # already normalized as if it were player 1.
        return self.opponent_func(state_dict, my_id=1)

    def step(self, action_idx):
        """
        Runs one step of the game.
        Returns: (next_state, reward, done, info)
        """
        
        # 1. Get DRL agent's move
        my_dir, my_boost = self.action_map[action_idx]
        
        # 2. Check for invalid move (180-degree turn)
        if my_dir == self.OPPOSITE_DIR_ENUM.get(self.game.agent1.direction):
            # Agent tried an illegal move.
            reward = -1.0 # Severe, immediate punishment
            done = True
            next_state = self._get_state_vector(player_num=1)
            return next_state, reward, done, {"result": GameResult.AGENT2_WIN}

        # 3. Get opponent's move
        opp_dir, opp_boost = self._get_opponent_move()
        
        # 4. Step the game
        result = self.game.step(my_dir, opp_dir, my_boost, opp_boost)
        
        # 5. Calculate reward
        done = False
        reward = 0.001 # Small "time-alive" bonus for every turn survived
        
        if result is not None:
            done = True
            if result == GameResult.AGENT1_WIN:
                reward = 1.0
            elif result == GameResult.AGENT2_WIN:
                reward = -1.0
            elif result == GameResult.DRAW:
                reward = 0.0
                
        # 6. Get next state
        next_state = self._get_state_vector(player_num=1)
        
        return next_state, reward, done, {"result": result}
        
    def close(self):
        """Placeholder for API compatibility."""
        pass