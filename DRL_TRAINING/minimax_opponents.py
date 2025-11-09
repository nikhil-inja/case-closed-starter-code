# DRL_TRAINING/minimax_opponents.py

import numpy as np
import copy
from collections import deque
import time

# Import all constants, enums, and helper functions from our new utils file
from drl_utils import (
    Direction, OPPOSITE_DIR, GRID_WIDTH, GRID_HEIGHT, EMPTY, AGENT_TRAIL, DIRS,
    _torus_check, flood_fill, count_escape_routes, get_possible_moves, simulate_move,
    calculate_center_distance, calculate_opponent_distance,
    _numba_wrap, _numba_corridor_penalty, _numba_voronoi
)

# --- Bot 1: Hybrid Agent (from agent.py) ---

def _hybrid_calculate_center_bonus(my_head, opp_head, my_id, turn_count):
    my_dist = calculate_center_distance(my_head)
    opp_dist = calculate_center_distance(opp_head)
    phase_weight = max(0, 1.0 - turn_count / 100.0)
    if my_id == 1: my_dist -= 0.5
    else: my_dist += 0.5
    return (opp_dist - my_dist) * 2.0 * phase_weight

def _hybrid_calculate_pressure_score(my_head, opp_head, turn_count):
    distance = calculate_opponent_distance(my_head, opp_head)
    if turn_count < 12: return 0
    if 5 <= distance <= 10: return 5.0
    elif distance < 5: return -2.0
    elif distance > 15: return -3.0
    return 0

def _hybrid_calculate_escape_quality(board, my_head, opp_head):
    my_routes = count_escape_routes(board, my_head)
    opp_routes = count_escape_routes(board, opp_head)
    route_advantage = (my_routes - opp_routes) * 4.0
    if opp_routes == 0 and my_routes > 0: route_advantage += 100.0
    elif opp_routes == 1 and my_routes > 2: route_advantage += 50.0
    elif opp_routes <= 2 and my_routes > 3: route_advantage += 25.0
    if my_routes >= 4: route_advantage += 10.0
    elif my_routes == 3: route_advantage += 5.0
    return route_advantage

def _hybrid_detect_opponent_vulnerability(opp_space, total_space):
    if total_space == 0: return 0
    opp_percentage = opp_space / total_space
    if opp_percentage < 0.1: return 50.0
    elif opp_percentage < 0.2: return 25.0
    elif opp_percentage < 0.3: return 10.0
    return 0

def _hybrid_calculate_trapping_bonus(board, my_space, opp_space, my_head, opp_head):
    vulnerability = _hybrid_detect_opponent_vulnerability(opp_space, my_space + opp_space)
    escape_quality = _hybrid_calculate_escape_quality(board, my_head, opp_head)
    return vulnerability + escape_quality

def _hybrid_calculate_openness_bonus(board, my_head, my_space):
    openness_ratio = my_space / (board.shape[0] * board.shape[1])
    if openness_ratio > 0.35: return 15.0
    elif openness_ratio > 0.25: return 10.0
    elif openness_ratio > 0.15: return 0
    elif openness_ratio > 0.08: return -15.0
    else: return -40.0

def _hybrid_get_game_phase(turn_count):
    if turn_count < 20: return "early"
    elif turn_count < 60: return "mid"
    else: return "late"

def evaluate_state_hybrid(state, my_id):
    board = state["board"]
    turn_count = state.get("turn_count", 0)
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    my_space = flood_fill(board, my_head, my_id)
    opp_space = flood_fill(board, opp_head, my_id)
    my_routes = count_escape_routes(board, my_head)
    
    if my_space < 20:
        if my_routes == 0: return -999999
        elif my_routes == 1: return -800000 + my_space * 1000
        elif my_routes == 2: return -50000 + my_space * 500
        else: return my_space * 100 + my_routes * 5000
    
    if my_routes <= 1: return -500000 + my_routes * 10000
    if my_routes == 2 and my_space < 15: return -100000 + my_space * 100
    
    space_score = my_space - opp_space
    center_bonus = _hybrid_calculate_center_bonus(my_head, opp_head, my_id, turn_count)
    pressure_score = _hybrid_calculate_pressure_score(my_head, opp_head, turn_count)
    trapping_bonus = _hybrid_calculate_trapping_bonus(board, my_space, opp_space, my_head, opp_head)
    openness = _hybrid_calculate_openness_bonus(board, my_head, my_space)
    
    phase = _hybrid_get_game_phase(turn_count)
    weights = {
        "early": {"space": 1.0, "center": 0.8, "pressure": 0.2, "trapping": 0.8, "openness": 0.5},
        "mid": {"space": 1.0, "center": 0.2, "pressure": 0.6, "trapping": 2.0, "openness": 0.8},
        "late": {"space": 1.0, "center": 0.1, "pressure": 0.4, "trapping": 3.5, "openness": 1.0}
    }[phase]
    
    final_score = (
        weights["space"] * space_score +
        weights["center"] * center_bonus +
        weights["pressure"] * pressure_score +
        weights["trapping"] * trapping_bonus +
        weights["openness"] * openness
    )
    if opp_space == 0 and my_space > 0: return 999999
    return final_score

def minimax_search_hybrid(state, depth, is_max_turn, my_id, alpha, beta, start_time, time_limit):
    if (time.time() - start_time) > time_limit or depth == 0:
        return evaluate_state_hybrid(state, my_id)

    max_id, min_id = 1, 2
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
                
                if not my_survived and not opp_survived: score = 0
                elif not my_survived: score = -999999
                elif not opp_survived: score = 999999
                else:
                    score = minimax_search_hybrid(state_after_opp_move, depth - 1, False, my_id, alpha, beta, start_time, time_limit)
                
                worst_case_score = min(worst_case_score, score)
                if worst_case_score <= alpha: break
            
            value = max(value, worst_case_score)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value
    else:
        return evaluate_state_hybrid(state, my_id)

def get_hybrid_move(state, my_id):
    start_time = time.time()
    my_dir = state["my_direction"]
    my_boosts = state["agent1_boosts"]
    opp_dir = state["opp_direction"]
    opp_boosts = state["agent2_boosts"]

    my_moves = get_possible_moves(my_dir, my_boosts)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    best_move = (my_dir, False)
    max_score = -np.inf
    
    MAX_SEARCH_DEPTH = 2  # Only looks 1 full turn ahead
    TIME_LIMIT = 0.05     # 50 milliseconds
    
    for move_dir, move_boost in my_moves:
        worst_case_score = np.inf
        for opp_move_dir, opp_move_boost in opp_moves:
            state_after_my_move, my_survived = simulate_move(state, 1, move_dir, move_boost)
            state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 2, opp_move_dir, opp_move_boost)
            
            if not my_survived and not opp_survived: score = 0
            elif not my_survived: score = -999999
            elif not opp_survived: score = 999999
            else:
                score = minimax_search_hybrid(state_after_opp_move, MAX_SEARCH_DEPTH - 1, True, my_id, -np.inf, np.inf, start_time, TIME_LIMIT)
            
            worst_case_score = min(worst_case_score, score)
            if worst_case_score > max_score: break
        
        if worst_case_score > max_score:
            max_score = worst_case_score
            best_move = (move_dir, move_boost)
        
        if time.time() - start_time > TIME_LIMIT:
            break
            
    final_dir, final_boost = best_move
    return final_dir, final_boost


# --- Bot 2: Aggressive Agent (from sample_agent.py) ---

def _aggressive_calculate_aggression_bonus(my_head, opp_head, turn_count):
    distance = calculate_opponent_distance(my_head, opp_head)
    if turn_count < 10: return 0
    elif turn_count < 30:
        if 5 <= distance <= 10: return 12.0
        elif 3 <= distance < 5: return 5.0
        elif distance < 3: return -25.0
        else: return -3.0
    else:
        if 4 <= distance <= 8: return 15.0
        elif distance < 4: return -20.0
        elif distance <= 12: return 3.0
        else: return -8.0

def evaluate_state_aggressive(state, my_id):
    board = state["board"]
    turn_count = state.get("turn_count", 0)
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    my_space = flood_fill(board, my_head, my_id)
    opp_space = flood_fill(board, opp_head, my_id)
    my_routes = count_escape_routes(board, my_head)

    if my_space < 20:
        if my_routes == 0: return -999999
        elif my_routes == 1: return -800000 + my_space * 1000
        elif my_routes == 2: return -50000 + my_space * 500
        else: return my_space * 100 + my_routes * 5000
    
    if my_routes <= 1: return -500000 + my_routes * 10000
    if my_routes == 2 and my_space < 15: return -100000 + my_space * 100

    space_score = my_space - opp_space
    aggression = _aggressive_calculate_aggression_bonus(my_head, opp_head, turn_count)
    route_advantage = _hybrid_calculate_escape_quality(board, my_head, opp_head) # Can reuse this
    openness = _hybrid_calculate_openness_bonus(board, my_head, my_space) # Can reuse this
    
    if turn_count < 15:
        weights = {"space": 1.0, "aggression": 0.4, "routes": 1.5}
    elif turn_count < 50:
        weights = {"space": 1.0, "aggression": 0.7, "routes": 2.5}
    else:
        weights = {"space": 0.9, "aggression": 0.4, "routes": 3.0}

    final_score = (
        weights["space"] * space_score +
        weights["aggression"] * aggression +
        weights["routes"] * route_advantage +
        openness # Always add openness
    )
    if opp_space == 0 and my_space > 0: return 999999
    return final_score

def minimax_search_aggressive(state, depth, is_max_turn, my_id, alpha, beta, start_time, time_limit):
    if (time.time() - start_time) > time_limit or depth == 0:
        return evaluate_state_aggressive(state, my_id)
    
    max_id, min_id = 1, 2
    max_dir, min_dir = state["my_direction"], state["opp_direction"]
    max_boosts, min_boosts = state["agent1_boosts"], state["agent2_boosts"]

    if is_max_turn:
        value = -np.inf
        max_moves = get_possible_moves(max_dir, max_boosts)
        min_moves = get_possible_moves(min_dir, min_boosts)
        for my_dir, my_boost in max_moves:
            worst_case_score = np.inf
            for opp_dir, opp_boost in min_moves:
                state_after_my_move, my_survived = simulate_move(state, max_id, my_dir, my_boost)
                state_after_opp_move, opp_survived = simulate_move(state_after_my_move, min_id, opp_dir, opp_boost)
                if not my_survived and not opp_survived: score = 0
                elif not my_survived: score = -999999
                elif not opp_survived: score = 999999
                else:
                    score = minimax_search_aggressive(state_after_opp_move, depth - 1, False, my_id, alpha, beta, start_time, time_limit)
                worst_case_score = min(worst_case_score, score)
                if worst_case_score <= alpha: break
            value = max(value, worst_case_score)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value
    else:
        return evaluate_state_aggressive(state, my_id)

def get_aggressive_move(state, my_id):
    start_time = time.time()
    my_dir, my_boosts = state["my_direction"], state["agent1_boosts"]
    opp_dir, opp_boosts = state["opp_direction"], state["agent2_boosts"]

    my_moves = get_possible_moves(my_dir, my_boosts)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    best_move = (my_dir, False)
    max_score = -np.inf
    MAX_SEARCH_DEPTH = 2
    TIME_LIMIT = 0.05
    
    for move_dir, move_boost in my_moves:
        worst_case_score = np.inf
        for opp_move_dir, opp_move_boost in opp_moves:
            state_after_my_move, my_survived = simulate_move(state, 1, move_dir, move_boost)
            state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 2, opp_move_dir, opp_move_boost)
            
            if not my_survived and not opp_survived: score = 0
            elif not my_survived: score = -999999
            elif not opp_survived: score = 999999
            else:
                score = minimax_search_aggressive(state_after_opp_move, MAX_SEARCH_DEPTH - 1, True, my_id, -np.inf, np.inf, start_time, TIME_LIMIT)
            
            worst_case_score = min(worst_case_score, score)
            if worst_case_score > max_score: break
        
        if worst_case_score > max_score:
            max_score = worst_case_score
            best_move = (move_dir, move_boost)
            
        if time.time() - start_time > TIME_LIMIT:
            break
            
    final_dir, final_boost = best_move
    return final_dir, final_boost

# --- Bot 3: Voronoi Agent (from bestsuperdemon.py) ---

def evaluate_state_voronoi(state, my_id):
    board_np = state["board"]
    turn_count = state.get("turn_count", 0)
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    my_len = len(state["agent1_trail"])
    opp_len = len(state["agent2_trail"])
    
    my_moves = count_escape_routes(board_np, my_head)
    opp_moves = count_escape_routes(board_np, opp_head)
    
    if my_moves == 0 and opp_moves == 0: return 0
    if my_moves == 0: return -1e9 + turn_count
    if opp_moves == 0: return 1e9 - turn_count

    # Max turns is 200 in the original game file
    if turn_count > 160: 
        return (my_len - opp_len) * 10000.0

    my_corridor_penalty = _numba_corridor_penalty(board_np, my_head[0], my_head[1], GRID_WIDTH, GRID_HEIGHT) * 10.0 # Scale it up
    opp_corridor_bonus = _numba_corridor_penalty(board_np, opp_head[0], opp_head[1], GRID_WIDTH, GRID_HEIGHT) * -10.0 # Invert
    
    my_area, opp_area = _numba_voronoi(board_np, my_head, opp_head, GRID_WIDTH, GRID_HEIGHT)

    territory_score = (my_area - opp_area) * 100.0
    length_score = (my_len - opp_len) * 10.0
    
    final_score = (
        territory_score 
        + length_score 
        + my_corridor_penalty 
        + opp_corridor_bonus
    )
    return final_score

def minimax_search_voronoi(state, depth, is_max_turn, my_id, alpha, beta, start_time, time_limit):
    if (time.time() - start_time) > time_limit or depth == 0:
        return evaluate_state_voronoi(state, my_id)
    
    max_id, min_id = 1, 2
    max_dir, min_dir = state["my_direction"], state["opp_direction"]
    max_boosts, min_boosts = state["agent1_boosts"], state["agent2_boosts"]

    if is_max_turn:
        value = -np.inf
        max_moves = get_possible_moves(max_dir, max_boosts)
        min_moves = get_possible_moves(min_dir, min_boosts)
        for my_dir, my_boost in max_moves:
            worst_case_score = np.inf
            for opp_dir, opp_boost in min_moves:
                state_after_my_move, my_survived = simulate_move(state, max_id, my_dir, my_boost)
                state_after_opp_move, opp_survived = simulate_move(state_after_my_move, min_id, opp_dir, opp_boost)
                if not my_survived and not opp_survived: score = 0
                elif not my_survived: score = -999999
                elif not opp_survived: score = 999999
                else:
                    score = minimax_search_voronoi(state_after_opp_move, depth - 1, False, my_id, alpha, beta, start_time, time_limit)
                worst_case_score = min(worst_case_score, score)
                if worst_case_score <= alpha: break
            value = max(value, worst_case_score)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value
    else:
        return evaluate_state_voronoi(state, my_id)

def get_voronoi_move(state, my_id):
    start_time = time.time()
    my_dir, my_boosts = state["my_direction"], state["agent1_boosts"]
    opp_dir, opp_boosts = state["opp_direction"], state["agent2_boosts"]

    my_moves = get_possible_moves(my_dir, my_boosts)
    opp_moves = get_possible_moves(opp_dir, opp_boosts)
    
    best_move = (my_dir, False)
    max_score = -np.inf
    MAX_SEARCH_DEPTH = 2
    TIME_LIMIT = 0.05
    
    for move_dir, move_boost in my_moves:
        worst_case_score = np.inf
        for opp_move_dir, opp_move_boost in opp_moves:
            state_after_my_move, my_survived = simulate_move(state, 1, move_dir, move_boost)
            state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 2, opp_move_dir, opp_move_boost)
            
            if not my_survived and not opp_survived: score = 0
            elif not my_survived: score = -999999
            elif not opp_survived: score = 999999
            else:
                score = minimax_search_voronoi(state_after_opp_move, MAX_SEARCH_DEPTH - 1, True, my_id, -np.inf, np.inf, start_time, TIME_LIMIT)
            
            worst_case_score = min(worst_case_score, score)
            if worst_case_score > max_score: break
        
        if worst_case_score > max_score:
            max_score = worst_case_score
            best_move = (move_dir, move_boost)
            
        if time.time() - start_time > TIME_LIMIT:
            break
            
    final_dir, final_boost = best_move
    return final_dir, final_boost