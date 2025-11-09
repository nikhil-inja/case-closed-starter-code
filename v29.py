import os
import time
import numpy as np
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque, defaultdict
from enum import Enum
import sys

sys.setrecursionlimit(2500)

class Direction(Enum):
    UP = (0, -1); DOWN = (0, 1); RIGHT = (1, 0); LEFT = (-1, 0)

DIRECTION_MAP = {d.name: d for d in Direction}
OPPOSITE_DIR = {
    Direction.UP: Direction.DOWN, Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT, Direction.RIGHT: Direction.LEFT
}

GRID_HEIGHT = 18; GRID_WIDTH = 20
MAX_SEARCH_DEPTH = 50
MAX_TIME_MS = 3750  # Tighter buffer
WIN_SCORE = 999999
LOSE_SCORE = -999999
DRAW_SCORE = 0

# === BITBOARD ===
class BitBoard:
    __slots__ = ['occupied']
    
    def __init__(self, occupied=0):
        self.occupied = occupied
    
    def set(self, x, y):
        bit = y * GRID_WIDTH + x
        self.occupied |= (1 << bit)
    
    def is_occupied(self, x, y):
        bit = y * GRID_WIDTH + x
        return (self.occupied >> bit) & 1
    
    def copy(self):
        return BitBoard(self.occupied)

# === ADJACENCY MASKS ===
ADJ_MASKS = [0] * 360
for i in range(360):
    x, y = i % GRID_WIDTH, i // GRID_WIDTH
    mask = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = (x + dx) % GRID_WIDTH, (y + dy) % GRID_HEIGHT
        mask |= (1 << (ny * GRID_WIDTH + nx))
    ADJ_MASKS[i] = mask

# === ZOBRIST WITH HEADS ===
np.random.seed(42)
ZOBRIST_POS = np.random.randint(1, 2**63 - 1, 360, dtype=np.uint64)
ZOBRIST_HEAD = np.random.randint(1, 2**63 - 1, (2, 360), dtype=np.uint64)
ZOBRIST_BOOST = np.random.randint(1, 2**63 - 1, (2, 4), dtype=np.uint64)

def compute_hash(bitboard, my_head, opp_head, boost1, boost2):
    """Hash with heads included"""
    h = np.uint64(0)
    temp = bitboard.occupied
    while temp:
        lsb = temp & -temp
        bit_pos = lsb.bit_length() - 1
        temp ^= lsb
        h ^= ZOBRIST_POS[bit_pos]
    
    # Include heads in hash
    h ^= ZOBRIST_HEAD[0, my_head[1] * GRID_WIDTH + my_head[0]]
    h ^= ZOBRIST_HEAD[1, opp_head[1] * GRID_WIDTH + opp_head[0]]
    h ^= ZOBRIST_BOOST[0, min(3, boost1)]
    h ^= ZOBRIST_BOOST[1, min(3, boost2)]
    return h

# === VORONOI (NO DOUBLE-COUNTING) ===
def voronoi_territory(bitboard, my_head, opp_head):
    """Voronoi that doesn't double-count shared regions"""
    my_bit = my_head[1] * GRID_WIDTH + my_head[0]
    opp_bit = opp_head[1] * GRID_WIDTH + opp_head[0]
    
    my_territory = 1 << my_bit
    opp_territory = 1 << opp_bit
    my_frontier = 1 << my_bit
    opp_frontier = 1 << opp_bit
    visited = (1 << my_bit) | (1 << opp_bit)
    contested = 0  # Shared cells
    
    for _ in range(25):
        if not my_frontier and not opp_frontier:
            break
        
        new_my = 0
        new_opp = 0
        
        # Expand my territory
        if my_frontier:
            temp = my_frontier
            while temp:
                lsb = temp & -temp
                bit_pos = lsb.bit_length() - 1
                temp ^= lsb
                available = ADJ_MASKS[bit_pos] & ~bitboard.occupied & ~visited
                new_my |= available
                visited |= available
        
        # Expand opp territory
        if opp_frontier:
            temp = opp_frontier
            while temp:
                lsb = temp & -temp
                bit_pos = lsb.bit_length() - 1
                temp ^= lsb
                available = ADJ_MASKS[bit_pos] & ~bitboard.occupied & ~visited
                new_opp |= available
                visited |= available
        
        # Find contested cells (reached by both in same iteration)
        overlap = new_my & new_opp
        contested |= overlap
        
        # Remove contested from both
        new_my &= ~overlap
        new_opp &= ~overlap
        
        my_territory |= new_my
        opp_territory |= new_opp
        my_frontier = new_my
        opp_frontier = new_opp
    
    my_count = my_territory.bit_count()
    opp_count = opp_territory.bit_count()
    contested_count = contested.bit_count()
    
    return my_count, opp_count, contested_count

# === ARTICULATION POINT DETECTION (SIMPLIFIED) ===
def find_chokepoints(bitboard, head):
    """Find critical cells (removing them disconnects large regions)"""
    chokepoints = []
    head_bit = head[1] * GRID_WIDTH + head[0]
    
    # Original reachable area
    original_reach = flood_fill_bitboard(bitboard, head[0], head[1])
    
    # Test each neighbor
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = (head[0] + dx) % GRID_WIDTH, (head[1] + dy) % GRID_HEIGHT
        if not bitboard.is_occupied(nx, ny):
            # Temporarily block this cell
            test_bb = bitboard.copy()
            test_bb.set(nx, ny)
            
            # Check if reachability drops significantly
            new_reach = flood_fill_bitboard(test_bb, head[0], head[1])
            if new_reach < original_reach * 0.5:  # Lost 50%+ of space
                chokepoints.append((nx, ny))
    
    return len(chokepoints)

def flood_fill_bitboard(bitboard, x, y):
    """Fast flood fill"""
    bit = y * GRID_WIDTH + x
    if (bitboard.occupied >> bit) & 1:
        return 0
    
    visited = 1 << bit
    frontier = 1 << bit
    count = 1
    
    while frontier:
        new_frontier = 0
        temp = frontier
        while temp:
            lsb = temp & -temp
            bit_pos = lsb.bit_length() - 1
            temp ^= lsb
            available = ADJ_MASKS[bit_pos] & ~bitboard.occupied & ~visited
            new_frontier |= available
            visited |= available
        count += new_frontier.bit_count()
        frontier = new_frontier
    
    return count

def count_liberties(bitboard, x, y):
    bit = y * GRID_WIDTH + x
    adj_mask = ADJ_MASKS[bit]
    free = adj_mask & ~bitboard.occupied
    return free.bit_count()

# === TT WITH LRU REPLACEMENT ===
class TTEntry:
    __slots__ = ['score', 'age']
    def __init__(self, score, age):
        self.score = score
        self.age = age

TT_CACHE = {}
MAX_CACHE_SIZE = 2_000_000  # 4x larger
TT_AGE = 0

# === GLOBALS ===
START_TIME = 0.0
NODES_SEARCHED = 0
LAST_TIME_CHECK = 0.0

app = Flask(__name__)
GLOBAL_GAME_STATE = {
    "board": None, "agent1_trail": deque(), "agent2_trail": deque(),
    "agent1_boosts": 3, "agent2_boosts": 3, "turn_count": 0,
    "player_number": 1, "my_direction": Direction.RIGHT, "opp_direction": Direction.LEFT
}
game_lock = Lock()

PARTICIPANT = "TOURNAMENT_CHAMPION"
AGENT_NAME = "TournamentChampion_v29"

def _torus(pos):
    return (pos[0] % GRID_WIDTH, pos[1] % GRID_HEIGHT)

class TimeLimitExceeded(Exception):
    pass

def get_final_position(head, direction, use_boost):
    dx, dy = direction.value
    if use_boost:
        nx1, ny1 = _torus((head[0] + dx, head[1] + dy))
        nx2, ny2 = _torus((nx1 + dx, ny1 + dy))
        return (nx2, ny2)
    else:
        return _torus((head[0] + dx, head[1] + dy))

def path_cells(head, direction, steps):
    dx, dy = direction.value
    cells = []
    x, y = head
    for _ in range(steps):
        x, y = _torus((x + dx, y + dy))
        cells.append((x, y))
    return cells

def joint_collision(my_head, opp_head, my_dir, opp_dir, my_boost, opp_boost):
    my_path = path_cells(my_head, my_dir, 2 if my_boost else 1)
    opp_path = path_cells(opp_head, opp_dir, 2 if opp_boost else 1)
    
    if my_path[-1] == opp_path[-1]:
        return "draw"
    
    max_steps = max(len(my_path), len(opp_path))
    for i in range(max_steps):
        a = my_path[i] if i < len(my_path) else my_path[-1]
        b = opp_path[i] if i < len(opp_path) else opp_path[-1]
        if a == b:
            return "draw"
    
    if len(my_path) == len(opp_path) == 1:
        if my_path[0] == opp_head and opp_path[0] == my_head:
            return "draw"
    
    return "continue"

# === ENHANCED EVALUATION ===
def evaluate(bitboard, my_head, opp_head, my_boosts, opp_boosts):
    """Enhanced evaluation with chokepoint detection and Voronoi"""
    my_libs = count_liberties(bitboard, my_head[0], my_head[1])
    opp_libs = count_liberties(bitboard, opp_head[0], opp_head[1])
    
    if my_libs == 0 and opp_libs == 0:
        return DRAW_SCORE
    if my_libs == 0:
        return LOSE_SCORE
    if opp_libs == 0:
        return WIN_SCORE
    
    score = 0
    
    # Mobility
    score += (my_libs - opp_libs) * 2000
    
    # Trapping
    if opp_libs == 1:
        score += 20000
    elif opp_libs == 2:
        score += 8000
    if my_libs == 1:
        score -= 25000
    elif my_libs == 2:
        score -= 12000
    
    # Distance
    dist = abs(my_head[0] - opp_head[0]) + abs(my_head[1] - opp_head[1])
    
    # Use Voronoi (no double-counting)
    my_territory, opp_territory, contested = voronoi_territory(bitboard, my_head, opp_head)
    
    # Weight contested cells less
    effective_my = my_territory + contested * 0.3
    effective_opp = opp_territory + contested * 0.3
    
    territory_weight = 200 if dist > 8 else 80
    score += (effective_my - effective_opp) * territory_weight
    
    # Chokepoint detection
    my_chokes = find_chokepoints(bitboard, my_head)
    opp_chokes = find_chokepoints(bitboard, opp_head)
    
    # Being in a chokepoint is bad
    score -= my_chokes * 3000
    score += opp_chokes * 3000
    
    score += (my_boosts - opp_boosts) * 800
    
    return score

def get_moves(direction, boosts):
    moves = []
    for d in Direction:
        if d != OPPOSITE_DIR.get(direction):
            moves.append((d, False))
            if boosts > 0:
                moves.append((d, True))
    return moves

def order_moves(bitboard, head, direction, boosts, opp_head):
    moves = get_moves(direction, boosts)
    scored = []
    
    for d, b in moves:
        fin = get_final_position(head, d, b)
        
        if bitboard.is_occupied(fin[0], fin[1]) and fin != opp_head:
            continue
        
        libs = count_liberties(bitboard, fin[0], fin[1])
        centerish = -(abs(fin[0] - GRID_WIDTH // 2) + abs(fin[1] - GRID_HEIGHT // 2))
        conserve = 1 if not b else 0
        
        # Penalty for moves toward chokepoints
        test_bb = bitboard.copy()
        test_bb.set(fin[0], fin[1])
        future_reach = flood_fill_bitboard(test_bb, fin[0], fin[1])
        
        score = libs * 100 + centerish * 2 + conserve * 5 + future_reach * 0.5
        scored.append((score, (d, b)))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]

# === MINIMAX WITH TIGHT TIME CONTROL ===
def maximin_value(bitboard, my_head, opp_head, my_dir, opp_dir, my_boosts, opp_boosts, depth, alpha, beta):
    global NODES_SEARCHED, START_TIME, TT_AGE, LAST_TIME_CHECK
    
    NODES_SEARCHED += 1
    
    # TIGHT time check (every 256 nodes)
    if (NODES_SEARCHED & 255) == 0:
        current_time = time.time() * 1000
        if current_time > START_TIME + MAX_TIME_MS:
            raise TimeLimitExceeded()
        LAST_TIME_CHECK = current_time
    
    if depth == 0:
        return evaluate(bitboard, my_head, opp_head, my_boosts, opp_boosts)
    
    # TT with heads
    state_hash = compute_hash(bitboard, my_head, opp_head, my_boosts, opp_boosts)
    cache_key = (state_hash, my_dir, opp_dir, my_boosts, opp_boosts, depth)
    
    if cache_key in TT_CACHE:
        entry = TT_CACHE[cache_key]
        entry.age = TT_AGE
        return entry.score
    
    my_moves = order_moves(bitboard, my_head, my_dir, my_boosts, opp_head)
    opp_moves = order_moves(bitboard, opp_head, opp_dir, opp_boosts, my_head)
    
    if not my_moves:
        return LOSE_SCORE
    if not opp_moves:
        return WIN_SCORE
    
    best_value = -np.inf
    
    for my_move_dir, my_boost in my_moves:
        my_final = get_final_position(my_head, my_move_dir, my_boost)
        worst_case = np.inf
        
        for opp_move_dir, opp_boost in opp_moves:
            opp_final = get_final_position(opp_head, opp_move_dir, opp_boost)
            
            outcome = joint_collision(my_head, opp_head, my_move_dir, opp_move_dir, my_boost, opp_boost)
            if outcome == "draw":
                value = DRAW_SCORE
                worst_case = min(worst_case, value)
                if worst_case <= alpha:
                    break
                continue
            
            mid = path_cells(my_head, my_move_dir, 1)[0] if my_boost else None
            mid2 = path_cells(opp_head, opp_move_dir, 1)[0] if opp_boost else None
            
            if mid2 is not None and my_final == mid2:
                value = LOSE_SCORE
                worst_case = min(worst_case, value)
                if worst_case <= alpha:
                    break
                continue
            
            if mid is not None and opp_final == mid:
                value = WIN_SCORE
                worst_case = min(worst_case, value)
                if worst_case <= alpha:
                    break
                continue
            
            new_bb = bitboard.copy()
            if mid is not None:
                new_bb.set(mid[0], mid[1])
            new_bb.set(my_final[0], my_final[1])
            if mid2 is not None:
                new_bb.set(mid2[0], mid2[1])
            new_bb.set(opp_final[0], opp_final[1])
            
            new_my_boosts = my_boosts - 1 if my_boost else my_boosts
            new_opp_boosts = opp_boosts - 1 if opp_boost else opp_boosts
            
            value = maximin_value(
                new_bb, my_final, opp_final, my_move_dir, opp_move_dir,
                new_my_boosts, new_opp_boosts, depth - 1, alpha, beta
            )
            
            worst_case = min(worst_case, value)
            if worst_case <= alpha:
                break
        
        best_value = max(best_value, worst_case)
        alpha = max(alpha, best_value)
        if alpha >= beta:
            break
    
    # TT with LRU replacement
    if len(TT_CACHE) >= MAX_CACHE_SIZE:
        # Remove oldest entry
        oldest_key = min(TT_CACHE.keys(), key=lambda k: TT_CACHE[k].age)
        del TT_CACHE[oldest_key]
    
    TT_CACHE[cache_key] = TTEntry(best_value, TT_AGE)
    
    return best_value

# === ROOT SEARCH ===
def get_best_move(state):
    global START_TIME, NODES_SEARCHED, TT_CACHE, TT_AGE, LAST_TIME_CHECK
    
    START_TIME = time.time() * 1000
    LAST_TIME_CHECK = START_TIME
    NODES_SEARCHED = 0
    TT_AGE = 0
    TT_CACHE.clear()
    
    bitboard = BitBoard()
    for x, y in state["agent1_trail"]:
        bitboard.set(x, y)
    for x, y in state["agent2_trail"]:
        bitboard.set(x, y)
    
    my_head = state["agent1_trail"][-1]
    opp_head = state["agent2_trail"][-1]
    my_dir = state["my_direction"]
    opp_dir = state["opp_direction"]
    my_boosts = state["agent1_boosts"]
    opp_boosts = state["agent2_boosts"]
    
    valid_moves = order_moves(bitboard, my_head, my_dir, my_boosts, opp_head)
    if not valid_moves:
        return "RIGHT"
    
    best_move = valid_moves[0]
    best_score = -np.inf
    
    try:
        for depth in range(1, MAX_SEARCH_DEPTH + 1):
            TT_AGE += 1
            
            elapsed = time.time() * 1000 - START_TIME
            if elapsed > MAX_TIME_MS - 300:
                break
            
            current_best = best_move
            current_score = -np.inf
            
            for my_move_dir, my_boost in valid_moves:
                my_final = get_final_position(my_head, my_move_dir, my_boost)
                opp_moves = order_moves(bitboard, opp_head, opp_dir, opp_boosts, my_head)
                worst_case = np.inf
                
                for opp_move_dir, opp_boost in opp_moves:
                    opp_final = get_final_position(opp_head, opp_move_dir, opp_boost)
                    
                    outcome = joint_collision(my_head, opp_head, my_move_dir, opp_move_dir, my_boost, opp_boost)
                    if outcome == "draw":
                        value = DRAW_SCORE
                        worst_case = min(worst_case, value)
                        continue
                    
                    mid = path_cells(my_head, my_move_dir, 1)[0] if my_boost else None
                    mid2 = path_cells(opp_head, opp_move_dir, 1)[0] if opp_boost else None
                    
                    if mid2 is not None and my_final == mid2:
                        value = LOSE_SCORE
                        worst_case = min(worst_case, value)
                        continue
                    
                    if mid is not None and opp_final == mid:
                        value = WIN_SCORE
                        worst_case = min(worst_case, value)
                        continue
                    
                    new_bb = bitboard.copy()
                    if mid is not None:
                        new_bb.set(mid[0], mid[1])
                    new_bb.set(my_final[0], my_final[1])
                    if mid2 is not None:
                        new_bb.set(mid2[0], mid2[1])
                    new_bb.set(opp_final[0], opp_final[1])
                    
                    new_my_boosts = my_boosts - 1 if my_boost else my_boosts
                    new_opp_boosts = opp_boosts - 1 if opp_boost else opp_boosts
                    
                    value = maximin_value(
                        new_bb, my_final, opp_final, my_move_dir, opp_move_dir,
                        new_my_boosts, new_opp_boosts, depth - 1, -np.inf, np.inf
                    )
                    
                    worst_case = min(worst_case, value)
                
                if worst_case > current_score:
                    current_score = worst_case
                    current_best = (my_move_dir, my_boost)
            
            best_move = current_best
            best_score = current_score
            print(f"[D{depth}] {best_move[0].name}{':BOOST' if best_move[1] else ''} = {best_score:.0f} (n={NODES_SEARCHED:,})")
    
    except (TimeLimitExceeded, RecursionError):
        pass
    
    move_dir, use_boost = best_move
    move_str = move_dir.name
    if use_boost:
        move_str += ":BOOST"
    
    print(f"Final: {move_str} (score={best_score})")
    return move_str

# === FLASK ===
def get_current_direction(trail, player_id):
    if len(trail) < 2:
        return Direction.RIGHT if player_id == 1 else Direction.LEFT
    head, prev = trail[-1], trail[-2]
    dx, dy = head[0] - prev[0], head[1] - prev[1]
    if abs(dx) > 1: dx = -1 if dx > 0 else 1
    if abs(dy) > 1: dy = -1 if dy > 0 else 1
    if dx == 1: return Direction.RIGHT
    if dx == -1: return Direction.LEFT
    if dy == 1: return Direction.DOWN
    if dy == -1: return Direction.UP
    return Direction.RIGHT if player_id == 1 else Direction.LEFT

def decide_action(current_state, player_number):
    import copy
    state = copy.deepcopy(current_state)
    
    if "agent1_trail" in state:
        state["agent1_trail"] = deque(tuple(p) for p in state["agent1_trail"])
    if "agent2_trail" in state:
        state["agent2_trail"] = deque(tuple(p) for p in state["agent2_trail"])
    
    state["my_direction"] = get_current_direction(state["agent1_trail"], 1)
    state["opp_direction"] = get_current_direction(state["agent2_trail"], 2)
    
    if player_number == 2:
        state["agent1_trail"], state["agent2_trail"] = state["agent2_trail"], state["agent1_trail"]
        state["agent1_boosts"], state["agent2_boosts"] = state["agent2_boosts"], state["agent1_boosts"]
        state["my_direction"], state["opp_direction"] = state["opp_direction"], state["my_direction"]
    
    try:
        return get_best_move(state)
    except Exception as e:
        print(f"[ERROR: {e}]")
        import traceback
        traceback.print_exc()
        return "RIGHT"

def _update_local_game_from_post(data: dict):
    with game_lock:
        GLOBAL_GAME_STATE.update(data)
        if "agent1_trail" in data:
            GLOBAL_GAME_STATE["agent1_trail"] = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME_STATE["agent2_trail"] = deque(tuple(p) for p in data["agent2_trail"])

@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

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
        import copy
        current_state = copy.deepcopy(GLOBAL_GAME_STATE)
    try:
        move = decide_action(current_state, player_number)
        return jsonify({"move": move}), 200
    except Exception as e:
        print(f"[ERROR: {e}]")
        return jsonify({"move": "RIGHT"}), 200

@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json()
    if data:
        result = data.get("result", "UNKNOWN")
        print(f"\nGame Over! Result: {result}")
    with game_lock:
        TT_CACHE.clear()
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"=== {AGENT_NAME} ({PARTICIPANT}) on port {port} ===")
    print(f"✅ Voronoi (no double-count) | ✅ Chokepoint detection | ✅ TT with heads + LRU")
    print(f"✅ Tight time control (256 nodes) | ✅ 2M TT entries | ✅ Enhanced eval")
    app.run(host="0.0.0.0", port=port, debug=False)
