# Critical Flood Fill Bug Fix

## Date: November 8, 2025

## Problem Identified

Both `agent.py` and `opponent.py` contained a **critical bug** in their `flood_fill` function that caused catastrophic evaluation failures, leading to self-collisions and poor strategic play.

## The Bug

**Location:** Lines 75-76 in `agent.py` and `opponent.py`

```python
def flood_fill(board, start_pos, my_id):
    """
    Flood-Fill heuristic: Calculates the size of the open, connected region
    accessible from the start_pos. This is the core evaluation function.
    """
    if board[start_pos[1], start_pos[0]] == AGENT_TRAIL:  # ❌ BUG
        return 0                                            # ❌ BUG
    
    q = deque([start_pos])
    visited = {start_pos}
    count = 0
    # ... rest of function
```

## Why This Was Catastrophic

1. **The agent's head position is ALWAYS on `AGENT_TRAIL`** - it's part of the agent's trail by definition
2. This check caused `flood_fill` to **always return 0** when called with the agent's head position
3. The evaluation function (`evaluate_state`) believed the agent had **0 reachable space**
4. With 0 space:
   - All moves appeared equally bad (score ~0)
   - The agent made essentially random decisions
   - No strategic planning occurred
   - Self-collisions were inevitable

## Impact

- **`agent.py`**: Self-collided when playing against identical logic (`opponent.py`)
- **`opponent.py`**: Self-collided when playing against `agent.py`
- **`sample_agent.py`**: **Won consistently** because this bug was already fixed during earlier debugging

## The Fix

**Remove the buggy check entirely:**

```python
def flood_fill(board, start_pos, my_id):
    """
    Flood-Fill heuristic: Calculates the size of the open, connected region
    accessible from the start_pos. This is the core evaluation function.
    """
    # Buggy check REMOVED ✓
    
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
```

## Why This Works

1. **The starting position is added to `visited`** - prevents infinite loops
2. **The starting position is counted** - gives accurate space evaluation
3. **Only EMPTY cells are explored** - the BFS naturally stops at trails/walls
4. **The agent's head counts as reachable space** - which it is!

## Files Fixed

- ✅ `agent.py` - Lines 75-76 removed
- ✅ `opponent.py` - Lines 75-76 removed
- ✅ `sample_agent.py` - Already correct (fixed during earlier debugging)

## Expected Results After Fix

- **No more self-collisions** from incorrect evaluation
- **Strategic play** based on accurate space calculation
- **Competitive games** between all three agents
- **Proper minimax decision-making** with correct heuristics

## Historical Note

This bug was originally present in all three agents but was fixed in `sample_agent.py` during debugging session when it exhibited self-collision. The fix was never propagated to `agent.py` and `opponent.py`, leading to the asymmetric performance where `sample_agent.py` consistently defeated the other agents.

## Testing Recommendation

Run test games:
1. `agent.py` vs `opponent.py` - Should now be competitive (mirror match)
2. `agent.py` vs `sample_agent.py` - Should now be strategic gameplay from both sides
3. Monitor for self-collisions - Should be rare/non-existent now

---

**This fix represents a fundamental correction to the core evaluation logic that powers all strategic decision-making in the minimax agents.**

