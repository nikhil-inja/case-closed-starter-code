# Critical Bug Fix: Variable Shadowing in agent.py

## The Bug That Caused Self-Collision

### Problem
Agent 1 crashed into itself on turn 18 because of a **variable shadowing bug** in the `get_best_move()` function.

### Root Cause

In lines 297-305, the code sets `my_dir` to the current direction:
```python
if my_id == 1:
    my_dir = state["my_direction"]  # Current direction
    ...
```

But then at line 317, the loop **overwrites** `my_dir`:
```python
for my_dir, my_boost in my_moves:  # BUG: my_dir is reused as loop variable!
```

And again at line 324:
```python
for opp_dir, opp_boost in opp_moves:  # BUG: opp_dir is reused as loop variable!
```

### The Impact

After these loops execute, `my_dir` and `opp_dir` no longer contain the original current directions - they contain whatever the last value was from the loop iteration.

**Example:**
```python
my_dir = Direction.UP  # Current direction

for my_dir, my_boost in my_moves:  # my_dir is UP, DOWN, LEFT, RIGHT in sequence
    # ... loop body ...

# After loop: my_dir = Direction.RIGHT (last value from loop)
# Lost the original Direction.UP!
```

This caused:
1. **Invalid move selection**: The agent might think its current direction is RIGHT when it's actually UP
2. **Unpredictable behavior**: Best move calculation uses wrong reference direction
3. **Self-collision**: Agent could turn back into itself or make illegal moves

### The Fix

Changed loop variable names to avoid shadowing:

**Before (BUGGY):**
```python
for my_dir, my_boost in my_moves:
    for opp_dir, opp_boost in opp_moves:
        state_after_my_move, my_survived = simulate_move(state, my_id, my_dir, my_boost)
        state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 3 - my_id, opp_dir, opp_boost)
        # ...
        best_move = (my_dir, my_boost)
```

**After (FIXED):**
```python
for move_dir, move_boost in my_moves:
    for opp_move_dir, opp_move_boost in opp_moves:
        state_after_my_move, my_survived = simulate_move(state, my_id, move_dir, move_boost)
        state_after_opp_move, opp_survived = simulate_move(state_after_my_move, 3 - my_id, opp_move_dir, opp_move_boost)
        # ...
        best_move = (move_dir, move_boost)
```

Now the original `my_dir` and `opp_dir` values are preserved throughout the function.

## Additional Enhancement: Colored Board Display

### What Was Added

Modified `judge_engine_fast.py` to display colored trails:
- **RED (A)**: Agent 1 trail
- **BLUE (B)**: Agent 2 (sample_agent) trail
- **White (.)**: Empty space

### Implementation

Added ANSI color codes:
```python
class Colors:
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
```

Added display method to Judge class:
```python
def display_colored_board(self):
    """Display board with colored trails for each agent."""
    agent1_positions = set(self.game.agent1.get_trail_positions())
    agent2_positions = set(self.game.agent2.get_trail_positions())
    
    for y in range(board.height):
        for x in range(board.width):
            if (x, y) in agent1_positions:
                print(f"{Colors.RED}{Colors.BOLD}A{Colors.RESET} ", end="")
            elif (x, y) in agent2_positions:
                print(f"{Colors.BLUE}{Colors.BOLD}B{Colors.RESET} ", end="")
            else:
                print(". ", end="")
        print()
```

### Benefits

1. **Easier visualization**: Instantly see which agent controls which territory
2. **Better debugging**: Spot collision patterns and movement strategies
3. **Improved testing**: Quickly identify if agents are moving strategically

## Testing Instructions

1. **Start the agents:**
   ```bash
   # Terminal 1: Minimax agent (fixed)
   python agent.py
   
   # Terminal 2: Greedy agent
   python sample_agent.py
   
   # Terminal 3: Fast judge with colored output
   python judge_engine_fast.py
   ```

2. **What to look for:**
   - ✅ No self-collisions from Agent 1
   - ✅ Colored RED and BLUE trails are clearly visible
   - ✅ Strategic movement patterns (not just parallel UP movement)
   - ✅ Debug output shows different scores for different moves
   - ✅ Agent 1 should win most games against the greedy agent

3. **Expected behavior:**
   - Both agents move toward center initially
   - Agents take different paths (not symmetric)
   - Active territorial control and pressure application
   - Games last 50-150 turns typically

## Files Modified

1. **agent.py**:
   - Fixed variable shadowing bug in `get_best_move()` (lines 318, 324, 327, 330, 356)
   - Also had the turn_count increment fix from earlier

2. **judge_engine_fast.py**:
   - Added Colors class for ANSI color codes
   - Added `display_colored_board()` method to Judge class
   - Modified board display to use colored output
   - Added colored agent labels in status printout

## Impact

This fix is **critical** because:
1. **Prevents crashes**: Agent won't collide with itself anymore
2. **Enables strategy**: Agent can now properly evaluate all possible moves
3. **Improves gameplay**: With correct move evaluation, the enhanced evaluation function can work as intended

Combined with the earlier fixes (turn_count increment, enhanced evaluation, symmetry breaking), your minimax agent should now play strategically and win consistently against the greedy agent!

## Success Validation

Run a test game and verify:
- ✅ No "crashed into itself" errors
- ✅ Colored trails display correctly
- ✅ Agent 1 (RED) wins most games
- ✅ Strategic movement patterns visible
- ✅ Game string shows varied directions (not just U-U-U-U)

Good luck! 🎮🚀

