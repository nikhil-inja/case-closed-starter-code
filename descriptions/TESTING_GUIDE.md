# Testing Guide: Enhanced Agent vs Greedy Agent

## What Was Done

### 1. Created New `sample_agent.py` - Greedy Center-Seeker

**Strategy:**
- **Early game (turns 0-20)**: Aggressively moves toward center
- **Mid game (turns 20-80)**: Balances center positioning with space maximization
- **Late game (turns 80+)**: Applies pressure, moves closer to opponent

**Key Features:**
- Greedy algorithm (no minimax, immediate evaluation only)
- Collision avoidance (checks if moves are safe)
- Look-ahead (simple 3-4 step future space calculation)
- Center-seeking behavior (strong preference for center)
- Strategic boost usage (mid-game when space is available)

**Why This Tests Your Agent Better:**
- ✅ Actually moves toward center (forces your agent to respond)
- ✅ Applies pressure mid-game (tests defensive capabilities)
- ✅ Uses different strategy (not symmetric)
- ✅ Simple enough to be predictable but smart enough to be challenging

### 2. Fixed Critical Bug in `agent.py`

**The Bug:** `simulate_move()` wasn't incrementing `turn_count`, causing all simulated future positions to use the same game phase logic.

**The Fix:** Added one line:
```python
if "turn_count" in new_state:
    new_state["turn_count"] += 1
```

**Impact:** Now simulated positions correctly use future turn counts:
- Root (turn 10): Uses early game logic
- 1 move ahead: Uses turn 11 logic
- 6 moves ahead: Uses turn 16 logic

This makes your phase-aware evaluation function work correctly in the minimax search tree!

## How to Test

### Setup

```bash
# Terminal 1: Start your enhanced minimax agent
python agent.py

# Terminal 2: Start the greedy agent
python sample_agent.py

# Terminal 3: Run the fast judge
python judge_engine_fast.py
```

### What to Look For

#### 1. Different Movement Patterns
- **Greedy agent** should move toward center immediately (turn 1-5)
- **Minimax agent** should also move toward center but more strategically
- **Both agents** should NOT move in parallel

#### 2. Debug Output
Watch for debug messages every 10 turns:
```
Turn 0: Chosen move RIGHT with estimated score 45.23
  DEBUG - Space:0.0, Center:150.0, Pressure:30.0, Escape:100.0, Trap:0.0
Turn 10: Chosen move DOWN with estimated score 52.67
  DEBUG - Space:-2.0, Center:140.0, Pressure:45.0, Escape:100.0, Trap:4.0
```

**Check that:**
- Scores are DIFFERENT for different moves (not all ~0)
- Center scores decrease as agents get closer to center
- Pressure scores increase as game progresses
- Trapping scores increase when one agent gains advantage

#### 3. Strategic Gameplay
- **Early game (0-30)**: Both agents should move toward center from different angles
- **Mid game (30-100)**: Agents should apply pressure, try to cut off space
- **Late game (100+)**: Active trapping attempts, space control battles

#### 4. Game String Analysis
The game string should show:
- ✅ Mix of directions: U, D, L, R (not just U-U-U-U)
- ✅ Different patterns for Agent 1 vs Agent 2
- ✅ Boost usage (B markers)
- ✅ Strategic sequences (e.g., R-R-D-R for diagonal movement)

## Expected Outcomes

### Good Signs (Enhanced Agent Working)

1. **Varied Movement**: Game string shows mix of U, D, L, R
2. **Center Control**: Both agents move toward center early game
3. **Pressure Application**: Agents get closer in mid-game
4. **Trapping Attempts**: One agent tries to cut off the other
5. **Competitive Game**: Game lasts 50-150 turns (not 20 turns)
6. **Score Differentiation**: Debug output shows different scores for different moves

### Bad Signs (Still Has Issues)

1. **Parallel Movement**: Both agents move same direction (U-U-U-U)
2. **Zero Scores**: Debug output shows all moves ~0
3. **Early Crash**: Game ends in <30 turns
4. **No Center Movement**: Agents stay near edges
5. **Identical Patterns**: Both agents make symmetric moves

## Comparing Strategies

### Greedy Agent (sample_agent.py)
- **Pros**: Fast, predictable, aggressive center-seeking
- **Cons**: No lookahead beyond 3-4 steps, can be trapped easily
- **Expected**: Should lose to minimax agent most of the time

### Minimax Agent (agent.py)
- **Pros**: 6-move lookahead, strategic evaluation, adaptive strategy
- **Cons**: Slower (but with fast judge, should be manageable)
- **Expected**: Should win by trapping greedy agent or controlling more space

## Troubleshooting

### If agents still move in parallel:
1. Check that the bug fix is applied (turn_count increments)
2. Verify debug output shows varying scores
3. Try increasing center positioning weight in agent.py (change 0.20 to 0.30)
4. Add more asymmetry to player-specific preferences

### If minimax agent loses consistently:
1. Check if evaluation weights are balanced
2. Verify minimax depth (should be 6)
3. Check if trapping bonus is working (should see in debug)
4. May need to adjust pressure score optimal distances

### If game is too slow:
1. Reduce MAX_SEARCH_DEPTH from 6 to 5
2. Use judge_engine_fast.py (parallel requests)
3. Check if MAX_TIME_MS is appropriate (3800ms)

## Success Criteria

Your enhanced agent should:
- ✅ Win 70-80% of games against greedy agent
- ✅ Show strategic positioning (not random)
- ✅ Demonstrate trapping behavior
- ✅ Use boosts strategically
- ✅ Adapt strategy by game phase
- ✅ Complete games in 50-150 turns typically

If these criteria are met, your enhanced evaluation function is working correctly!

## Next Steps After Testing

If the enhanced agent works well:
1. **Phase 2**: Add transposition table (DP optimization) for depth 7-8
2. **Phase 2**: Add better move ordering for improved pruning
3. **Phase 2**: Add iterative deepening for consistent performance
4. **Tuning**: Adjust evaluation weights based on test results

Good luck testing! 🎮🚀

