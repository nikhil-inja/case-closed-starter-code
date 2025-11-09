# Aggressive Challenger Agent - Testing Guide

## Overview

`sample_agent.py` has been upgraded from a simple greedy agent to a **competitive minimax-based AI** with different strategic priorities than `agent.py`. This creates challenging matchups that can expose weaknesses in `agent.py`'s strategic thinking.

## Strategic Differences

### Agent.py (Defensive Strategist)
- **Philosophy**: Balanced, phase-aware strategy
- **Search Depth**: 6 (deeper search)
- **Early Game**: Center control priority
- **Mid Game**: Balanced space + moderate pressure
- **Late Game**: Trapping focus (3x weight)
- **Aggression Level**: Moderate (optimal distance 5-10 cells)
- **Evaluation**: Multi-factor with gradual weight transitions

### Sample_agent.py (Aggressive Challenger)
- **Philosophy**: Aggressive, high-pressure gameplay
- **Search Depth**: 5 (slightly shallower, faster decisions)
- **Early Game**: Space claiming + immediate aggression (0.5x aggression)
- **Mid Game**: Very aggressive (1.0x aggression, 2.0x dominance)
- **Late Game**: All-in trapping (4.0x dominance bonus!)
- **Aggression Level**: High (optimal distance 3-8 cells, earlier engagement)
- **Evaluation**: Rewards close pursuit and space dominance

## Key Differences That Create Strategic Tension

### 1. Aggression Timing
- **agent.py**: Waits until turn 15 before applying pressure
- **sample_agent.py**: Starts aggression at turn 10
- **Result**: Challenger engages earlier, testing agent.py's early defense

### 2. Optimal Distance
- **agent.py**: Prefers 5-10 cell distance (safer)
- **sample_agent.py**: Prefers 3-8 cell distance (riskier but more pressure)
- **Result**: Closer encounters, more tactical exchanges

### 3. Space Dominance Rewards
- **agent.py**: Uses percentage-based vulnerability detection (10%, 20%, 30%)
- **sample_agent.py**: Uses more aggressive thresholds (15%, 25%) with higher bonuses
- **Result**: Challenger pushes harder when winning, may overextend

### 4. Phase Weight Differences

| Phase | Metric | agent.py | sample_agent.py | Difference |
|-------|--------|----------|-----------------|------------|
| Early | Aggression | 0.1 | 0.5 | **5x more aggressive** |
| Mid | Aggression | 0.5 | 1.0 | **2x more aggressive** |
| Mid | Dominance | 1.5 | 2.0 | Higher push |
| Late | Dominance | 3.0 | 4.0 | **Even more aggressive** |

### 5. Search Depth
- **agent.py**: Depth 6 = looks ~3 turns ahead (simultaneous moves)
- **sample_agent.py**: Depth 5 = looks ~2.5 turns ahead
- **Result**: agent.py may see deeper tactics, but Challenger is faster

## What This Tests

### 1. Early Game Defense
**Challenge**: Sample_agent applies pressure sooner (turn 10 vs turn 15)
- Does agent.py handle early aggression well?
- Can agent.py maintain space advantage under pressure?

### 2. Close-Range Tactics
**Challenge**: Sample_agent gets closer (3-8 cells vs 5-10 cells)
- Does agent.py's evaluation handle close encounters?
- Are there bugs in collision detection at close range?
- Can agent.py counter aggressive positioning?

### 3. Overextension Punishment
**Challenge**: Sample_agent may overextend due to high aggression
- Can agent.py recognize when opponent is overextended?
- Does agent.py capitalize on aggressive mistakes?
- Will deeper search (depth 6 vs 5) provide tactical advantage?

### 4. Endgame Execution
**Challenge**: Both agents have high trapping weights in endgame
- Which trapping strategy is more effective?
- Does agent.py's 3x weight beat Challenger's 4x weight?
- Who executes the final trap better?

### 5. Boost Strategy
**Challenge**: Both agents use boosts, but with different triggers
- Who uses boosts more effectively?
- Can agent.py punish wasteful boost usage?

## Expected Test Scenarios

### Scenario 1: Early Confrontation
```
Turns 10-20: Challenger approaches aggressively
Expected: 
- If agent.py backs off appropriately → Good defensive play
- If agent.py gets trapped → Bug in early game evaluation
- If agent.py counters effectively → Strong defensive strategy
```

### Scenario 2: Space Race
```
Turns 0-15: Both race for territory
Expected:
- agent.py should leverage center control bonus
- Challenger may claim different areas due to aggression
- Test: Does agent.py's center preference work?
```

### Scenario 3: Mid-Game Pressure Battle
```
Turns 20-50: High-pressure tactical exchanges
Expected:
- Close positioning (3-10 cells apart)
- Multiple direction changes
- Boost usage
- Test: Can agent.py handle sustained pressure?
```

### Scenario 4: Endgame Trap Execution
```
Turns 50+: Final trapping phase
Expected:
- Both agents trying to cut off opponent
- High trapping bonuses active (3x vs 4x)
- Test: Which trapping algorithm is superior?
```

## How to Identify Bugs in agent.py

### Bug Indicators

**1. Passive Retreat Pattern**
- agent.py consistently retreats from Challenger
- Never engages even when it has advantage
- **Diagnosis**: Pressure score weights may be too conservative

**2. Overaggression**
- agent.py rushes toward Challenger and gets trapped
- Ignores escape routes in pursuit
- **Diagnosis**: Pressure bonus too high, escape route evaluation weak

**3. Poor Trapping Execution**
- agent.py has space advantage but can't finish
- Allows Challenger to escape when cornered
- **Diagnosis**: Trapping bonus weights may be too low

**4. Symmetry/Stalemate**
- Both agents move in parallel or mirror each other
- Long games with no decisive action
- **Diagnosis**: Evaluation functions may be too similar despite differences

**5. Inconsistent Decisions**
- agent.py makes random-looking moves
- No clear strategic pattern
- **Diagnosis**: Evaluation factors may be conflicting or buggy

**6. Early Deaths**
- agent.py crashes in first 20 turns
- Especially if consistent across multiple games
- **Diagnosis**: Early game evaluation or collision detection bug

## Testing Protocol

### Run Multiple Games

```bash
# Terminal 1
python agent.py

# Terminal 2
python sample_agent.py

# Terminal 3
python judge_engine_colored.py
```

### Test Matrix (Run 5-10 games)

Track:
- [ ] Winner of each game
- [ ] Game length (turns)
- [ ] Victory method (space, trap, collision)
- [ ] Early game behavior (turns 0-20)
- [ ] Mid game behavior (turns 20-50)
- [ ] Late game behavior (turns 50+)
- [ ] Boost usage patterns
- [ ] Critical mistakes

### Expected Results (Competitive Balance)

**Healthy Results:**
- Win rate: 40-60% for either agent
- Game length: 50-150 turns typically
- Varied strategies and outcomes
- Both agents show tactical competence

**Concerning Results:**
- One agent wins >80% (indicates imbalance or bug)
- Very short games (<30 turns consistently)
- Repeated identical patterns
- One agent crashes frequently

## Performance Comparison

### Computational Cost

**agent.py:**
- Depth 6: ~7-15x more nodes than depth 5
- More comprehensive helper functions
- Slower but more thorough

**sample_agent.py:**
- Depth 5: Faster evaluations
- Simpler evaluation function
- May finish search quicker

**Implication**: agent.py should make better decisions but take longer. If sample_agent wins consistently, either:
1. Depth 6 isn't providing enough advantage
2. agent.py's evaluation has bugs
3. Aggressive strategy counters balanced strategy

## Strategic Lessons

### If agent.py Wins Consistently (>70%)
**Good signs:**
- Deeper search providing tactical advantage
- Balanced strategy beats aggression
- Evaluation function is working well

**Areas to improve sample_agent:**
- Reduce overaggression
- Better escape route evaluation
- Smarter boost usage

### If sample_agent Wins Consistently (>70%)
**Concerning signs:**
- agent.py's evaluation may have bugs
- Center control bonus may be overvalued
- Pressure scoring may be too conservative
- Trapping detection may be weak

**Areas to improve agent.py:**
- Increase aggression weights
- Better close-range tactics
- Improve trapping execution

### If Results Are Balanced (45-55%)
**Excellent!**
- Both strategies are viable
- Game is strategically deep
- No major bugs detected
- Ready for competition!

## Debug Tips

### Enable Debug Output

**agent.py** (line 33):
```python
DEBUG_EVAL = True
```

**sample_agent.py** already prints scores.

Compare score patterns:
- Are scores realistic (-100 to +100 range typically)?
- Do scores correlate with board position?
- Are there sudden unexplained score jumps?

### Watch for Patterns

**Good strategic play:**
- Scores change gradually
- Clear cause-effect between moves and scores
- Agents respond to each other's moves
- Boost usage at strategic moments

**Buggy behavior:**
- Scores stuck at same value
- Bizarre moves despite good position
- No response to opponent threats
- Random-looking decisions

## Summary

The upgraded `sample_agent.py` provides a **worthy adversary** that will thoroughly test `agent.py` through:

1. **Different playstyle**: Aggressive vs Balanced
2. **Different timing**: Earlier engagement
3. **Different priorities**: Space dominance vs Center control
4. **Similar capability**: Both use minimax, so tactical depth is comparable

This matchup will expose:
- Evaluation function bugs
- Strategic weaknesses
- Tactical oversights
- Boost usage issues
- Trapping execution problems

**Goal**: If agent.py can consistently beat or match this aggressive challenger, it's ready for competition! 🎮🚀

---

## Quick Reference: Key Differences

| Aspect | agent.py | sample_agent.py |
|--------|----------|-----------------|
| **Depth** | 6 | 5 |
| **Style** | Balanced | Aggressive |
| **Early Aggression** | Turn 15+ | Turn 10+ |
| **Optimal Distance** | 5-10 cells | 3-8 cells |
| **Late Dominance** | 3.0x | 4.0x |
| **Philosophy** | Defend then attack | Attack to defend |

Good luck testing! May the best strategy win! 🏆

