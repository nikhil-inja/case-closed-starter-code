# Documentation Index

This folder contains all technical documentation and guides for the Case Closed AI agents.

## Bug Fixes & Critical Issues

### Flood Fill Issues
- **[ALL_AGENTS_FLOOD_FILL_FIX.md](ALL_AGENTS_FLOOD_FILL_FIX.md)** - Universal flood fill bug fix across all 4 agents (Latest)
- **[FLOOD_FILL_COUNTING_BUG_FIX.md](FLOOD_FILL_COUNTING_BUG_FIX.md)** - Detailed analysis of the counting bug
- **[FLOOD_FILL_BUG_FIX.md](FLOOD_FILL_BUG_FIX.md)** - Original flood fill bug discovery and fix

### Player ID & Variable Issues
- **[CRITICAL_PLAYER_ID_BUG_FIX.md](CRITICAL_PLAYER_ID_BUG_FIX.md)** - Player ID mismatch after state normalization
- **[CRITICAL_BUG_FIX.md](CRITICAL_BUG_FIX.md)** - Variable shadowing bug in minimax loop

## Strategy & Evaluation

### Agent Implementations
- **[HYBRID_EVALUATION_UPGRADE.md](HYBRID_EVALUATION_UPGRADE.md)** - Hybrid evaluation strategy combining survival + strategy
- **[ENHANCED_STRATEGY_SUMMARY.md](ENHANCED_STRATEGY_SUMMARY.md)** - Multi-factor evaluation with phase-aware weights
- **[SURVIVAL_OPTIMIZATIONS_COMPLETE.md](SURVIVAL_OPTIMIZATIONS_COMPLETE.md)** - Emergency mode and safety checks for sample_agent

### Testing & Guides
- **[CHALLENGER_AGENT_GUIDE.md](CHALLENGER_AGENT_GUIDE.md)** - Aggressive minimax challenger (sample_agent) guide
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - How to test enhanced agents

## Visualization
- **[ARROW_VISUALIZATION_UPDATE.md](ARROW_VISUALIZATION_UPDATE.md)** - Colored directional arrows in judge_engine_colored.py

## Quick Reference

### Current Agent Capabilities

**agent.py & opponent.py (Hybrid Strategy)**
- ✅ Dual-layer survival mechanisms
- ✅ Multi-factor strategic evaluation
- ✅ Phase-aware decision making
- ✅ Center control with symmetry breaking
- ✅ Accurate space calculation

**sample_agent.py & opp.py (Aggressive Strategy)**
- ✅ Emergency survival mode
- ✅ Aggressive positioning
- ✅ High trapping priority
- ✅ Accurate space calculation

### Key Metrics
- Self-collision rate: < 5%
- Average game length: 80-120 turns
- Competitive win rate: 45-55% (balanced matchups)

---

**Last Updated:** November 8, 2025

