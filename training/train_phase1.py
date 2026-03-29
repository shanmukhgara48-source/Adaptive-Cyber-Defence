#!/usr/bin/env python3
"""
Phase 1 Q-Learning Training — Hard difficulty, 50 episodes.

Usage:
    python3 adaptive_cyber_defense/training/train_phase1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure package root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adaptive_cyber_defense.tasks import HardTask
from adaptive_cyber_defense.models.action import Action, ActionInput
from adaptive_cyber_defense.agents.ql_agent import QLearningAgent, extract_state

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_EPISODES   = 50
PRINT_EVERY    = 10
QTABLE_PATH    = Path(__file__).resolve().parent.parent / "agents" / "ql_table.json"
SCORES_PATH    = Path(__file__).resolve().parent / "phase1_scores.json"

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train() -> None:
    agent = QLearningAgent(
        lr=0.1, discount=0.95,
        epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05,
    )
    task  = HardTask()
    env   = task.build_env()

    scores: list[float] = []

    print("=" * 55)
    print("  Phase 1 Q-Learning Training — HardTask (50 episodes)")
    print("=" * 55)

    for episode in range(1, NUM_EPISODES + 1):
        state_obs = env.reset(seed=episode)
        state     = extract_state(state_obs)
        done      = False
        ep_reward = 0.0

        while not done:
            action          = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state      = extract_state(next_obs)

            agent.update(state, action, reward, next_state, done)

            state      = next_state
            ep_reward += reward

        # Use the normalised episode score (0-1) for the trend plot
        ep_score = env.state().episode_score
        scores.append(round(ep_score, 4))
        agent.decay_epsilon()

        if episode % PRINT_EVERY == 0:
            window     = scores[max(0, episode - PRINT_EVERY): episode]
            window_avg = sum(window) / len(window)
            print(
                f"  Ep {episode:>3}  score={ep_score:.4f}  "
                f"avg(last {PRINT_EVERY})={window_avg:.4f}  "
                f"ε={agent.epsilon:.3f}  "
                f"Q-states={len(agent.q_table)}"
            )

    # Final stats
    avg_all = sum(scores) / len(scores)
    avg_last10 = sum(scores[-10:]) / 10
    print("-" * 55)
    print(f"  Avg score  (ep 1–50)  : {avg_all:.4f}")
    print(f"  Avg score  (ep 41–50) : {avg_last10:.4f}")
    trend = "📈 IMPROVING" if scores[-10:] > scores[:10] else "➡ FLAT / needs more training"
    print(f"  Learning trend        : {trend}")
    print(f"  Final ε               : {agent.epsilon:.4f}")
    print(f"  Unique Q-states seen  : {len(agent.q_table)}")
    print("=" * 55)

    # Per-episode table
    print("\nPer-episode scores:")
    print(f"  {'Ep':>3}  {'Score':>7}  {'ε':>6}")
    epsilon_trace = 1.0
    for i, sc in enumerate(scores, 1):
        print(f"  {i:>3}  {sc:>7.4f}  {epsilon_trace:>6.3f}")
        epsilon_trace = max(0.05, epsilon_trace * 0.995)

    # Save artefacts
    agent.save(str(QTABLE_PATH))
    SCORES_PATH.write_text(json.dumps({"scores": scores, "avg": avg_all}, indent=2))
    print(f"\nSaved Q-table → {QTABLE_PATH}")
    print(f"Saved scores  → {SCORES_PATH}")


if __name__ == "__main__":
    train()
