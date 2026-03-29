#!/usr/bin/env python3
"""
Phase 2 Q-Learning Training — continue from Phase 1 checkpoint.
Hard difficulty, 150 more episodes (200 total).

Usage:
    python3 adaptive_cyber_defense/training/train_phase2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adaptive_cyber_defense.tasks import HardTask
from adaptive_cyber_defense.models.action import Action, ActionInput
from adaptive_cyber_defense.agents.ql_agent import QLearningAgent, extract_state
from adaptive_cyber_defense.agents.baseline import BaselineAgent
from adaptive_cyber_defense.agents.ignore import IgnoreAgent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG        = Path(__file__).resolve().parent.parent
QTABLE_PATH = _PKG / "agents" / "ql_table.json"
P1_SCORES   = Path(__file__).resolve().parent / "phase1_scores.json"
P2_SCORES   = Path(__file__).resolve().parent / "phase2_scores.json"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NEW_EPISODES  = 150
PRINT_EVERY   = 25
EVAL_SEEDS    = list(range(200, 210))   # 10 fixed seeds for evaluation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rolling_avg(scores: list[float], window: int = 50) -> float:
    tail = scores[-window:]
    return sum(tail) / len(tail)


def evaluate(agent_or_cls, seeds: list[int], greedy: bool = True) -> list[float]:
    """Run evaluation episodes; greedy=True sets ε=0 for QLearningAgent."""
    task = HardTask()
    scores = []

    if isinstance(agent_or_cls, QLearningAgent):
        saved_eps = agent_or_cls.epsilon
        if greedy:
            agent_or_cls.epsilon = 0.0
        for seed in seeds:
            env = task.build_env()
            obs = env.reset(seed=seed)
            done = False
            while not done:
                action = agent_or_cls.select_action(extract_state(obs))
                obs, _, done, _ = env.step(action)
            scores.append(env.state().episode_score)
        if greedy:
            agent_or_cls.epsilon = saved_eps
    else:
        # BaselineAgent / IgnoreAgent — use task.run()
        agent = agent_or_cls()
        for seed in seeds:
            result = task.run(agent, seed=seed)
            scores.append(result.episode_score)

    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train() -> None:
    # ---- load Phase 1 artefacts -----------------------------------------
    agent = QLearningAgent()
    agent.load(str(QTABLE_PATH))
    p1_data = json.loads(P1_SCORES.read_text())
    all_scores: list[float] = p1_data["scores"]          # 50 scores so far
    start_episode = len(all_scores) + 1                   # 51

    print("=" * 60)
    print("  Phase 2 Q-Learning Training — HardTask (150 episodes)")
    print(f"  Resuming from ep {start_episode}  ε={agent.epsilon:.4f}")
    print("=" * 60)

    task = HardTask()
    env  = task.build_env()

    for episode in range(start_episode, start_episode + NEW_EPISODES):
        obs   = env.reset(seed=episode)
        state = extract_state(obs)
        done  = False

        while not done:
            action              = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state          = extract_state(next_obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        ep_score = env.state().episode_score
        all_scores.append(round(ep_score, 4))
        agent.decay_epsilon()

        rel_ep = episode - start_episode + 1          # 1-based within Phase 2
        if rel_ep % PRINT_EVERY == 0:
            ravg = rolling_avg(all_scores, 50)
            print(
                f"  Ep {episode:>3}  score={ep_score:.4f}  "
                f"roll-avg(50)={ravg:.4f}  "
                f"ε={agent.epsilon:.3f}  "
                f"Q-states={len(agent.q_table)}"
            )

    # ---- final training stats -------------------------------------------
    p1_avg  = sum(all_scores[:50])  / 50
    p2_avg  = sum(all_scores[50:])  / NEW_EPISODES
    last50  = rolling_avg(all_scores, 50)
    first50 = sum(all_scores[:50]) / 50

    print("-" * 60)
    print(f"  Phase 1 avg (ep   1–50)  : {p1_avg:.4f}")
    print(f"  Phase 2 avg (ep  51–200) : {p2_avg:.4f}")
    print(f"  Roll-avg last 50         : {last50:.4f}")
    trend = "📈 IMPROVING" if last50 > first50 else "➡  FLAT — consider tuning"
    print(f"  Overall trend            : {trend}")
    print(f"  Final ε                  : {agent.epsilon:.4f}")
    print(f"  Unique Q-states          : {len(agent.q_table)}")

    # ---- evaluation: QL vs Baseline vs Ignore ---------------------------
    print("\n" + "=" * 60)
    print("  Evaluation (10 seeds, ε=0, Hard difficulty)")
    print("=" * 60)

    ql_scores       = evaluate(agent,        EVAL_SEEDS, greedy=True)
    baseline_scores = evaluate(BaselineAgent, EVAL_SEEDS)
    ignore_scores   = evaluate(IgnoreAgent,   EVAL_SEEDS)

    ql_avg   = sum(ql_scores)       / len(ql_scores)
    base_avg = sum(baseline_scores) / len(baseline_scores)
    ign_avg  = sum(ignore_scores)   / len(ignore_scores)

    print(f"\n  {'Seed':>5}  {'QL Agent':>9}  {'Baseline':>9}  {'Ignore':>7}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*7}")
    for seed, ql, bl, ig in zip(EVAL_SEEDS, ql_scores, baseline_scores, ignore_scores):
        print(f"  {seed:>5}  {ql:>9.4f}  {bl:>9.4f}  {ig:>7.4f}")
    print(f"  {'AVG':>5}  {ql_avg:>9.4f}  {base_avg:>9.4f}  {ign_avg:>7.4f}")

    delta_vs_base = (ql_avg - base_avg) / max(base_avg, 1e-9) * 100
    delta_vs_ign  = (ql_avg - ign_avg)  / max(ign_avg,  1e-9) * 100
    print(f"\n  QL vs Baseline : {delta_vs_base:+.1f}%")
    print(f"  QL vs Ignore   : {delta_vs_ign:+.1f}%")

    # ---- save artefacts -------------------------------------------------
    agent.save(str(QTABLE_PATH))
    P2_SCORES.write_text(json.dumps({
        "all_scores": all_scores,
        "phase1_avg": round(p1_avg, 4),
        "phase2_avg": round(p2_avg, 4),
        "eval": {
            "seeds": EVAL_SEEDS,
            "ql":       [round(s, 4) for s in ql_scores],
            "baseline": [round(s, 4) for s in baseline_scores],
            "ignore":   [round(s, 4) for s in ignore_scores],
            "ql_avg":   round(ql_avg, 4),
            "base_avg": round(base_avg, 4),
            "ign_avg":  round(ign_avg, 4),
        },
    }, indent=2))

    print(f"\nSaved Q-table → {QTABLE_PATH}")
    print(f"Saved scores  → {P2_SCORES}")
    print("=" * 60)


if __name__ == "__main__":
    train()
