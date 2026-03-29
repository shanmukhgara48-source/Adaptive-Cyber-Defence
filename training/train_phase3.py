#!/usr/bin/env python3
"""
Phase 3 Q-Learning Training — final 300 episodes (500 total).

Usage:
    python3 adaptive_cyber_defense/training/train_phase3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from adaptive_cyber_defense.tasks import HardTask
from adaptive_cyber_defense.agents.ql_agent import QLearningAgent, extract_state
from adaptive_cyber_defense.agents.baseline import BaselineAgent
from adaptive_cyber_defense.agents.ignore import IgnoreAgent

# ---------------------------------------------------------------------------
_PKG        = Path(__file__).resolve().parent.parent
QTABLE_PATH = _PKG / "agents" / "ql_table.json"
P2_SCORES   = Path(__file__).resolve().parent / "phase2_scores.json"
P3_SCORES   = Path(__file__).resolve().parent / "phase3_scores.json"

NEW_EPISODES  = 300
PRINT_EVERY   = 50
EVAL_SEEDS    = list(range(500, 520))   # 20 fixed held-out seeds


def rolling_avg(scores: list[float], window: int = 50) -> float:
    tail = scores[-window:]
    return sum(tail) / len(tail)


def evaluate_ql(agent: QLearningAgent, seeds: list[int]) -> list[float]:
    task = HardTask()
    saved_eps = agent.epsilon
    agent.epsilon = 0.0
    scores = []
    for seed in seeds:
        env = task.build_env()
        obs = env.reset(seed=seed)
        done = False
        while not done:
            action = agent.select_action(extract_state(obs))
            obs, _, done, _ = env.step(action)
        scores.append(env.state().episode_score)
    agent.epsilon = saved_eps
    return scores


def evaluate_agent(agent_cls, seeds: list[int]) -> list[float]:
    task = HardTask()
    agent = agent_cls()
    return [task.run(agent, seed=s).episode_score for s in seeds]


def train() -> None:
    # ── load Phase 2 checkpoint ──────────────────────────────────────────
    agent = QLearningAgent()
    agent.load(str(QTABLE_PATH))
    p2_data = json.loads(P2_SCORES.read_text())
    all_scores: list[float] = p2_data["all_scores"]   # 200 scores
    start_ep = len(all_scores) + 1                     # 201

    print("=" * 62)
    print("  Phase 3 Q-Learning Training — HardTask (300 episodes)")
    print(f"  Resuming from ep {start_ep}  ε={agent.epsilon:.4f}")
    print("=" * 62)

    task = HardTask()
    env  = task.build_env()

    for episode in range(start_ep, start_ep + NEW_EPISODES):
        obs   = env.reset(seed=episode)
        state = extract_state(obs)
        done  = False

        while not done:
            action = agent.select_action(state)
            next_obs, reward, done, _ = env.step(action)
            next_state = extract_state(next_obs)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        ep_score = env.state().episode_score
        all_scores.append(round(ep_score, 4))
        agent.decay_epsilon()

        rel = episode - start_ep + 1
        if rel % PRINT_EVERY == 0:
            ravg = rolling_avg(all_scores, 50)
            print(
                f"  Ep {episode:>3}  score={ep_score:.4f}  "
                f"roll-avg(50)={ravg:.4f}  "
                f"ε={agent.epsilon:.3f}  "
                f"Q-states={len(agent.q_table)}"
            )

    # ── training summary ─────────────────────────────────────────────────
    p1 = sum(all_scores[:50])  / 50
    p2 = sum(all_scores[50:200]) / 150
    p3 = sum(all_scores[200:]) / NEW_EPISODES
    last50 = rolling_avg(all_scores, 50)

    print("-" * 62)
    print(f"  Phase 1 avg (ep   1–50)  : {p1:.4f}")
    print(f"  Phase 2 avg (ep  51–200) : {p2:.4f}")
    print(f"  Phase 3 avg (ep 201–500) : {p3:.4f}")
    print(f"  Roll-avg last 50         : {last50:.4f}")
    trend = "📈 IMPROVING" if p3 > p1 else "➡  FLAT"
    print(f"  Overall trend            : {trend}")
    print(f"  Final ε                  : {agent.epsilon:.4f}")
    print(f"  Unique Q-states          : {len(agent.q_table)}")

    # ── 20-seed evaluation ───────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("  Final Evaluation — 20 seeds, ε=0, Hard difficulty")
    print("=" * 62)

    ql_scores   = evaluate_ql(agent, EVAL_SEEDS)
    base_scores = evaluate_agent(BaselineAgent, EVAL_SEEDS)
    ign_scores  = evaluate_agent(IgnoreAgent,   EVAL_SEEDS)

    ql_avg   = sum(ql_scores)   / len(ql_scores)
    base_avg = sum(base_scores) / len(base_scores)
    ign_avg  = sum(ign_scores)  / len(ign_scores)

    print(f"\n  {'Seed':>5}  {'QL Agent':>9}  {'Baseline':>9}  {'Ignore':>7}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*7}")
    for s, ql, bl, ig in zip(EVAL_SEEDS, ql_scores, base_scores, ign_scores):
        marker = " ✓" if ql >= bl else "  "
        print(f"  {s:>5}  {ql:>9.4f}  {bl:>9.4f}  {ig:>7.4f}{marker}")
    print(f"  {'AVG':>5}  {ql_avg:>9.4f}  {base_avg:>9.4f}  {ign_avg:>7.4f}")

    vs_base = (ql_avg - base_avg) / max(base_avg, 1e-9) * 100
    vs_ign  = (ql_avg - ign_avg)  / max(ign_avg,  1e-9) * 100
    print(f"\n  QL vs Baseline : {vs_base:+.1f}%")
    print(f"  QL vs Ignore   : {vs_ign:+.1f}%")

    if ql_avg >= base_avg:
        print("\n  ✅ QL AGENT OUTPERFORMS BASELINE")
    else:
        gap = base_avg - ql_avg
        print(f"\n  ℹ️  QL agent is {gap:.4f} below baseline — more episodes needed")

    # ── save artefacts ───────────────────────────────────────────────────
    agent.save(str(QTABLE_PATH))
    P3_SCORES.write_text(json.dumps({
        "all_scores": all_scores,
        "phase_avgs": {"p1": round(p1,4), "p2": round(p2,4), "p3": round(p3,4)},
        "eval": {
            "seeds":    EVAL_SEEDS,
            "ql":       [round(s, 4) for s in ql_scores],
            "baseline": [round(s, 4) for s in base_scores],
            "ignore":   [round(s, 4) for s in ign_scores],
            "ql_avg":   round(ql_avg, 4),
            "base_avg": round(base_avg, 4),
            "ign_avg":  round(ign_avg, 4),
        },
    }, indent=2))

    print(f"\nSaved Q-table → {QTABLE_PATH}")
    print(f"Saved scores  → {P3_SCORES}")
    print("=" * 62)


if __name__ == "__main__":
    train()
