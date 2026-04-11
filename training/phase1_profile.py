"""
Phase 1 — DQN Training Profiler.

Runs 50 training episodes on the hard task and reports:
  - Per-episode step-reward distribution (min, max, mean, std)
  - Epsilon at every 10 episodes
  - Action frequency histogram across all steps
  - Whether step rewards ever exceed the pure-survival baseline (~0.10)
  - Episode score trace

Uses a FRESH DQNAgent (default hyperparams) so we observe training dynamics
from scratch, not a stale greedy policy.  The profiler writes a JSON report to
training/logs/phase1_profile.json and prints a human-readable summary.

Diagnosis rules
---------------
  HEALTHY  : mean step-reward > 0.15 and at least one episode score > 0.30
  MARGINAL : mean reward in [0.10, 0.15] — reward is signal but very noisy
  SICK     : mean reward < 0.10 — reward is indistinguishable from random noise;
             investigate reward weights or task config before training longer

Usage
-----
    python training/phase1_profile.py
    python training/phase1_profile.py --episodes 50 --seed 42 --task hard
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent        # adaptive_cyber_defense/training/
_PKG  = _HERE.parent                           # adaptive_cyber_defense/
_ROOT = _PKG.parent                            # /Users/…/Documents/

for _p in (_ROOT, _PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adaptive_cyber_defense.tasks import (    # noqa: E402
    EasyTask, MediumTask, HardTask,
)
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent  # noqa: E402

_TASK_MAP   = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}
_ACTION_MAP = {0: "BLOCK_IP", 1: "ISOLATE_NODE", 2: "PATCH_SYSTEM",
               3: "RUN_DEEP_SCAN", 4: "IGNORE"}

_LOGS_DIR   = _HERE / "logs"
_OUT_PATH   = _LOGS_DIR / "phase1_profile.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 DQN profiler")
    p.add_argument("--episodes", type=int, default=50,
                   help="Number of training episodes to profile (default: 50)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Base RNG seed (default: 42)")
    p.add_argument("--task",     type=str, default="hard",
                   help="Task to profile on (default: hard)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

def profile(args: argparse.Namespace) -> dict:
    n_episodes = args.episodes
    base_seed  = args.seed
    task_name  = args.task.lower()

    if task_name not in _TASK_MAP:
        raise ValueError(f"Unknown task '{task_name}'. Choose: {list(_TASK_MAP)}")

    sep  = "=" * 72
    dash = "-" * 72
    print(f"\n{sep}")
    print(f"  PHASE 1 — DQN PROFILER  |  task={task_name}  episodes={n_episodes}  seed={base_seed}")
    print(sep)

    # Build env and a fresh agent
    task = _TASK_MAP[task_name]()
    env  = task.build_env()

    # Use config values from dqn_config.yaml (confirmed match)
    agent = DQNAgent(
        state_dim=54, n_actions=5,
        hidden_sizes=[128, 128],
        lr=0.001, gamma=0.95,
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_eps=150,
        batch_size=64, buffer_capacity=10000,
        target_update_freq=20, min_buffer_steps=256,
        seed=base_seed,
    )

    # ── Tracking containers ───────────────────────────────────────────────────
    ep_records           = []   # one dict per episode
    global_action_counts = Counter()
    first_positive_ep    = None   # episode index where mean step-reward > baseline
    _SURVIVAL_BASELINE   = 0.10   # survival_bonus alone ≈ 0.15 × avg_health

    max_steps = env.config.max_steps   # hard: 30

    print(f"\n  {'Ep':>4}  {'StepRew-Mean':>13}  {'StepRew-Min':>12}  "
          f"{'StepRew-Max':>12}  {'EpScore':>8}  {'ε':>6}  {'Buf':>6}")
    print(dash)

    for ep in range(1, n_episodes + 1):
        state      = env.reset(seed=base_seed + ep)
        done       = False
        step_rews  = []
        ep_actions = Counter()
        last_loss  = None

        for _ in range(max_steps):
            action_input, action_idx = agent.select_action(state)
            next_state, reward, done, info = env.step(action_input)

            agent.store(state, action_idx, float(reward), next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            step_rews.append(float(reward))
            ep_actions[_ACTION_MAP[action_idx]] += 1
            global_action_counts[_ACTION_MAP[action_idx]] += 1

            state = next_state
            if done:
                break

        # Episode-level stats
        agent.maybe_update_target(ep)
        agent.decay_epsilon(ep)   # linear decay

        ep_score   = float(env.state().episode_score)
        mean_r     = sum(step_rews) / max(1, len(step_rews))
        min_r      = min(step_rews)
        max_r      = max(step_rews)
        variance_r = sum((r - mean_r) ** 2 for r in step_rews) / max(1, len(step_rews))
        std_r      = math.sqrt(variance_r)

        if first_positive_ep is None and mean_r > _SURVIVAL_BASELINE:
            first_positive_ep = ep

        rec = {
            "episode":       ep,
            "epsilon":       round(agent.epsilon, 4),
            "n_steps":       len(step_rews),
            "step_rew_mean": round(mean_r,  4),
            "step_rew_min":  round(min_r,   4),
            "step_rew_max":  round(max_r,   4),
            "step_rew_std":  round(std_r,   4),
            "episode_score": round(ep_score, 4),
            "last_loss":     round(last_loss, 6) if last_loss is not None else None,
            "actions":       dict(ep_actions),
        }
        ep_records.append(rec)

        # Print every episode; mark epsilon milestones
        eps_marker = f" ← ε={agent.epsilon:.3f}" if ep % 10 == 0 else ""
        print(f"  {ep:>4}  {mean_r:>13.4f}  {min_r:>12.4f}  "
              f"{max_r:>12.4f}  {ep_score:>8.4f}  {agent.epsilon:>6.3f}"
              f"  {len(agent._buffer):>6}{eps_marker}")

    # ── Aggregate stats ────────────────────────────────────────────────────────
    all_mean_r    = [r["step_rew_mean"] for r in ep_records]
    all_ep_scores = [r["episode_score"] for r in ep_records]
    all_epsilons  = [r["epsilon"]       for r in ep_records]

    grand_mean  = sum(all_mean_r)    / len(all_mean_r)
    grand_std   = math.sqrt(sum((x - grand_mean) ** 2 for x in all_mean_r) / len(all_mean_r))
    score_mean  = sum(all_ep_scores) / len(all_ep_scores)
    score_max   = max(all_ep_scores)
    eps_initial = all_epsilons[0]
    eps_final   = all_epsilons[-1]

    total_actions = sum(global_action_counts.values())
    action_pct    = {
        a: round(100.0 * c / max(1, total_actions), 1)
        for a, c in global_action_counts.most_common()
    }

    # ── Diagnosis ──────────────────────────────────────────────────────────────
    if grand_mean > 0.15:
        health = "HEALTHY"
        health_detail = (
            f"mean step-reward={grand_mean:.4f} is well above survival baseline "
            f"({_SURVIVAL_BASELINE:.2f}). Reward signal is useful."
        )
    elif grand_mean >= 0.10:
        health = "MARGINAL"
        health_detail = (
            f"mean step-reward={grand_mean:.4f} is slightly above survival baseline. "
            f"Reward exists but is noisy. Consider tuning reward weights."
        )
    else:
        health = "SICK"
        health_detail = (
            f"mean step-reward={grand_mean:.4f} is at or below survival baseline. "
            f"Reward is indistinguishable from noise. Diagnose before training."
        )

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  PHASE 1 SUMMARY")
    print(dash)
    print(f"  Reward (step-level, all {n_episodes} episodes):")
    print(f"    grand mean ± std : {grand_mean:.4f} ± {grand_std:.4f}")
    print(f"    min of per-ep means : {min(all_mean_r):.4f}")
    print(f"    max of per-ep means : {max(all_mean_r):.4f}")
    print(f"  Episode scores:")
    print(f"    mean : {score_mean:.4f}    max : {score_max:.4f}")
    print(f"  Epsilon decay (linear over {agent.epsilon_decay_eps} eps):")
    print(f"    ep 1 : {eps_initial:.3f}  →  ep {n_episodes} : {eps_final:.3f}")
    print(f"  Action distribution (% of all steps):")
    for a, pct in action_pct.items():
        bar = "█" * int(pct / 2)
        print(f"    {a:<16} {pct:>5.1f}%  {bar}")
    print(f"  First episode mean-reward > baseline ({_SURVIVAL_BASELINE}): "
          f"{first_positive_ep if first_positive_ep else 'never'}")
    print(f"\n  HEALTH STATUS: {health}")
    print(f"  {health_detail}")
    print(dash)

    # ── Build report ───────────────────────────────────────────────────────────
    report = {
        "task":            task_name,
        "n_episodes":      n_episodes,
        "base_seed":       base_seed,
        "health_status":   health,
        "health_detail":   health_detail,
        "step_reward": {
            "grand_mean": round(grand_mean, 4),
            "grand_std":  round(grand_std,  4),
            "ep_min_mean": round(min(all_mean_r), 4),
            "ep_max_mean": round(max(all_mean_r), 4),
        },
        "episode_scores": {
            "mean": round(score_mean, 4),
            "max":  round(score_max,  4),
        },
        "epsilon": {
            "initial": eps_initial,
            "final":   eps_final,
            "decay_mode": "linear",
            "decay_episodes": agent.epsilon_decay_eps,
        },
        "action_pct":          action_pct,
        "first_positive_ep":   first_positive_ep,
        "episodes":            ep_records,
    }

    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(report, indent=2))
    print(f"\n  Report → {_OUT_PATH}")
    print(sep + "\n")
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    profile(_parse_args())
