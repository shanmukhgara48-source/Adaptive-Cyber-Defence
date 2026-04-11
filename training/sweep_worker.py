"""
Phase 2 Sweep Worker — trains ONE DQN configuration and saves results.

This script is called by run_all_phases.py as a subprocess.  It implements
EXPONENTIAL epsilon decay (rate per episode) rather than the linear decay
in DQNAgent.decay_epsilon(), to match the sweep spec:

    epsilon_{ep+1} = max(epsilon_end, epsilon_{ep} × decay_rate)

NOTE: weights are saved as JSON despite the `.pt` extension — the project
uses pure-NumPy DQN (PyTorch unavailable on this runtime).

Sweep configurations
--------------------
  A: lr=1e-3, gamma=0.99,  exp_decay=0.995  (baseline)
  B: lr=5e-4, gamma=0.995, exp_decay=0.997  (slower decay, longer horizon)
  C: lr=1e-3, gamma=0.99,  exp_decay=0.990  (fast exploration collapse)
  D: lr=3e-4, gamma=0.995, exp_decay=0.998  (conservative, high discount)

Usage (called by orchestrator — not meant for direct use)
---------------------------------------------------------
    python training/sweep_worker.py --config A --episodes 500 --seed 42

Outputs
-------
    models/sweep_A.pt               — trained weights (JSON content)
    training/logs/sweep_A.json      — per-episode stats for analysis
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

_HERE = Path(__file__).resolve().parent
_PKG  = _HERE.parent
_ROOT = _PKG.parent

for _p in (_ROOT, _PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adaptive_cyber_defense.tasks import HardTask        # noqa: E402
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent  # noqa: E402

_MODELS_DIR = _PKG / "models"
_LOGS_DIR   = _HERE / "logs"

# Sweep configurations: (lr, gamma, exp_decay_rate_per_episode)
_SWEEP_CONFIGS = {
    "A": dict(lr=1e-3,  gamma=0.99,  decay=0.995, label="baseline"),
    "B": dict(lr=5e-4,  gamma=0.995, decay=0.997, label="slow-decay long-horizon"),
    "C": dict(lr=1e-3,  gamma=0.99,  decay=0.990, label="fast-collapse"),
    "D": dict(lr=3e-4,  gamma=0.995, decay=0.998, label="conservative high-discount"),
}

_ACTION_MAP = {0: "BLOCK_IP", 1: "ISOLATE_NODE", 2: "PATCH_SYSTEM",
               3: "RUN_DEEP_SCAN", 4: "IGNORE"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep worker — trains one DQN config")
    p.add_argument("--config",   type=str, required=True,
                   choices=list(_SWEEP_CONFIGS),
                   help="Config label: A B C D")
    p.add_argument("--episodes", type=int, default=500,
                   help="Training episodes (default: 500)")
    p.add_argument("--seed",     type=int, default=42,
                   help="Base seed (default: 42)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop with exponential decay
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> dict:
    cfg       = _SWEEP_CONFIGS[args.config]
    n_eps     = args.episodes
    base_seed = args.seed
    label     = cfg["label"]

    sep  = "=" * 68
    dash = "-" * 68

    print(f"\n{sep}")
    print(f"  SWEEP-{args.config}  |  {label}")
    print(f"  lr={cfg['lr']}  γ={cfg['gamma']}  exp_decay={cfg['decay']}/ep  "
          f"episodes={n_eps}  seed={base_seed}")
    print(sep)

    # Build environment
    task = HardTask()
    env  = task.build_env()

    # Build agent (no epsilon_decay_eps dependency — we do exponential manually)
    agent = DQNAgent(
        state_dim=54, n_actions=5,
        hidden_sizes=[128, 128],
        lr=cfg["lr"], gamma=cfg["gamma"],
        epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay_eps=99999,   # effectively disable linear decay
        batch_size=64, buffer_capacity=10000,
        target_update_freq=20, min_buffer_steps=256,
        seed=base_seed,
    )

    max_steps    = env.config.max_steps   # 30 for hard
    ep_records   = []
    ep_scores    = []
    ep_rewards   = []
    action_totals = Counter()

    print(f"\n  {'Ep':>5}  {'AvgRew':>8}  {'EpScore':>8}  {'ε':>6}  "
          f"{'Loss':>9}  {'Buf':>6}")
    print(dash)

    for ep in range(1, n_eps + 1):
        state      = env.reset(seed=base_seed + ep)
        done       = False
        step_rews  = []
        last_loss  = None

        for _ in range(max_steps):
            action_input, action_idx = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_input)

            agent.store(state, action_idx, float(reward), next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            step_rews.append(float(reward))
            action_totals[_ACTION_MAP[action_idx]] += 1
            state = next_state
            if done:
                break

        # Episode bookkeeping
        agent.maybe_update_target(ep)
        # EXPONENTIAL decay (bypass linear decay_epsilon)
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * cfg["decay"])

        ep_score  = float(env.state().episode_score)
        avg_rew   = sum(step_rews) / max(1, len(step_rews))
        ep_scores.append(round(ep_score, 4))
        ep_rewards.append(round(avg_rew,  4))

        ep_records.append({
            "episode":      ep,
            "epsilon":      round(agent.epsilon, 4),
            "avg_step_rew": round(avg_rew,   4),
            "episode_score":round(ep_score,  4),
            "loss":         round(last_loss, 6) if last_loss else None,
            "n_steps":      len(step_rews),
        })

        if ep % 50 == 0:
            win = ep_scores[-20:]
            rolling = sum(win) / len(win)
            loss_str = f"{last_loss:.5f}" if last_loss else "   n/a"
            print(f"  {ep:>5}  {avg_rew:>8.4f}  {ep_score:>8.4f}  "
                  f"{agent.epsilon:>6.3f}  {loss_str:>9}  {len(agent._buffer):>6}"
                  f"  (rolling20={rolling:.4f})")

    # ── Final stats ────────────────────────────────────────────────────────────
    avg_all     = sum(ep_scores) / len(ep_scores)
    avg_last20  = sum(ep_scores[-20:]) / min(20, len(ep_scores))
    avg_last50  = sum(ep_scores[-50:]) / min(50, len(ep_scores))
    total_acts  = sum(action_totals.values())
    action_pct  = {a: round(100 * c / max(1, total_acts), 1)
                   for a, c in action_totals.most_common()}

    print(dash)
    print(f"  Config {args.config}  avg_all={avg_all:.4f}  "
          f"avg_last20={avg_last20:.4f}  avg_last50={avg_last50:.4f}  "
          f"final_ε={agent.epsilon:.4f}")
    print(sep)

    # ── Save weights ───────────────────────────────────────────────────────────
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = _MODELS_DIR / f"sweep_{args.config}.pt"
    agent.save(str(weights_path))
    print(f"  Weights → {weights_path}")

    # ── Save per-episode log ───────────────────────────────────────────────────
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _LOGS_DIR / f"sweep_{args.config}.json"
    summary = {
        "config":         args.config,
        "label":          label,
        "lr":             cfg["lr"],
        "gamma":          cfg["gamma"],
        "exp_decay_rate": cfg["decay"],
        "n_episodes":     n_eps,
        "base_seed":      base_seed,
        "avg_score_all":  round(avg_all,    4),
        "avg_score_last20": round(avg_last20, 4),
        "avg_score_last50": round(avg_last50, 4),
        "final_epsilon":  round(agent.epsilon, 4),
        "action_pct":     action_pct,
        "episodes":       ep_records,
    }
    log_path.write_text(json.dumps(summary, indent=2))
    print(f"  Log    → {log_path}")

    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = train(_parse_args())
    # Print final score so orchestrator can capture it from stdout if needed
    print(f"SWEEP_RESULT config={result['config']} "
          f"avg_last20={result['avg_score_last20']} "
          f"avg_last50={result['avg_score_last50']}")
