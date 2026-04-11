"""
DQN Training Script — Adaptive Cyber Defense Simulator.

Trains the DQNAgent on a single task using the OOP environment layer
(AdaptiveCyberDefenseEnv from env.py) with deterministic seeds.

Usage
-----
    python training/train_dqn.py --episodes 50 --seed 42
    python training/train_dqn.py --episodes 200 --task hard --seed 0
    python training/train_dqn.py --help

Outputs
-------
    agents/dqn_weights.json   — trained network weights (JSON)
    training/dqn_scores.json  — per-episode episode_score trace

Reports every --log-every episodes:
    episode | total_reward | episode_score | epsilon | replay_size | loss
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — must match the pattern used by train_phase1.py
# _ROOT is the parent of adaptive_cyber_defense/ so that
#     from adaptive_cyber_defense.xxx import yyy
# works correctly.
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent          # adaptive_cyber_defense/training/
_PKG  = _HERE.parent                            # adaptive_cyber_defense/
_ROOT = _PKG.parent                             # /Users/…/Documents/

for _p in (_ROOT, _PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import yaml  # noqa: E402 (after path setup)

from adaptive_cyber_defense.tasks import (   # noqa: E402
    EasyTask, MediumTask, HardTask, NightmareTask, EliteTask,
)
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent  # noqa: E402

_TASK_MAP = {
    "easy":      EasyTask,
    "medium":    MediumTask,
    "hard":      HardTask,
    "nightmare": NightmareTask,
    "elite":     EliteTask,
}

# Default paths (relative to project root _PKG)
_DEFAULT_CONFIG = _PKG / "config" / "dqn_config.yaml"
_DEFAULT_SAVE   = _PKG / "agents" / "dqn_weights.json"
_SCORES_PATH    = _HERE / "dqn_scores.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN agent on Adaptive Cyber Defense Simulator"
    )
    p.add_argument("--episodes",  type=int,   default=None,   help="Training episodes (overrides config)")
    p.add_argument("--seed",      type=int,   default=None,   help="Base seed (overrides config)")
    p.add_argument("--task",      type=str,   default=None,   help="Task name: easy|medium|hard|nightmare|elite")
    p.add_argument("--config",    type=str,   default=str(_DEFAULT_CONFIG), help="Path to dqn_config.yaml")
    p.add_argument("--save",      type=str,   default=None,   help="Output path for weights JSON (overrides config)")
    p.add_argument("--log-every", type=int,   default=None,   help="Log interval in episodes (overrides config)")
    p.add_argument("--no-save",   action="store_true",        help="Skip saving weights after training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> DQNAgent:
    # ── Load config ───────────────────────────────────────────────────────────
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = _DEFAULT_CONFIG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    dqn_cfg  = cfg["dqn"]
    train_cfg = cfg["training"]

    # CLI args override config
    n_episodes  = args.episodes  if args.episodes  is not None else train_cfg["n_episodes"]
    base_seed   = args.seed      if args.seed       is not None else train_cfg["seed"]
    task_name   = args.task      if args.task       is not None else train_cfg["task"]
    log_every   = args.log_every if args.log_every  is not None else train_cfg["log_every"]
    save_path   = Path(args.save) if args.save      is not None else _PKG / train_cfg["save_path"]
    max_steps   = train_cfg.get("max_steps_per_episode", 200)

    # ── Build environment ─────────────────────────────────────────────────────
    task_cls = _TASK_MAP.get(task_name.lower())
    if task_cls is None:
        raise ValueError(f"Unknown task '{task_name}'. Choose: {list(_TASK_MAP)}")
    task = task_cls()
    env  = task.build_env()

    # ── Build agent ───────────────────────────────────────────────────────────
    agent = DQNAgent(
        state_dim=          cfg["state"]["dim"],
        n_actions=          cfg["actions"]["n_actions"],
        hidden_sizes=       dqn_cfg["hidden_sizes"],
        lr=                 dqn_cfg["lr"],
        gamma=              dqn_cfg["gamma"],
        epsilon_start=      dqn_cfg["epsilon_start"],
        epsilon_end=        dqn_cfg["epsilon_end"],
        epsilon_decay_eps=  dqn_cfg["epsilon_decay_episodes"],
        batch_size=         dqn_cfg["batch_size"],
        buffer_capacity=    dqn_cfg["buffer_capacity"],
        target_update_freq= dqn_cfg["target_update_freq"],
        min_buffer_steps=   dqn_cfg["min_buffer_steps"],
        seed=               base_seed,
    )

    # ── Print header ──────────────────────────────────────────────────────────
    sep  = "=" * 68
    dash = "-" * 68
    print(f"\n{sep}")
    print(f"  DQN Training — task={task_name}  episodes={n_episodes}  seed={base_seed}")
    print(f"  state_dim={agent.state_dim}  n_actions={agent.n_actions}  "
          f"hidden={dqn_cfg['hidden_sizes']}")
    print(f"  lr={agent.lr}  γ={agent.gamma}  ε: {agent.epsilon_start}→{agent.epsilon_end} "
          f"over {agent.epsilon_decay_eps} eps")
    print(f"  buffer={dqn_cfg['buffer_capacity']}  batch={agent.batch_size}  "
          f"target_sync_every={agent.target_update_freq} eps")
    print(sep)
    print(f"{'Ep':>5}  {'TotalRew':>9}  {'Score':>7}  {'ε':>6}  {'Buffer':>7}  {'Loss':>9}")
    print(dash)

    # ── Training loop ─────────────────────────────────────────────────────────
    episode_scores: list[float] = []
    episode_rewards: list[float] = []

    for ep in range(1, n_episodes + 1):
        state = env.reset(seed=base_seed + ep)
        done       = False
        total_reward = 0.0
        last_loss    = None

        for _ in range(max_steps):
            action_input, action_idx = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_input)

            agent.store(state, action_idx, float(reward), next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            state = next_state
            total_reward += reward

            if done:
                break

        # Episode bookkeeping
        ep_score = float(env.state().episode_score)
        episode_scores.append(round(ep_score, 4))
        episode_rewards.append(round(total_reward, 4))

        agent.maybe_update_target(ep)
        agent.decay_epsilon(ep)

        if ep % log_every == 0:
            win_size = min(log_every, ep)
            avg_score = sum(episode_scores[-win_size:]) / win_size
            loss_str  = f"{last_loss:.5f}" if last_loss is not None else "   n/a"
            print(
                f"{ep:>5}  {total_reward:>9.3f}  {ep_score:>7.4f}  "
                f"{agent.epsilon:>6.3f}  {len(agent._buffer):>7}  {loss_str:>9}"
                f"  (avg{win_size}={avg_score:.4f})"
            )

    # ── Final stats ───────────────────────────────────────────────────────────
    avg_all   = sum(episode_scores) / len(episode_scores)
    avg_last  = sum(episode_scores[-10:]) / min(10, len(episode_scores))
    print(dash)
    print(f"  Episodes trained : {n_episodes}")
    print(f"  Avg score (all)  : {avg_all:.4f}")
    print(f"  Avg score (last 10): {avg_last:.4f}")
    print(f"  Final ε          : {agent.epsilon:.4f}")
    print(f"  Replay buffer    : {len(agent._buffer)} transitions")
    print(sep)

    # ── Save artefacts ────────────────────────────────────────────────────────
    if not args.no_save:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(save_path))
        _SCORES_PATH.write_text(json.dumps({
            "task":         task_name,
            "n_episodes":   n_episodes,
            "seed":         base_seed,
            "scores":       episode_scores,
            "rewards":      episode_rewards,
            "avg_score":    round(avg_all, 4),
            "avg_last_10":  round(avg_last, 4),
        }, indent=2))
        print(f"  Saved weights → {save_path}")
        print(f"  Saved scores  → {_SCORES_PATH}")

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train(_parse_args())
