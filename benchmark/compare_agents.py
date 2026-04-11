"""
Agent benchmark — compares DQN and Heuristic Baseline across multiple seeds.

Runs each agent for every (task, seed) combination using BaseTask.run(),
which provides consistent episode management and TaskResult scoring
(same formula as the existing test suite).

Usage
-----
    # Minimal: one seed, skip LLM, easy task
    python benchmark/compare_agents.py --seeds 42 --no-llm

    # Full: three seeds, all tasks, no LLM
    python benchmark/compare_agents.py --seeds 42 43 44 --no-llm

    # Specific tasks
    python benchmark/compare_agents.py --seeds 42 --tasks easy medium --no-llm

    # Choose which agents to include
    python benchmark/compare_agents.py --seeds 42 --agents heuristic dqn --no-llm

Flags
-----
    --seeds         One or more integer seeds (required)
    --tasks         Subset of tasks to benchmark (default: easy medium hard)
    --agents        Agents to include: heuristic dqn (default: both)
    --weights       Path to DQN weights JSON (default: agents/dqn_weights.json)
    --no-llm        Skip LLM agent (required unless LLM proxy is running)
    --no-train      Skip DQN training and use existing weights only

Output
------
    Per-agent, per-task summary table with mean score ± std over the seeds.
    Pass/fail against TASK_PASSING_SCORES thresholds.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent     # adaptive_cyber_defense/benchmark/
_PKG  = _HERE.parent                        # adaptive_cyber_defense/
_ROOT = _PKG.parent                         # /Users/…/Documents/

for _p in (_ROOT, _PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adaptive_cyber_defense.tasks import (   # noqa: E402
    EasyTask, MediumTask, HardTask, NightmareTask, EliteTask,
)
from adaptive_cyber_defense.agents.baseline import BaselineAgent  # noqa: E402
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent      # noqa: E402
from grader import TASK_PASSING_SCORES                             # noqa: E402

_TASK_MAP = {
    "easy":      EasyTask,
    "medium":    MediumTask,
    "hard":      HardTask,
    "nightmare": NightmareTask,
    "elite":     EliteTask,
}

_DEFAULT_WEIGHTS = _PKG / "agents" / "dqn_weights.json"
_DEFAULT_TASKS   = ["easy", "medium", "hard"]
_DEFAULT_AGENTS  = ["heuristic", "dqn"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark DQN vs Heuristic agents")
    p.add_argument("--seeds",    type=int,  nargs="+",  required=True,
                   help="Seeds to average over, e.g. --seeds 42 43 44")
    p.add_argument("--tasks",    type=str,  nargs="+",  default=_DEFAULT_TASKS,
                   help="Tasks to benchmark (default: easy medium hard)")
    p.add_argument("--agents",   type=str,  nargs="+",  default=_DEFAULT_AGENTS,
                   help="Agents to include: heuristic dqn")
    p.add_argument("--weights",  type=str,  default=str(_DEFAULT_WEIGHTS),
                   help="Path to DQN weights JSON")
    p.add_argument("--no-llm",   action="store_true",
                   help="Skip LLM agent (required unless proxy is running)")
    p.add_argument("--no-train", action="store_true",
                   help="Skip DQN auto-training; fail if weights file missing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_agent(agent, task_name: str, seeds: List[int]) -> Dict:
    """Run agent on task for each seed. Returns per-seed scores and mean/std."""
    task_cls = _TASK_MAP[task_name]
    scores: List[float] = []
    for seed in seeds:
        task   = task_cls()
        result = task.run(agent, seed=seed)
        scores.append(result.episode_score)
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores))
    std  = variance ** 0.5
    return {
        "scores": scores,
        "mean":   round(mean, 4),
        "std":    round(std, 4),
        "min":    round(min(scores), 4),
        "max":    round(max(scores), 4),
    }


def _auto_train_dqn(weights_path: Path) -> None:
    """Quick 100-episode warm-up train if weights file is missing."""
    import subprocess
    train_script = _PKG / "training" / "train_dqn.py"
    print(f"  [benchmark] DQN weights not found — running quick 100-episode train ...")
    result = subprocess.run(
        [sys.executable, str(train_script), "--episodes", "100", "--seed", "42"],
        cwd=str(_PKG),
    )
    if result.returncode != 0:
        raise RuntimeError("DQN auto-training failed. Run training/train_dqn.py manually.")
    print(f"  [benchmark] Auto-train complete.")


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(args: argparse.Namespace) -> None:
    seeds      = args.seeds
    task_names = [t.lower() for t in args.tasks]
    agent_keys = [a.lower() for a in args.agents]
    weights    = Path(args.weights)

    # Validate task names
    for t in task_names:
        if t not in _TASK_MAP:
            raise ValueError(f"Unknown task '{t}'. Choose from: {list(_TASK_MAP)}")

    sep  = "=" * 80
    dash = "-" * 80
    print(f"\n{sep}")
    print(f"  AGENT BENCHMARK  |  seeds={seeds}  tasks={task_names}")
    print(sep)

    # ── Build agent instances ─────────────────────────────────────────────────
    agents: Dict[str, object] = {}

    if "heuristic" in agent_keys:
        agents["heuristic"] = BaselineAgent()
        print("  [heuristic] BaselineAgent ready (MITRE-lookup heuristic)")

    if "dqn" in agent_keys:
        if not weights.exists():
            if args.no_train:
                print(f"  [dqn] WARNING: weights file not found at {weights}")
                print(f"        Run: python training/train_dqn.py --episodes 200 --seed 42")
                print(f"        Skipping DQN.")
                agent_keys = [k for k in agent_keys if k != "dqn"]
            else:
                _auto_train_dqn(weights)

        if weights.exists():
            agents["dqn"] = DQNAgent.load(str(weights))
            dqn_ep = agents["dqn"].epsilon
            print(f"  [dqn] DQNAgent loaded from {weights}  (ε={dqn_ep:.3f} — eval mode uses greedy)")
            # Force greedy evaluation: set epsilon to 0 for the benchmark
            agents["dqn"].epsilon = 0.0

    if not agents:
        print("  No agents to benchmark. Exiting.")
        return

    # ── Collect results ────────────────────────────────────────────────────────
    # results[agent_name][task_name] = {mean, std, scores, ...}
    results: Dict[str, Dict[str, Dict]] = {name: {} for name in agents}

    for agent_name, agent in agents.items():
        print(f"\n  Running {agent_name.upper()} agent ...")
        for task_name in task_names:
            stats = _run_agent(agent, task_name, seeds)
            results[agent_name][task_name] = stats
            print(f"    task={task_name:<10}  mean={stats['mean']:.4f}  "
                  f"std={stats['std']:.4f}  scores={[round(s,3) for s in stats['scores']]}")

    # ── Print results table ────────────────────────────────────────────────────
    col_w = 12
    print(f"\n{sep}")
    print("  RESULTS TABLE  (mean score ± std over seeds)")
    print(sep)

    # Header row
    header = f"{'Agent':<14} {'Task':<12} {'Threshold':>10} {'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7}  {'Result'}"
    print(header)
    print(dash)

    for agent_name in agents:
        first = True
        for task_name in task_names:
            stats     = results[agent_name][task_name]
            threshold = TASK_PASSING_SCORES.get(task_name, 0.50)
            label     = "PASS" if stats["mean"] >= threshold else "FAIL"
            agent_col = agent_name if first else ""
            first = False
            print(
                f"  {agent_col:<12}  {task_name:<12} {threshold:>10.2f} "
                f"{stats['mean']:>8.4f} {stats['std']:>7.4f} "
                f"{stats['min']:>7.4f} {stats['max']:>7.4f}  {label}"
            )
        print(dash)

    # ── Per-seed detail ────────────────────────────────────────────────────────
    print("\n  PER-SEED DETAIL")
    print(dash)
    print(f"  {'Agent':<14} {'Task':<12} " + "  ".join(f"seed={s:<6}" for s in seeds))
    print(dash)
    for agent_name in agents:
        for task_name in task_names:
            scores = results[agent_name][task_name]["scores"]
            row    = f"  {agent_name:<14} {task_name:<12} " + \
                     "  ".join(f"{s:<10.4f}" for s in scores)
            print(row)
    print(dash)

    # ── Summary: DQN vs Heuristic ─────────────────────────────────────────────
    if "heuristic" in results and "dqn" in results:
        print("\n  DQN vs HEURISTIC DELTA  (positive = DQN better)")
        print(dash)
        for task_name in task_names:
            h_mean = results["heuristic"][task_name]["mean"]
            d_mean = results["dqn"][task_name]["mean"]
            delta  = d_mean - h_mean
            sign   = "+" if delta >= 0 else ""
            print(f"  {task_name:<12}  heuristic={h_mean:.4f}  dqn={d_mean:.4f}  "
                  f"delta={sign}{delta:.4f}")
        print(dash)

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    benchmark(_parse_args())
