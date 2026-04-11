"""
Extended benchmark — adds `hf://username/repo` model URI support.

This is a new script and does NOT modify compare_agents.py.
It accepts all the same flags as compare_agents.py, plus:

    --model hf://Glory-royyuru/adaptive-cyber-defense-dqn

When an `hf://` URI is supplied, the script:
  1. Parses the repo id from the URI.
  2. Downloads `dqn_weights.json` via huggingface_hub (cached locally).
  3. Passes the local cache path to DQNAgent.load(), then runs normally.

Usage
-----
    # Local weights (same as compare_agents.py)
    python benchmark/compare_agents_hf.py --seeds 42 --no-llm

    # Remote weights from HF Hub
    python benchmark/compare_agents_hf.py \
        --seeds 42 43 44 \
        --model hf://Glory-royyuru/adaptive-cyber-defense-dqn \
        --no-llm

    # HF Hub with auth token (private repos)
    HF_TOKEN=hf_xxx python benchmark/compare_agents_hf.py \
        --seeds 42 \
        --model hf://Glory-royyuru/adaptive-cyber-defense-dqn \
        --no-llm

Flags
-----
    --seeds    One or more integer seeds (required)
    --tasks    Subset of tasks (default: easy medium hard)
    --agents   Agents to include: heuristic dqn ql (default: all three)
    --model    Local path OR hf://username/repo URI for DQN weights
               (default: agents/dqn_weights.json)
    --no-llm   Skip LLM agent
    --no-train Skip DQN auto-training if weights missing
"""

from __future__ import annotations

import argparse
import os
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
from adaptive_cyber_defense.agents.baseline import BaselineAgent   # noqa: E402
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent       # noqa: E402
from adaptive_cyber_defense.agents.ql_agent import QLearningAgent  # noqa: E402
from grader import TASK_PASSING_SCORES                              # noqa: E402

_TASK_MAP = {
    "easy":      EasyTask,
    "medium":    MediumTask,
    "hard":      HardTask,
    "nightmare": NightmareTask,
    "elite":     EliteTask,
}

_DEFAULT_WEIGHTS = _PKG / "agents" / "dqn_weights.json"
_DEFAULT_QL      = _PKG / "agents" / "ql_table.json"
_DEFAULT_TASKS   = ["easy", "medium", "hard"]
_DEFAULT_AGENTS  = ["heuristic", "dqn", "ql"]


# ---------------------------------------------------------------------------
# HF URI resolution
# ---------------------------------------------------------------------------

def _resolve_weights(model_arg: str) -> Path:
    """
    Resolve --model value to a local Path.

    Accepts:
      - A local file path  (returned as-is)
      - hf://username/repo (downloads dqn_weights.json from HF Hub)
    """
    if not model_arg.startswith("hf://"):
        return Path(model_arg)

    repo_id = model_arg[len("hf://"):]
    token   = os.environ.get("HF_TOKEN") or None

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("ERROR: huggingface_hub is not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    filename = "dqn_weights.json"
    print(f"  [hf] Downloading {filename} from {repo_id} ...")
    local = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="model",
        token=token,
    )
    print(f"  [hf] Cached at: {local}")
    return Path(local)


# ---------------------------------------------------------------------------
# Helpers (mirrors compare_agents.py logic)
# ---------------------------------------------------------------------------

def _run_agent(agent, task_name: str, seeds: List[int]) -> Dict:
    task_cls = _TASK_MAP[task_name]
    scores: List[float] = []
    for seed in seeds:
        task   = task_cls()
        result = task.run(agent, seed=seed)
        scores.append(result.episode_score)
    mean     = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores))
    std      = variance ** 0.5
    return {
        "scores": scores,
        "mean":   round(mean, 4),
        "std":    round(std,  4),
        "min":    round(min(scores), 4),
        "max":    round(max(scores), 4),
    }


def _auto_train_dqn(weights_path: Path) -> None:
    import subprocess
    train_script = _PKG / "training" / "train_dqn.py"
    print("  [benchmark] DQN weights not found — running quick 100-episode train ...")
    result = subprocess.run(
        [sys.executable, str(train_script), "--episodes", "100", "--seed", "42"],
        cwd=str(_PKG),
    )
    if result.returncode != 0:
        raise RuntimeError("DQN auto-training failed. Run training/train_dqn.py manually.")
    print("  [benchmark] Auto-train complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark Heuristic / DQN / QL agents (supports hf:// model URIs)"
    )
    p.add_argument("--seeds",    type=int, nargs="+", required=True,
                   help="Seeds to average over, e.g. --seeds 42 43 44")
    p.add_argument("--tasks",    type=str, nargs="+", default=_DEFAULT_TASKS,
                   help="Tasks to benchmark (default: easy medium hard)")
    p.add_argument("--agents",   type=str, nargs="+", default=_DEFAULT_AGENTS,
                   help="Agents: heuristic dqn ql (default: all three)")
    p.add_argument("--model",    type=str, default=str(_DEFAULT_WEIGHTS),
                   help="DQN weights path OR hf://username/repo URI")
    p.add_argument("--no-llm",   action="store_true",
                   help="Skip LLM agent (required unless proxy is running)")
    p.add_argument("--no-train", action="store_true",
                   help="Skip DQN auto-training if weights missing")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def benchmark(args: argparse.Namespace) -> None:
    seeds      = args.seeds
    task_names = [t.lower() for t in args.tasks]
    agent_keys = [a.lower() for a in args.agents]

    for t in task_names:
        if t not in _TASK_MAP:
            raise ValueError(f"Unknown task '{t}'. Choose from: {list(_TASK_MAP)}")

    sep  = "=" * 80
    dash = "-" * 80
    print(f"\n{sep}")
    print(f"  AGENT BENCHMARK (HF-enabled)  |  seeds={seeds}  tasks={task_names}")
    print(sep)

    # ── Resolve DQN weights ───────────────────────────────────────────────────
    weights = _resolve_weights(args.model)

    # ── Build agent instances ─────────────────────────────────────────────────
    agents: Dict[str, object] = {}

    if "heuristic" in agent_keys:
        agents["heuristic"] = BaselineAgent()
        print("  [heuristic] BaselineAgent ready (MITRE-lookup heuristic)")

    if "dqn" in agent_keys:
        if not weights.exists():
            if args.no_train:
                print(f"  [dqn] WARNING: weights not found at {weights}. Skipping DQN.")
                agent_keys = [k for k in agent_keys if k != "dqn"]
            else:
                _auto_train_dqn(weights)
        if weights.exists():
            agents["dqn"] = DQNAgent.load(str(weights))
            agents["dqn"].epsilon = 0.0   # greedy eval
            print(f"  [dqn] DQNAgent loaded from {weights}  (ε=0.0 greedy)")

    if "ql" in agent_keys:
        ql = QLearningAgent(epsilon=0.0)  # greedy eval
        if _DEFAULT_QL.exists():
            ql.load(str(_DEFAULT_QL))
            ql.epsilon = 0.0
            print(f"  [ql] QLearningAgent loaded from {_DEFAULT_QL}  (ε=0.0 greedy)")
        else:
            print(f"  [ql] QLearningAgent — no Q-table found, using zero-init (random-ish)")
        agents["ql"] = ql

    if not agents:
        print("  No agents to benchmark. Exiting.")
        return

    # ── Collect results ────────────────────────────────────────────────────────
    results: Dict[str, Dict[str, Dict]] = {name: {} for name in agents}

    for agent_name, agent in agents.items():
        print(f"\n  Running {agent_name.upper()} agent ...")
        for task_name in task_names:
            stats = _run_agent(agent, task_name, seeds)
            results[agent_name][task_name] = stats
            print(f"    task={task_name:<10}  mean={stats['mean']:.4f}  "
                  f"std={stats['std']:.4f}  scores={[round(s,3) for s in stats['scores']]}")

    # ── Results table ─────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  RESULTS TABLE  (mean score ± std over seeds)")
    print(sep)
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
            first     = False
            print(
                f"  {agent_col:<12}  {task_name:<12} {threshold:>10.2f} "
                f"{stats['mean']:>8.4f} {stats['std']:>7.4f} "
                f"{stats['min']:>7.4f} {stats['max']:>7.4f}  {label}"
            )
        print(dash)

    # ── Per-seed detail ───────────────────────────────────────────────────────
    print("\n  PER-SEED DETAIL")
    print(dash)
    print(f"  {'Agent':<14} {'Task':<12} " + "  ".join(f"seed={s:<6}" for s in seeds))
    print(dash)
    for agent_name in agents:
        for task_name in task_names:
            scores = results[agent_name][task_name]["scores"]
            row = (f"  {agent_name:<14} {task_name:<12} "
                   + "  ".join(f"{s:<10.4f}" for s in scores))
            print(row)
    print(dash)

    print(f"\n{sep}\n")


if __name__ == "__main__":
    benchmark(_parse_args())
