"""
Final evaluation — 3 agents × 3 tasks × 5 seeds.

Runs the full benchmark for heuristic, DQN, and QL agents across easy/medium/hard
tasks with 5 seeds each (45 episodes total). Writes a structured report to
evaluation/report.json and validates every episode_score through grader.py's
safe_score() to confirm all scores are strictly in (0.0, 1.0).

Usage
-----
    # Run and write report
    python evaluation/final_eval.py

    # Dry-run only (validate report.json without running new episodes)
    python evaluation/final_eval.py --dry-run

    # Custom seeds or tasks
    python evaluation/final_eval.py --seeds 0 1 2 3 4 --tasks easy medium hard

Flags
-----
    --seeds     Five integer seeds (default: 10 20 30 40 50)
    --tasks     Tasks to evaluate (default: easy medium hard)
    --dqn-weights  Path to DQN weights (default: agents/dqn_weights.json)
    --output    Output path for report JSON (default: evaluation/report.json)
    --dry-run   Skip episodes; validate an existing report.json instead

Output
------
    evaluation/report.json  — structured results with per-episode metrics,
                              summaries, and grader validation status.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent     # adaptive_cyber_defense/evaluation/
_PKG  = _HERE.parent                        # adaptive_cyber_defense/
_ROOT = _PKG.parent                         # /Users/…/Documents/

for _p in (_ROOT, _PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adaptive_cyber_defense.tasks import (   # noqa: E402
    EasyTask, MediumTask, HardTask,
)
from adaptive_cyber_defense.agents.baseline import BaselineAgent   # noqa: E402
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent       # noqa: E402
from adaptive_cyber_defense.agents.ql_agent import QLearningAgent  # noqa: E402
from grader import TASK_PASSING_SCORES, safe_score                  # noqa: E402

_TASK_MAP = {
    "easy":   EasyTask,
    "medium": MediumTask,
    "hard":   HardTask,
}

_DEFAULT_SEEDS   = [10, 20, 30, 40, 50]
_DEFAULT_TASKS   = ["easy", "medium", "hard"]
_DEFAULT_WEIGHTS = _PKG / "agents" / "dqn_weights.json"
_DEFAULT_QL      = _PKG / "agents" / "ql_table.json"
_DEFAULT_OUTPUT  = _HERE / "report.json"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Final 3-agent evaluation with grader validation")
    p.add_argument("--seeds",       type=int, nargs="+", default=_DEFAULT_SEEDS,
                   help="Evaluation seeds (default: 10 20 30 40 50)")
    p.add_argument("--tasks",       type=str, nargs="+", default=_DEFAULT_TASKS,
                   help="Tasks to evaluate (default: easy medium hard)")
    p.add_argument("--dqn-weights", type=str, default=str(_DEFAULT_WEIGHTS),
                   help="Path to DQN weights JSON")
    p.add_argument("--output",      type=str, default=str(_DEFAULT_OUTPUT),
                   help="Output path for report.json")
    p.add_argument("--dry-run",     action="store_true",
                   help="Validate an existing report.json only (no episodes)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _run_episode(agent, task_name: str, seed: int) -> dict:
    """Run a single episode and return a dict of all TaskResult fields."""
    task   = _TASK_MAP[task_name]()
    result = task.run(agent, seed=seed)
    return {
        "task_name":         result.task_name,
        "seed":              result.seed,
        "episode_score":     result.episode_score,
        "passed":            result.passed,
        "steps_taken":       result.steps_taken,
        "threats_total":     result.threats_total,
        "threats_contained": result.threats_contained,
        "containment_rate":  result.containment_rate,
        "critical_health_end":  result.critical_health_end,
        "avg_resource_left": result.avg_resource_left,
        "total_reward":      result.total_reward,
        "terminal_reason":   result.terminal_reason,
        # grader-compatible fields for compute_grader_score()
        "grader_inputs": {
            "containment_rate":    result.containment_rate,
            "critical_health":     result.critical_health_end,
            "resource_efficiency": result.avg_resource_left,
            # speed_bonus is approximated from containment speed — BaseTask
            # does not expose containment event ages, so we use episode_score
            # as a proxy here (strictly informational; the authoritative score
            # is episode_score computed by BaseTask._compute_episode_score).
            "speed_bonus_proxy":   round(result.episode_score, 4),
        },
    }


# ---------------------------------------------------------------------------
# Grader validation
# ---------------------------------------------------------------------------

def _validate_report(report: dict) -> dict:
    """
    Pass every episode_score through grader.safe_score() and confirm
    the value is unchanged (i.e., already strictly in (0.0, 1.0)).

    Returns a validation summary dict.
    """
    errors   = []
    total    = 0
    boundary = []

    for agent_name, task_data in report["results"].items():
        for task_name, stats in task_data.items():
            for ep in stats["episodes"]:
                score = ep["episode_score"]
                total += 1
                validated = safe_score(score)
                if validated != score:
                    # safe_score had to clamp — score was at or outside boundary
                    boundary.append({
                        "agent": agent_name,
                        "task":  task_name,
                        "seed":  ep["seed"],
                        "score": score,
                        "clamped_to": validated,
                    })
                if score <= 0.0 or score >= 1.0:
                    errors.append({
                        "agent": agent_name,
                        "task":  task_name,
                        "seed":  ep["seed"],
                        "score": score,
                    })

    return {
        "total_episodes":    total,
        "strict_violations": len(errors),
        "boundary_clamps":   len(boundary),
        "errors":            errors,
        "boundary_details":  boundary,
        "passed":            len(errors) == 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_eval(args: argparse.Namespace) -> None:
    output_path = Path(args.output)
    sep  = "=" * 80
    dash = "-" * 80

    # ── Dry-run: validate existing report ─────────────────────────────────────
    if args.dry_run:
        if not output_path.exists():
            print(f"ERROR: --dry-run requires an existing report at {output_path}")
            sys.exit(1)
        print(f"\n{sep}")
        print(f"  DRY-RUN: Validating existing {output_path}")
        print(sep)
        report = json.loads(output_path.read_text())
        val    = _validate_report(report)
        _print_validation(val)
        return

    seeds      = args.seeds
    task_names = [t.lower() for t in args.tasks]
    weights    = Path(args.dqn_weights)

    for t in task_names:
        if t not in _TASK_MAP:
            raise ValueError(f"Unknown task '{t}'. Choose: {list(_TASK_MAP)}")

    print(f"\n{sep}")
    print(f"  FINAL EVALUATION  |  seeds={seeds}  tasks={task_names}")
    print(f"  3 agents × {len(task_names)} tasks × {len(seeds)} seeds = "
          f"{3 * len(task_names) * len(seeds)} episodes")
    print(sep)

    # ── Build agents ──────────────────────────────────────────────────────────
    agents: Dict[str, object] = {}

    agents["heuristic"] = BaselineAgent()
    print("  [heuristic] BaselineAgent ready")

    if weights.exists():
        dqn = DQNAgent.load(str(weights))
        dqn.epsilon = 0.0   # greedy eval
        agents["dqn"] = dqn
        print(f"  [dqn] DQNAgent loaded from {weights}  (ε=0.0 greedy)")
    else:
        print(f"  [dqn] WARNING: weights not found at {weights}. Skipping DQN.")

    ql = QLearningAgent(epsilon=0.0)
    if _DEFAULT_QL.exists():
        ql.load(str(_DEFAULT_QL))
        ql.epsilon = 0.0
        print(f"  [ql] QLearningAgent loaded from {_DEFAULT_QL}  (ε=0.0 greedy)")
    else:
        print("  [ql] QLearningAgent — no Q-table, using zero-init")
    agents["ql"] = ql

    # ── Run episodes ───────────────────────────────────────────────────────────
    results: Dict[str, Dict[str, dict]] = {}

    for agent_name, agent in agents.items():
        results[agent_name] = {}
        print(f"\n  Running {agent_name.upper()} ...")

        for task_name in task_names:
            episodes = []
            for seed in seeds:
                ep = _run_episode(agent, task_name, seed)
                episodes.append(ep)

            scores   = [ep["episode_score"] for ep in episodes]
            mean     = sum(scores) / len(scores)
            variance = sum((s - mean) ** 2 for s in scores) / max(1, len(scores))
            std      = variance ** 0.5
            threshold = TASK_PASSING_SCORES.get(task_name, 0.50)

            results[agent_name][task_name] = {
                "episodes":  episodes,
                "scores":    [round(s, 4) for s in scores],
                "mean":      round(mean, 4),
                "std":       round(std,  4),
                "min":       round(min(scores), 4),
                "max":       round(max(scores), 4),
                "threshold": threshold,
                "passed":    mean >= threshold,
            }

            print(f"    task={task_name:<8}  mean={mean:.4f}  "
                  f"std={std:.4f}  scores={[round(s,3) for s in scores]}")

    # ── Build report ───────────────────────────────────────────────────────────
    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seeds":         seeds,
            "tasks":         task_names,
            "agents":        list(agents.keys()),
            "n_episodes":    sum(
                len(results[a][t]["episodes"])
                for a in results for t in results[a]
            ),
            "dqn_weights":   str(weights),
            "ql_table":      str(_DEFAULT_QL),
            "grader_formula": (
                "0.50 × containment_rate + 0.20 × critical_health "
                "+ 0.15 × resource_efficiency + 0.15 × speed_bonus"
            ),
            "thresholds":    {k: v for k, v in TASK_PASSING_SCORES.items()
                              if k in task_names},
        },
        "results": results,
        "summary": {
            agent_name: {
                task_name: {
                    "mean":      results[agent_name][task_name]["mean"],
                    "std":       results[agent_name][task_name]["std"],
                    "threshold": results[agent_name][task_name]["threshold"],
                    "result":    "PASS" if results[agent_name][task_name]["passed"] else "FAIL",
                }
                for task_name in task_names
                if task_name in results.get(agent_name, {})
            }
            for agent_name in results
        },
    }

    # ── Validate via grader.safe_score ────────────────────────────────────────
    validation = _validate_report(report)
    report["grader_validation"] = validation

    # ── Write report ───────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Report written → {output_path}")

    # ── Print results table ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  FINAL RESULTS TABLE  (mean score ± std over seeds)")
    print(sep)
    header = (f"{'Agent':<14} {'Task':<10} {'Threshold':>10} "
              f"{'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7}  Result")
    print(header)
    print(dash)

    for agent_name in agents:
        if agent_name not in results:
            continue
        first = True
        for task_name in task_names:
            if task_name not in results[agent_name]:
                continue
            s         = results[agent_name][task_name]
            agent_col = agent_name if first else ""
            first     = False
            label     = "PASS" if s["passed"] else "FAIL"
            print(
                f"  {agent_col:<12}  {task_name:<10} {s['threshold']:>10.2f} "
                f"{s['mean']:>8.4f} {s['std']:>7.4f} "
                f"{s['min']:>7.4f} {s['max']:>7.4f}  {label}"
            )
        print(dash)

    # ── Grader validation summary ──────────────────────────────────────────────
    _print_validation(validation)
    print(f"\n{sep}\n")


def _print_validation(val: dict) -> None:
    sep  = "=" * 80
    dash = "-" * 80
    print(f"\n  GRADER VALIDATION (safe_score check on all {val['total_episodes']} episodes)")
    print(dash)
    if val["passed"]:
        print(f"  PASS — all {val['total_episodes']} episode_scores strictly in (0.0, 1.0)")
        print(f"         boundary_clamps={val['boundary_clamps']}  "
              f"strict_violations={val['strict_violations']}")
    else:
        print(f"  FAIL — {val['strict_violations']} score(s) outside strict (0, 1):")
        for err in val["errors"]:
            print(f"    agent={err['agent']}  task={err['task']}  "
                  f"seed={err['seed']}  score={err['score']}")
    if val["boundary_clamps"]:
        print(f"  NOTE: {val['boundary_clamps']} score(s) were at boundary (clamped by safe_score):")
        for b in val["boundary_details"]:
            print(f"    agent={b['agent']}  task={b['task']}  seed={b['seed']}  "
                  f"{b['score']} → {b['clamped_to']}")
    print(dash)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_eval(_parse_args())
