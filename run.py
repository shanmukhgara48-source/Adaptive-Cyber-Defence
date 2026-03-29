#!/usr/bin/env python3
"""
run.py — CLI runner for the Adaptive Cyber Defense Simulator.

Usage
-----
    python run.py [--task TASK] [--agent AGENT] [--seed SEED]
                  [--episodes N] [--verbose] [--json]

Examples
--------
    # Quick smoke-test with defaults
    python run.py

    # Hard task, 5 episodes, verbose output
    python run.py --task hard --episodes 5 --verbose

    # JSON output for programmatic consumption
    python run.py --task medium --agent baseline --seed 0 --json

Arguments
---------
    --task      Task difficulty: easy | medium | hard  (default: easy)
    --agent     Agent type: ignore | baseline | ql       (default: baseline)
    --seed      Starting RNG seed (default: 0).  Multi-episode runs use
                seed, seed+1, seed+2, …
    --episodes  Number of episodes to run               (default: 1)
    --verbose   Print per-step reward breakdowns
    --json      Print final results as JSON to stdout
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

# Ensure the package parent directory is on sys.path so that
# `python run.py` works without installing the package.
_PACKAGE_PARENT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

# ---------------------------------------------------------------------------
# Lazy imports (keep startup fast)
# ---------------------------------------------------------------------------

def _build_task(name: str):
    from adaptive_cyber_defense.tasks import EasyTask, MediumTask, HardTask
    tasks = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}
    if name not in tasks:
        print(f"[error] Unknown task '{name}'. Choose from: {list(tasks)}", file=sys.stderr)
        sys.exit(1)
    return tasks[name]()


def _build_agent(name: str):
    from adaptive_cyber_defense.models.action import Action, ActionInput

    if name == "ignore":
        class IgnoreAgent:
            def choose(self, state, **_):
                return ActionInput(action=Action.IGNORE)
        return IgnoreAgent()

    if name == "baseline":
        from adaptive_cyber_defense.agents import BaselineAgent
        return BaselineAgent()

    if name == "ql":
        _ql_path = Path(__file__).resolve().parent / "agents" / "ql_table.json"
        if not _ql_path.exists():
            print("[error] ql_table.json not found — run training first:", file=sys.stderr)
            print("  python3 adaptive_cyber_defense/training/train_phase1.py", file=sys.stderr)
            sys.exit(1)
        from adaptive_cyber_defense.agents.ql_agent import QLearningAgent
        agent = QLearningAgent()
        agent.load(str(_ql_path))
        agent.epsilon = 0.0   # greedy at evaluation time
        return agent

    print(f"[error] Unknown agent '{name}'. Choose from: ignore, baseline, ql", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Verbose episode runner
# ---------------------------------------------------------------------------

def run_verbose(task, agent, seed: int):
    """Run one episode, printing per-step detail to stdout."""
    env = task.build_env()
    state = env.reset(seed=seed)

    print(f"\n{'='*60}")
    print(f"Task    : {task.config.name}  (seed={seed})")
    print(f"Agent   : {type(agent).__name__}")
    print(f"MaxSteps: {task.config.max_steps}")
    print(f"{'='*60}\n")

    step_rewards: list[float] = []
    done = False
    step = 0

    while not done:
        recs = env.recommend()
        action = agent.choose(state, recommendations=recs) if hasattr(agent, 'choose') else agent.choose(state)
        state, reward, done, info = env.step(action)
        step += 1
        step_rewards.append(reward)

        bd = info.get("reward_breakdown", {})
        threats_active = info.get("threats_active", "?")
        action_name = action.action.name
        target = action.target_node or "-"

        print(
            f"  step {step:3d} | {action_name:<14} -> {target:<12} | "
            f"reward={reward:.4f} | "
            f"threats_active={threats_active} | "
            f"containment={bd.get('containment', 0):.3f}  "
            f"survival={bd.get('survival', 0):.3f}  "
            f"waste={bd.get('waste_penalty', 0):.3f}"
        )

    result = task.run.__func__  # unused — build TaskResult manually via run()
    # Re-run via task.run() for proper TaskResult (verbose run was for display only)
    result = task.run(agent, seed=seed)
    print(f"\n{result.summary()}\n")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Adaptive Cyber Defense Simulator — episode runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task",     default="easy",     help="Task difficulty: easy|medium|hard")
    parser.add_argument("--agent",    default="baseline", help="Agent: ignore|baseline|ql")
    parser.add_argument("--seed",     type=int, default=0, help="Starting RNG seed")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--verbose",  action="store_true", help="Per-step breakdown")
    parser.add_argument("--json",     action="store_true", help="Output JSON results")

    args = parser.parse_args(argv)

    task  = _build_task(args.task)
    agent = _build_agent(args.agent)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        if args.verbose and args.episodes == 1:
            result = run_verbose(task, agent, seed)
        else:
            result = task.run(agent, seed=seed)
            if not args.json:
                print(result.summary())
        results.append(result)

    # Aggregate stats
    scores = [r.episode_score for r in results]
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)

    if not args.json:
        if args.episodes > 1:
            print(f"\n--- {args.episodes}-episode summary ---")
            print(f"  avg score  : {avg_score:.4f}")
            print(f"  best score : {best_score:.4f}")
            print(f"  worst score: {worst_score:.4f}")
            print(f"  pass rate  : {passed}/{args.episodes} ({passed/args.episodes:.0%})")
            print(f"  passing bar: {task.config.passing_score}")
    else:
        output = {
            "task": args.task,
            "agent": args.agent,
            "episodes": args.episodes,
            "passing_score": task.config.passing_score,
            "aggregate": {
                "avg_score": round(avg_score, 4),
                "best_score": round(best_score, 4),
                "worst_score": round(worst_score, 4),
                "pass_rate": round(passed / args.episodes, 4),
                "passed": passed,
            },
            "results": [
                {
                    "seed": r.seed,
                    "episode_score": r.episode_score,
                    "passed": r.passed,
                    "steps_taken": r.steps_taken,
                    "threats_total": r.threats_total,
                    "threats_contained": r.threats_contained,
                    "containment_rate": r.containment_rate,
                    "critical_health_end": r.critical_health_end,
                    "avg_resource_left": r.avg_resource_left,
                    "total_reward": r.total_reward,
                    "terminal_reason": r.terminal_reason,
                }
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))

    # Exit code: 0 if all passed, 1 if any failed
    sys.exit(0 if passed == args.episodes else 1)


if __name__ == "__main__":
    main()
