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
    from adaptive_cyber_defense.models.threat import generate_mitre_summary

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
    all_threats = list(state.active_threats)

    while not done:
        recs = env.recommend()
        action = agent.choose(state, recommendations=recs) if hasattr(agent, 'choose') else agent.choose(state)
        state, reward, done, info = env.step(action)
        step += 1
        step_rewards.append(reward)
        all_threats.extend(state.active_threats)

        bd = info.get("reward_breakdown", {})
        threats_active = info.get("threats_active", "?")
        action_name = action.action.name
        target = action.target_node or "-"

        # Show MITRE technique for the top active threat
        mitre_tag = ""
        if state.active_threats:
            top = max(state.active_threats, key=lambda t: t.effective_severity())
            tid  = getattr(top, "mitre_technique_id", "") or top.stage.technique_id
            tname = getattr(top, "mitre_technique_name", "") or top.stage.technique_name
            mitre_tag = f" [{tid} {tname}]"

        print(
            f"  step {step:3d} | {action_name:<14} -> {target:<12} | "
            f"reward={reward:.4f} | "
            f"threats_active={threats_active}{mitre_tag} | "
            f"containment={bd.get('containment', 0):.3f}  "
            f"survival={bd.get('survival', 0):.3f}  "
            f"waste={bd.get('waste_penalty', 0):.3f}"
        )

    # MITRE ATT&CK summary table
    mitre_counts = generate_mitre_summary(all_threats)
    if mitre_counts:
        print(f"\n{'─'*60}")
        print("MITRE ATT&CK Techniques Observed:")
        print(f"  {'Technique ID':<12}  {'Count':>5}")
        for tid, cnt in sorted(mitre_counts.items()):
            print(f"  {tid:<12}  {cnt:>5}")

    result = task.run.__func__  # unused — build TaskResult manually via run()
    # Re-run via task.run() for proper TaskResult (verbose run was for display only)
    result = task.run(agent, seed=seed)
    print(f"\n{result.summary()}\n")
    return result


# ---------------------------------------------------------------------------
# Episode runners that accept a pre-built env + adaptive attacker
# ---------------------------------------------------------------------------

def _run_with_attacker(env, task, agent, seed, attacker, strategy):
    """Run one episode on a pre-built env, feeding defender actions to attacker."""
    from adaptive_cyber_defense.tasks.base import TaskResult
    state = env.reset(seed=seed)
    step_rewards, breakdowns, resource_leftovers = [], [], []
    threats_seen: set = set()

    for t in state.active_threats:
        threats_seen.add(t.id)

    done = False
    while not done:
        action = agent.choose(state)
        attacker.observe_defender_action(action.action.name, strategy)
        state, reward, done, info = env.step(action)
        step_rewards.append(reward)
        if "reward_breakdown" in info:
            breakdowns.append(info["reward_breakdown"])
        resource_leftovers.append(info.get("resource_utilisation", 0.0))
        for t in state.active_threats:
            threats_seen.add(t.id)

    final = env.state()
    threats_total      = len(threats_seen)
    threats_still      = len(final.active_threats)
    threats_contained  = max(0, threats_total - threats_still)
    containment_rate   = threats_contained / threats_total if threats_total > 0 else 1.0
    critical_assets    = [a for a in final.assets.values() if a.criticality >= 0.7]
    critical_health    = (
        sum(a.health for a in critical_assets) / len(critical_assets)
        if critical_assets else 1.0
    )
    avg_res = (
        1.0 - sum(resource_leftovers) / len(resource_leftovers)
        if resource_leftovers else 1.0
    )

    # Reuse task scoring formula
    episode_score = task._compute_episode_score(
        containment_rate=containment_rate,
        critical_health=critical_health,
        avg_resource_left=avg_res,
        step_rewards=step_rewards,
    )

    terminal_reason = "max_steps"
    if threats_still == 0 and threats_total > 0:
        terminal_reason = "all_contained"
    elif any(a.criticality >= 0.9 and a.health <= 0.0 for a in final.assets.values()):
        terminal_reason = "critical_asset_failure"

    return TaskResult(
        task_name=task.config.name,
        seed=seed,
        episode_score=round(episode_score, 4),
        passed=episode_score >= task.config.passing_score,
        steps_taken=len(step_rewards),
        threats_total=threats_total,
        threats_contained=threats_contained,
        containment_rate=round(containment_rate, 4),
        critical_health_end=round(critical_health, 4),
        avg_resource_left=round(avg_res, 4),
        total_reward=round(sum(step_rewards), 4),
        step_rewards=step_rewards,
        reward_breakdowns=breakdowns,
        terminal_reason=terminal_reason,
    )


def run_verbose_with_env(env, task, agent, seed, attacker, strategy):
    """Verbose episode runner that feeds attacker observations and prints MITRE."""
    from adaptive_cyber_defense.models.threat import generate_mitre_summary

    state = env.reset(seed=seed)
    print(f"\nTask    : {task.config.name}  (seed={seed})")
    print(f"Agent   : {type(agent).__name__}  |  MaxSteps: {task.config.max_steps}\n")

    step_rewards, all_threats = [], list(state.active_threats)
    done, step = False, 0

    while not done:
        recs   = env.recommend()
        action = agent.choose(state, recommendations=recs) if hasattr(agent, 'choose') else agent.choose(state)
        attacker.observe_defender_action(action.action.name, strategy)
        state, reward, done, info = env.step(action)
        step += 1
        step_rewards.append(reward)
        all_threats.extend(state.active_threats)

        bd            = info.get("reward_breakdown", {})
        threats_active = info.get("threats_active", "?")
        action_name   = action.action.name
        target        = action.target_node or "-"

        mitre_tag = ""
        if state.active_threats:
            top   = max(state.active_threats, key=lambda t: t.effective_severity())
            tid   = getattr(top, "mitre_technique_id",   "") or top.stage.technique_id
            tname = getattr(top, "mitre_technique_name", "") or top.stage.technique_name
            mitre_tag = f" [{tid} {tname}]"

        print(
            f"  step {step:3d} | {action_name:<14} -> {target:<12} | "
            f"reward={reward:.4f} | "
            f"threats_active={threats_active}{mitre_tag} | "
            f"containment={bd.get('containment', 0):.3f}  "
            f"survival={bd.get('survival', 0):.3f}  "
            f"waste={bd.get('waste_penalty', 0):.3f}"
        )

    mitre_counts = generate_mitre_summary(all_threats)
    if mitre_counts:
        print(f"\n{'─'*60}")
        print("MITRE ATT&CK Techniques Observed:")
        print(f"  {'Technique ID':<12}  {'Count':>5}")
        for tid, cnt in sorted(mitre_counts.items()):
            print(f"  {tid:<12}  {cnt:>5}")

    result = _run_with_attacker(task.build_env(), task, agent, seed, attacker, strategy)
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

    from adaptive_cyber_defense.engines.adaptive_attacker import AdaptiveAttacker
    adaptive_attacker = AdaptiveAttacker(seed=args.seed)

    results = []
    for ep in range(args.episodes):
        seed = args.seed + ep

        # ── Red Team: choose strategy for this episode ───────────────────────
        attack_plan   = adaptive_attacker.on_episode_start()
        strategy      = attack_plan["attack_strategy"]
        cfg_override  = adaptive_attacker.get_attack_config_override(strategy)
        def_profile   = attack_plan["defender_profile"]

        if args.verbose or args.episodes > 1:
            print(f"\n{'='*50}")
            print(f"EPISODE {ep + 1} — RED TEAM STRATEGY: {strategy}")
            print(f"Attacker reasoning: {attack_plan['reasoning']}")
            print(f"Defender profile:   {def_profile['strategy_label']}")
            print(f"{'='*50}")

        # Apply overrides to the environment's attack engine
        env = task.build_env()
        env.set_attack_overrides(cfg_override)

        if args.verbose:
            result = run_verbose_with_env(env, task, agent, seed, adaptive_attacker, strategy)
        else:
            result = _run_with_attacker(env, task, agent, seed, adaptive_attacker, strategy)
            if not args.json:
                print(result.summary())
        results.append(result)

        # ── Red Team: end of episode update ─────────────────────────────────
        defender_won = result.containment_rate >= 0.8
        adaptive_attacker.on_episode_end(
            defender_won=defender_won,
            score=result.episode_score,
        )

    # Aggregate stats
    scores = [r.episode_score for r in results]
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(scores) / len(scores)
    best_score = max(scores)
    worst_score = min(scores)

    # ── Print final adaptation report ────────────────────────────────────────
    if not args.json and args.episodes >= 2:
        print(f"\n{adaptive_attacker.get_full_adaptation_report()}")

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
