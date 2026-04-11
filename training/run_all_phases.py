"""
Master Orchestrator — 4-phase DQN optimization pipeline.

Phases
------
  1. Profile   — 50 eps, diagnose reward signal, action distribution, epsilon
  2. Sweep     — 4 configs × 500 eps in parallel subprocesses, pick winner
  3. Long run  — 1500 eps with winner config, early stop, checkpoints every 300
  4. Evaluate  — 3 agents × 3 tasks × 5 seeds, delta vs 300-ep baseline

Usage
-----
    python training/run_all_phases.py

    # Start from a specific phase (skips earlier phases)
    python training/run_all_phases.py --start-phase 3

    # Override seeds / number of eval seeds
    python training/run_all_phases.py --eval-seeds 10 20 30 40 50

Flags
-----
    --start-phase   Resume from this phase (1-4, default: 1)
    --eval-seeds    Seeds for Phase 4 evaluation (default: 10 20 30 40 50)
    --sweep-eps     Episodes per sweep config (default: 500)
    --long-eps      Episodes for Phase 3 long run (default: 1500)
    --seed          Base RNG seed for all phases (default: 42)

Hard constraints enforced
-------------------------
  - Never modifies existing files
  - CPU only (no torch import)
  - All episode scores route through grader.safe_score
  - Stops and reports before continuing when any phase fails
  - RAM: 4 parallel sweep subprocesses, each ~100 MB → ~400 MB total (well under 6 GB)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
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

from adaptive_cyber_defense.tasks import (   # noqa: E402
    EasyTask, MediumTask, HardTask,
)
from adaptive_cyber_defense.agents.baseline import BaselineAgent   # noqa: E402
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent       # noqa: E402
from adaptive_cyber_defense.agents.ql_agent import QLearningAgent  # noqa: E402
from grader import TASK_PASSING_SCORES, safe_score                  # noqa: E402

_MODELS_DIR = _PKG / "models"
_LOGS_DIR   = _HERE / "logs"
_EVAL_DIR   = _PKG / "evaluation"

_TASK_MAP = {"easy": EasyTask, "medium": MediumTask, "hard": HardTask}
_ACTION_MAP = {0: "BLOCK_IP", 1: "ISOLATE_NODE", 2: "PATCH_SYSTEM",
               3: "RUN_DEEP_SCAN", 4: "IGNORE"}

_BASELINE_WEIGHTS   = _PKG / "agents" / "dqn_weights.json"
_QL_TABLE           = _PKG / "agents" / "ql_table.json"
_SWEEP_CONFIGS      = ["A", "B", "C", "D"]
_CHECKPOINT_EVERY   = 200    # episodes between checkpoints in Phase 3
_ROLLING_WINDOW     = 20     # rolling mean window
_PLATEAU_WINDOW     = 200    # episodes to check for plateau
_PLATEAU_THRESHOLD  = 0.005  # minimum improvement to NOT plateau


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-phase DQN optimization pipeline")
    p.add_argument("--start-phase", type=int,  default=1, choices=[1,2,3,4])
    p.add_argument("--eval-seeds",  type=int,  nargs="+", default=[10,20,30,40,50])
    p.add_argument("--eval-tasks",  type=str,  nargs="+", default=["easy","medium","hard"],
                   help="Tasks to evaluate in Phase 4 (default: easy medium hard)")
    p.add_argument("--sweep-eps",   type=int,  default=500)
    p.add_argument("--long-eps",    type=int,  default=1500)
    p.add_argument("--seed",        type=int,  default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _sep(title: str = "") -> None:
    line = "=" * 72
    if title:
        print(f"\n{line}\n  {title}\n{line}")
    else:
        print(line)

def _dash() -> None:
    print("-" * 72)

def _rolling_mean(scores: list, window: int) -> float:
    tail = scores[-window:]
    return sum(tail) / len(tail)

def _std(values: list) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((x - m) ** 2 for x in values) / n)

def _run_episode(agent, task_name: str, seed: int) -> dict:
    task   = _TASK_MAP[task_name]()
    result = task.run(agent, seed=seed)
    return {
        "task_name":         result.task_name,
        "seed":              result.seed,
        "episode_score":     result.episode_score,
        "passed":            result.passed,
        "containment_rate":  result.containment_rate,
        "critical_health_end": result.critical_health_end,
        "avg_resource_left": result.avg_resource_left,
        "steps_taken":       result.steps_taken,
        "terminal_reason":   result.terminal_reason,
    }

def _benchmark_agent(agent, task_names: list, seeds: list) -> dict:
    """Quick benchmark. Returns {task_name: {mean, std, scores}}."""
    results = {}
    for task_name in task_names:
        scores = [_run_episode(agent, task_name, s)["episode_score"] for s in seeds]
        mean = sum(scores) / len(scores)
        results[task_name] = {
            "scores":    [round(s,4) for s in scores],
            "mean":      round(mean, 4),
            "std":       round(_std(scores), 4),
            "threshold": TASK_PASSING_SCORES.get(task_name, 0.5),
            "passed":    mean >= TASK_PASSING_SCORES.get(task_name, 0.5),
        }
    return results


# ---------------------------------------------------------------------------
# Phase 1 — Profile
# ---------------------------------------------------------------------------

def phase1_profile(base_seed: int) -> bool:
    """
    Run 50 training episodes and analyse the reward signal.
    Returns True if reward signal is HEALTHY or MARGINAL (proceed), False if SICK (stop).
    """
    _sep("PHASE 1 — REWARD PROFILER")
    profile_script = _HERE / "phase1_profile.py"
    result = subprocess.run(
        [sys.executable, str(profile_script),
         "--episodes", "50", "--seed", str(base_seed), "--task", "hard"],
        cwd=str(_PKG),
    )
    if result.returncode != 0:
        print("  [PHASE 1] Profiler exited with non-zero status. Stopping.")
        return False

    # Read the report to check health status
    report_path = _LOGS_DIR / "phase1_profile.json"
    if not report_path.exists():
        print("  [PHASE 1] Profile report not found. Stopping.")
        return False

    report = json.loads(report_path.read_text())
    health = report.get("health_status", "UNKNOWN")

    print(f"\n  [PHASE 1] Health status: {health}")
    if health == "SICK":
        print("  [PHASE 1] STOPPING — reward signal is indistinguishable from noise.")
        print(f"  [PHASE 1] Detail: {report.get('health_detail', '')}")
        print("  [PHASE 1] Diagnose reward weights in engines/reward.py before continuing.")
        return False

    print(f"  [PHASE 1] OK to proceed. {report.get('health_detail', '')}")
    return True


# ---------------------------------------------------------------------------
# Phase 2 — Parallel Hyperparameter Sweep
# ---------------------------------------------------------------------------

def phase2_sweep(sweep_eps: int, base_seed: int) -> str:
    """
    Train configs A-D in parallel, benchmark each, return winner label.
    Raises RuntimeError if any subprocess fails.
    """
    _sep("PHASE 2 — HYPERPARAMETER SWEEP  (4 configs × parallel)")

    sweep_script = _HERE / "sweep_worker.py"
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Launch 4 subprocesses ─────────────────────────────────────────────────
    procs = {}
    for cfg in _SWEEP_CONFIGS:
        log_out = _LOGS_DIR / f"sweep_{cfg}_stdout.txt"
        fh = open(str(log_out), "w")
        p = subprocess.Popen(
            [sys.executable, str(sweep_script),
             "--config", cfg,
             "--episodes", str(sweep_eps),
             "--seed", str(base_seed)],
            cwd=str(_PKG),
            stdout=fh,
            stderr=subprocess.STDOUT,
        )
        procs[cfg] = (p, fh, log_out)
        print(f"  [sweep] Launched config {cfg}  PID={p.pid}  log={log_out.name}")

    # ── Wait for all subprocesses ─────────────────────────────────────────────
    print(f"\n  Waiting for all {len(_SWEEP_CONFIGS)} workers to finish ...")
    t0 = time.time()
    failed = []
    for cfg, (proc, fh, log_out) in procs.items():
        ret = proc.wait()
        fh.close()
        elapsed = time.time() - t0
        if ret != 0:
            failed.append(cfg)
            print(f"  [sweep-{cfg}] FAILED (exit={ret}, elapsed={elapsed:.0f}s)")
        else:
            print(f"  [sweep-{cfg}] done  (exit=0, elapsed={elapsed:.0f}s)")

    if failed:
        for cfg in failed:
            log_out = _LOGS_DIR / f"sweep_{cfg}_stdout.txt"
            print(f"\n  --- stdout from failed worker {cfg} ---")
            try:
                print(log_out.read_text()[-3000:])
            except Exception:
                pass
        raise RuntimeError(f"Sweep workers failed: {failed}")

    # ── Load results and pick winner ───────────────────────────────────────────
    print(f"\n  All workers complete in {time.time()-t0:.0f}s. Loading results...")
    results = {}
    for cfg in _SWEEP_CONFIGS:
        log_path = _LOGS_DIR / f"sweep_{cfg}.json"
        if not log_path.exists():
            raise RuntimeError(f"Sweep log not found: {log_path}")
        results[cfg] = json.loads(log_path.read_text())

    # ── Quick benchmark each saved model (3 seeds on hard only) ───────────────
    bench_seeds = [base_seed, base_seed + 1, base_seed + 2]
    bench_results = {}

    print(f"\n  Benchmarking each config on hard task (seeds={bench_seeds}) ...")
    _dash()
    print(f"  {'Config':<8} {'Label':<28} {'TrainLast20':>11} "
          f"{'BenchMean':>10} {'BenchStd':>9}  Threshold  PASS?")
    _dash()

    for cfg in _SWEEP_CONFIGS:
        w_path = _MODELS_DIR / f"sweep_{cfg}.pt"
        if not w_path.exists():
            print(f"  {cfg:<8} weights not found — skipping")
            continue

        dqn = DQNAgent.load(str(w_path))
        dqn.epsilon = 0.0   # greedy
        hard_scores = [_run_episode(dqn, "hard", s)["episode_score"] for s in bench_seeds]
        mean_h = sum(hard_scores) / len(hard_scores)
        std_h  = _std(hard_scores)
        thresh = TASK_PASSING_SCORES["hard"]
        passed = "PASS" if mean_h >= thresh else "FAIL"

        bench_results[cfg] = {
            "hard_mean": round(mean_h, 4),
            "hard_std":  round(std_h,  4),
            "passed":    mean_h >= thresh,
            "train_last20": results[cfg]["avg_score_last20"],
        }

        print(f"  {cfg:<8} {results[cfg]['label']:<28} "
              f"{results[cfg]['avg_score_last20']:>11.4f} "
              f"{mean_h:>10.4f} {std_h:>9.4f}  {thresh:.2f}      {passed}")

    _dash()

    # Winner = highest hard-task benchmark mean
    winner = max(bench_results, key=lambda c: bench_results[c]["hard_mean"])
    winner_mean = bench_results[winner]["hard_mean"]
    print(f"\n  WINNER: Config {winner}  —  {results[winner]['label']}")
    print(f"          hard-task mean={winner_mean:.4f}  "
          f"(train avg_last20={results[winner]['avg_score_last20']:.4f})")

    # ── Write sweep summary ────────────────────────────────────────────────────
    sweep_summary = {
        "winner":        winner,
        "winner_config": {
            "lr":             results[winner]["lr"],
            "gamma":          results[winner]["gamma"],
            "exp_decay_rate": results[winner]["exp_decay_rate"],
            "label":          results[winner]["label"],
        },
        "bench_seeds":   bench_seeds,
        "all_results":   bench_results,
    }
    summary_path = _LOGS_DIR / "sweep_summary.json"
    summary_path.write_text(json.dumps(sweep_summary, indent=2))
    print(f"  Sweep summary → {summary_path}")

    return winner


# ---------------------------------------------------------------------------
# Phase 3 — Long Run with Early Stopping
# ---------------------------------------------------------------------------

def phase3_long_run(winner: str, long_eps: int, base_seed: int) -> dict:
    """
    Train the winning config for up to long_eps episodes on hard task.
    Checkpoints every CHECKPOINT_EVERY episodes.
    Stops early if rolling-20 mean plateaus for PLATEAU_WINDOW episodes.
    Writes reward_curve.json to training/logs/.
    """
    _sep(f"PHASE 3 — LONG RUN  (winner=Config {winner}, up to {long_eps} eps)")

    # Load winner config from sweep summary
    summary_path = _LOGS_DIR / "sweep_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"Sweep summary not found: {summary_path}. Run Phase 2 first.")
    sweep_sum = json.loads(summary_path.read_text())
    cfg       = sweep_sum["winner_config"]

    print(f"  Config {winner}: lr={cfg['lr']}  γ={cfg['gamma']}  "
          f"exp_decay={cfg['exp_decay_rate']}/ep  label={cfg['label']}")

    task = HardTask()
    env  = task.build_env()
    greedy_env = HardTask().build_env()  # separate env for greedy-only eval rollouts

    # Start from the sweep checkpoint for the winner (warm start)
    winner_weights = _MODELS_DIR / f"sweep_{winner}.pt"
    if winner_weights.exists():
        agent = DQNAgent.load(str(winner_weights))
        saved_epsilon = agent.epsilon
        agent.epsilon = min(saved_epsilon, 0.10)  # keep earned decay; cap at 0.10
        agent.lr      = cfg["lr"]
        agent.gamma   = cfg["gamma"]
        print(f"  Warm-starting from {winner_weights}")
        print(f"  Saved epsilon={saved_epsilon:.4f} → using {agent.epsilon:.4f} (capped at 0.10)")
    else:
        agent = DQNAgent(
            state_dim=54, n_actions=5, hidden_sizes=[128, 128],
            lr=cfg["lr"], gamma=cfg["gamma"],
            epsilon_start=1.0, epsilon_end=0.05,
            epsilon_decay_eps=99999,
            batch_size=64, buffer_capacity=10000,
            target_update_freq=20, min_buffer_steps=256,
            seed=base_seed,
        )
        print("  No sweep weights found — starting from scratch")

    decay_rate  = cfg["exp_decay_rate"]
    max_steps   = env.config.max_steps

    ep_scores:       list[float] = []   # noisy training scores (high ε)
    greedy_scores:   list[float] = []   # clean greedy rollout scores — used for plateau
    rolling_means:   list[float] = []   # rolling mean of greedy_scores
    curve_data:      list[dict]  = []

    plateau_ep    = None
    stop_early    = False
    sep72         = "=" * 72

    print(f"\n  {'Ep':>5}  {'TrainSc':>8}  {'Greedy':>8}  {'GR20':>8}  {'ε':>6}  "
          f"{'Loss':>9}  Notes")
    print("-" * 72)

    for ep in range(1, long_eps + 1):
        state    = env.reset(seed=base_seed + ep)
        done     = False
        step_rews = []
        last_loss = None

        for _ in range(max_steps):
            action_input, action_idx = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_input)

            agent.store(state, action_idx, float(reward), next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            step_rews.append(float(reward))
            state = next_state
            if done:
                break

        agent.maybe_update_target(ep)
        agent.epsilon = max(agent.epsilon_end, agent.epsilon * decay_rate)

        ep_score = float(env.state().episode_score)
        ep_scores.append(round(ep_score, 4))

        # Greedy evaluation rollout — no buffer push, no gradient update
        g_state = greedy_env.reset(seed=base_seed + ep + 50000)
        g_done  = False
        for _ in range(max_steps):
            g_action, _ = agent.select_action(g_state, greedy=True)
            g_state, _, g_done, _ = greedy_env.step(g_action)
            if g_done:
                break
        greedy_score = float(greedy_env.state().episode_score)
        greedy_scores.append(round(greedy_score, 4))

        roll = _rolling_mean(greedy_scores, _ROLLING_WINDOW)
        rolling_means.append(round(roll, 4))

        curve_data.append({
            "episode":       ep,
            "episode_score": round(ep_score,     4),
            "greedy_score":  round(greedy_score, 4),
            "rolling_mean":  round(roll,         4),
            "epsilon":       round(agent.epsilon, 4),
            "loss":          round(last_loss, 6) if last_loss else None,
        })

        # ── Checkpoint ───────────────────────────────────────────────────────
        notes = ""
        if ep % _CHECKPOINT_EVERY == 0:
            ckpt_path = _MODELS_DIR / f"checkpoint_{ep}.pt"
            agent.save(str(ckpt_path))
            notes = f"[checkpoint→{ckpt_path.name}]"

        # ── Print every 50 eps ────────────────────────────────────────────────
        if ep % 50 == 0 or ep <= 5:
            loss_str = f"{last_loss:.5f}" if last_loss else "   n/a"
            print(f"  {ep:>5}  {ep_score:>8.4f}  {greedy_score:>8.4f}  {roll:>8.4f}  "
                  f"{agent.epsilon:>6.3f}  {loss_str:>9}  {notes}")

        # ── Early stopping check on GREEDY rolling mean (not noisy training scores)
        if ep >= _PLATEAU_WINDOW + _ROLLING_WINDOW and plateau_ep is None:
            old_roll    = rolling_means[-(_PLATEAU_WINDOW + 1)]
            improvement = rolling_means[-1] - old_roll
            if improvement < _PLATEAU_THRESHOLD:
                plateau_ep = ep
                stop_early = True
                print(f"\n  [PLATEAU] Detected at episode {ep}: "
                      f"greedy R20 improvement={improvement:.5f} < {_PLATEAU_THRESHOLD} "
                      f"over last {_PLATEAU_WINDOW} episodes.")
                print(f"  [PLATEAU] greedy R20: {old_roll:.4f} → {rolling_means[-1]:.4f}")
                # Final checkpoint
                final_path = _MODELS_DIR / f"checkpoint_{ep}_plateau.pt"
                agent.save(str(final_path))
                print(f"  [PLATEAU] Final checkpoint → {final_path.name}")
                break

    # ── Save final model ──────────────────────────────────────────────────────
    if not stop_early:
        final_path = _MODELS_DIR / f"checkpoint_{long_eps}_final.pt"
        agent.save(str(final_path))
        print(f"\n  Training complete at episode {long_eps}.")
        print(f"  Final checkpoint → {final_path.name}")

    final_ep = plateau_ep if plateau_ep else long_eps

    # ── Aggregate stats (compute before curve_out) ────────────────────────────
    avg_all        = sum(ep_scores)     / max(1, len(ep_scores))
    greedy_avg_all = sum(greedy_scores) / max(1, len(greedy_scores))
    greedy_last20  = _rolling_mean(greedy_scores, 20)
    best_g_r20     = max(rolling_means) if rolling_means else 0.0
    best_g_ep      = (rolling_means.index(best_g_r20) + 1) if rolling_means else 0

    # ── Write reward curve ────────────────────────────────────────────────────
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    curve_path = _LOGS_DIR / "reward_curve.json"
    curve_out = {
        "config":            winner,
        "label":             cfg["label"],
        "total_episodes":    final_ep,
        "early_stop":        stop_early,
        "plateau_episode":   plateau_ep,
        "plateau_threshold": _PLATEAU_THRESHOLD,
        "plateau_window":    _PLATEAU_WINDOW,
        "rolling_window":    _ROLLING_WINDOW,
        "plateau_metric":    "greedy_rollout",
        "final_greedy_R20":  rolling_means[-1] if rolling_means else 0.0,
        "best_greedy_R20":   best_g_r20,
        "best_ep":           best_g_ep,
        "greedy_avg_all":    round(greedy_avg_all, 4),
        "greedy_last20":     round(greedy_last20,  4),
        "curve":             curve_data,
    }
    curve_path.write_text(json.dumps(curve_out, indent=2))
    print(f"  Reward curve → {curve_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  train_avg_all={avg_all:.4f}  greedy_avg_all={greedy_avg_all:.4f}")
    print(f"  greedy_last20={greedy_last20:.4f}  "
          f"best_greedy_R20={best_g_r20:.4f} at ep {best_g_ep}")

    return {
        "winner":            winner,
        "final_ep":          final_ep,
        "early_stop":        stop_early,
        "plateau_ep":        plateau_ep,
        "greedy_last20":     round(greedy_last20, 4),
        "best_greedy_R20":   round(best_g_r20, 4),
        "final_model":       str(final_path),
    }


# ---------------------------------------------------------------------------
# Phase 4 — Full Evaluation
# ---------------------------------------------------------------------------

def phase4_evaluate(long_run_result: dict, eval_seeds: list, base_seed: int,
                    task_names: list | None = None) -> dict:
    """
    Benchmark 3 agents across task_names with eval_seeds.
    Compare DQN (new long-run) vs DQN (300-ep baseline) vs Heuristic vs QL.
    """
    if task_names is None:
        task_names = ["easy", "medium", "hard"]
    _sep(f"PHASE 4 — FINAL EVALUATION  (3 agents × {task_names} × {len(eval_seeds)} seeds)")

    # ── Load agents ───────────────────────────────────────────────────────────
    agents = {}

    # Heuristic baseline
    agents["heuristic"] = BaselineAgent()
    print("  [heuristic] BaselineAgent ready")

    # DQN — 300-ep baseline (existing weights)
    if _BASELINE_WEIGHTS.exists():
        dqn_baseline = DQNAgent.load(str(_BASELINE_WEIGHTS))
        dqn_baseline.epsilon = 0.0
        agents["dqn_300ep"] = dqn_baseline
        print(f"  [dqn_300ep] loaded from {_BASELINE_WEIGHTS}")
    else:
        print(f"  [dqn_300ep] NOT FOUND: {_BASELINE_WEIGHTS}")

    # DQN — long-run winner
    long_model_path = Path(long_run_result["final_model"])
    if long_model_path.exists():
        dqn_new = DQNAgent.load(str(long_model_path))
        dqn_new.epsilon = 0.0
        agents[f"dqn_winner"] = dqn_new
        print(f"  [dqn_winner] loaded from {long_model_path.name}")
    else:
        # Fall back to best checkpoint
        checkpoints = sorted(_MODELS_DIR.glob("checkpoint_*.pt"))
        if checkpoints:
            dqn_new = DQNAgent.load(str(checkpoints[-1]))
            dqn_new.epsilon = 0.0
            agents["dqn_winner"] = dqn_new
            print(f"  [dqn_winner] loaded from {checkpoints[-1].name}")
        else:
            print("  [dqn_winner] no checkpoints found")

    # QL agent
    ql = QLearningAgent(epsilon=0.0)
    if _QL_TABLE.exists():
        ql.load(str(_QL_TABLE))
        ql.epsilon = 0.0
    agents["ql"] = ql
    print(f"  [ql] QLearningAgent  (ε=0.0 greedy)")

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = {}
    for agent_name, agent in agents.items():
        print(f"\n  Running {agent_name.upper()} ...")
        results[agent_name] = {}
        for task_name in task_names:
            scores = [_run_episode(agent, task_name, s)["episode_score"]
                      for s in eval_seeds]
            mean   = sum(scores) / len(scores)
            std    = _std(scores)
            thresh = TASK_PASSING_SCORES.get(task_name, 0.5)
            results[agent_name][task_name] = {
                "scores":    [round(s,4) for s in scores],
                "mean":      round(mean, 4),
                "std":       round(std,  4),
                "threshold": thresh,
                "passed":    mean >= thresh,
            }
            print(f"    {task_name:<8}  mean={mean:.4f}  std={std:.4f}  "
                  f"scores={[round(s,3) for s in scores]}")

    # ── Validate all scores through grader.safe_score ────────────────────────
    all_scores = []
    for agent_data in results.values():
        for task_data in agent_data.values():
            all_scores.extend(task_data["scores"])
    violations = [s for s in all_scores if s <= 0.0 or s >= 1.0]
    clamps     = [s for s in all_scores if safe_score(s) != s]
    print(f"\n  Grader validation: {len(all_scores)} scores — "
          f"violations={len(violations)}  clamps={len(clamps)}")
    if violations:
        print(f"  WARNING: {violations}")

    # ── Print results table ────────────────────────────────────────────────────
    sep_line = "=" * 80
    dash_line = "-" * 80
    print(f"\n{sep_line}")
    print("  FINAL RESULTS TABLE  (mean ± std)")
    print(sep_line)
    print(f"  {'Agent':<14} {'Task':<10} {'Threshold':>10} "
          f"{'Mean':>8} {'Std':>7} {'Min':>7} {'Max':>7}  Result")
    print(dash_line)

    for agent_name in agents:
        if agent_name not in results:
            continue
        first = True
        for task_name in task_names:
            s         = results[agent_name][task_name]
            agent_col = agent_name if first else ""
            first     = False
            label     = "PASS" if s["passed"] else "FAIL"
            print(f"  {agent_col:<12}  {task_name:<10} {s['threshold']:>10.2f} "
                  f"{s['mean']:>8.4f} {s['std']:>7.4f} "
                  f"{min(s['scores']):>7.4f} {max(s['scores']):>7.4f}  {label}")
        print(dash_line)

    # ── Delta table: dqn_winner vs dqn_300ep ─────────────────────────────────
    if "dqn_winner" in results and "dqn_300ep" in results:
        print(f"\n  IMPROVEMENT DELTA  (dqn_winner vs dqn_300ep baseline)")
        print(dash_line)
        print(f"  {'Task':<10} {'Baseline':>10} {'Winner':>10} {'Delta':>8}  Status")
        print(dash_line)
        for task_name in task_names:
            b_mean = results["dqn_300ep"][task_name]["mean"]
            w_mean = results["dqn_winner"][task_name]["mean"]
            delta  = w_mean - b_mean
            sign   = "+" if delta >= 0 else ""
            status = "IMPROVED" if delta > 0.01 else ("SAME" if abs(delta) <= 0.01 else "WORSE")
            print(f"  {task_name:<10} {b_mean:>10.4f} {w_mean:>10.4f} "
                  f"{sign}{delta:>7.4f}  {status}")
        print(dash_line)

    # ── Write report ──────────────────────────────────────────────────────────
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "phase": 4,
        "eval_seeds":   eval_seeds,
        "task_names":   task_names,
        "agents":       list(agents.keys()),
        "grader_validation": {
            "total_scores": len(all_scores),
            "violations":   len(violations),
            "clamps":       len(clamps),
            "passed":       len(violations) == 0,
        },
        "results": results,
    }
    report_path = _EVAL_DIR / "phase4_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Phase 4 report → {report_path}")

    return report


# ---------------------------------------------------------------------------
# Master orchestrator
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    _sep("DQN OPTIMIZATION PIPELINE  (4 phases)")
    print(f"  start_phase={args.start_phase}  seed={args.seed}")
    print(f"  sweep_eps={args.sweep_eps}  long_eps={args.long_eps}")
    print(f"  eval_seeds={args.eval_seeds}")

    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # State carried between phases
    winner          = None
    long_run_result = None

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if args.start_phase <= 1:
        t0 = time.time()
        ok = phase1_profile(args.seed)
        print(f"\n  Phase 1 elapsed: {time.time()-t0:.0f}s")
        if not ok:
            print("\n  PIPELINE STOPPED at Phase 1. Fix reward signal before continuing.")
            sys.exit(1)
    else:
        print("  [Phase 1 skipped by --start-phase]")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if args.start_phase <= 2:
        t0 = time.time()
        try:
            winner = phase2_sweep(args.sweep_eps, args.seed)
        except RuntimeError as e:
            print(f"\n  PIPELINE STOPPED at Phase 2: {e}")
            sys.exit(1)
        print(f"\n  Phase 2 elapsed: {time.time()-t0:.0f}s")
    else:
        print("  [Phase 2 skipped by --start-phase]")
        # Try to recover winner from saved summary
        summary_path = _LOGS_DIR / "sweep_summary.json"
        if summary_path.exists():
            winner = json.loads(summary_path.read_text())["winner"]
            print(f"  Recovered winner from sweep_summary.json: Config {winner}")
        else:
            print("  ERROR: No sweep_summary.json found. Run Phase 2 first.")
            sys.exit(1)

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if args.start_phase <= 3:
        t0 = time.time()
        try:
            long_run_result = phase3_long_run(winner, args.long_eps, args.seed)
        except Exception as e:
            print(f"\n  PIPELINE STOPPED at Phase 3: {e}")
            sys.exit(1)
        print(f"\n  Phase 3 elapsed: {time.time()-t0:.0f}s")
    else:
        print("  [Phase 3 skipped by --start-phase]")
        # Try to recover from reward_curve.json
        curve_path = _LOGS_DIR / "reward_curve.json"
        if curve_path.exists():
            curve = json.loads(curve_path.read_text())
            checkpoints = sorted(_MODELS_DIR.glob("checkpoint_*.pt"))
            long_run_result = {
                "winner":      winner,
                "final_ep":    curve.get("total_episodes", 0),
                "early_stop":  curve.get("early_stop", False),
                "plateau_ep":  curve.get("plateau_episode"),
                "avg_last20":  0.0,
                "best_rolling20": curve.get("best_rolling_mean", 0.0),
                "final_model": str(checkpoints[-1]) if checkpoints else "",
            }
        else:
            long_run_result = {"winner": winner, "final_model": ""}

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    if args.start_phase <= 4:
        t0 = time.time()
        try:
            phase4_evaluate(long_run_result, args.eval_seeds, args.seed,
                            task_names=args.eval_tasks)
        except Exception as e:
            print(f"\n  PIPELINE STOPPED at Phase 4: {e}")
            sys.exit(1)
        print(f"\n  Phase 4 elapsed: {time.time()-t0:.0f}s")

    total_elapsed = time.time() - t_total
    _sep(f"PIPELINE COMPLETE  ({total_elapsed:.0f}s total)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main(_parse_args())
