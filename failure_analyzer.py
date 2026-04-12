"""
failure_analyzer.py — Cross-episode failure pattern analysis.

Tracks every episode outcome, detects recurring weakness scenarios, and
produces a structured post-training report with actionable insights.

What counts as a failure
------------------------
An episode is a failure if its grader_score falls below the task's passing
threshold (imported from grader.TASK_PASSING_SCORES).

What gets recorded per episode
-------------------------------
  - task / seed / score / passed
  - attack_strategy used by the red team
  - resource condition at episode end (exhausted / tight / ok)
  - observability condition (hidden threats remaining)
  - scan budget usage fraction
  - health at episode end
  - per-step action distribution

Usage
-----
    from failure_analyzer import FailureAnalyzer

    analyzer = FailureAnalyzer()

    # after each episode:
    analyzer.record(
        task           = "hard",
        seed           = 42,
        score          = 0.61,
        attack_strategy= "APT",
        resource_remaining_frac = 0.08,
        hidden_threats_remaining = 2,
        scan_coverage  = 0.40,
        health_final   = 55,
        action_counts  = {"block_ip": 3, "scan_node_2": 8, "ignore": 1},
        steps_taken    = 30,
    )

    # after training is complete:
    print(analyzer.generate_report())
    analyzer.save_report("outputs/failure_report.txt")
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from grader import TASK_PASSING_SCORES


# ---------------------------------------------------------------------------
# Condition buckets
# ---------------------------------------------------------------------------

def _resource_condition(remaining_frac: float) -> str:
    """Classify resource state at episode end into a named bucket."""
    if remaining_frac <= 0.10:
        return "exhausted"    # ≤10% budget left
    elif remaining_frac <= 0.30:
        return "tight"        # ≤30% budget left
    else:
        return "ok"


def _observability_condition(hidden_threats: int) -> str:
    """Classify how many threats the agent never found."""
    if hidden_threats == 0:
        return "full"         # agent found everything
    elif hidden_threats <= 1:
        return "partial"
    else:
        return "poor"         # ≥2 threats never revealed


def _health_condition(health_pct: float) -> str:
    """Classify system health at episode end (health / 100)."""
    if health_pct >= 0.80:
        return "healthy"
    elif health_pct >= 0.50:
        return "damaged"
    else:
        return "critical"


# ---------------------------------------------------------------------------
# Per-episode record
# ---------------------------------------------------------------------------

@dataclass
class EpisodeRecord:
    task:                    str
    seed:                    int
    score:                   float
    passed:                  bool
    attack_strategy:         str
    resource_condition:      str   # exhausted | tight | ok
    observability_condition: str   # full | partial | poor
    health_condition:        str   # healthy | damaged | critical
    scan_coverage:           float
    health_final:            float   # 0.0–1.0
    steps_taken:             int
    action_distribution:     Dict[str, float]   # action → fraction of steps


# ---------------------------------------------------------------------------
# Weakness scenario
# ---------------------------------------------------------------------------

@dataclass
class WeaknessScenario:
    """
    A named combination of conditions under which the agent fails repeatedly.

    label       : short human-readable name, e.g. "stealth + low budget"
    conditions  : dict of {field: value} that define the scenario
    fail_count  : how many episodes matched AND failed
    total_count : how many episodes matched (pass or fail)
    fail_rate   : fail_count / total_count
    avg_score   : mean score of matched episodes
    example_seeds: up to 3 seed values for reproducibility
    """
    label:         str
    conditions:    Dict[str, str]
    fail_count:    int
    total_count:   int
    fail_rate:     float
    avg_score:     float
    example_seeds: List[int]


# ---------------------------------------------------------------------------
# Failure analyzer
# ---------------------------------------------------------------------------

class FailureAnalyzer:
    """
    Cross-episode failure tracker.

    Accumulates EpisodeRecord objects, then detects weakness patterns by
    grouping on (attack_strategy, resource_condition, observability_condition)
    and flagging combinations with fail_rate ≥ threshold.
    """

    # Minimum episodes in a scenario group before it's reported as a weakness
    _MIN_SAMPLE = 2
    # Fail-rate threshold to qualify as a weakness
    _FAIL_RATE_THRESHOLD = 0.60

    def __init__(self) -> None:
        self._records: List[EpisodeRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        task:                    str,
        seed:                    int,
        score:                   float,
        attack_strategy:         str,
        resource_remaining_frac: float,
        hidden_threats_remaining: int,
        scan_coverage:           float,
        health_final:            float,   # pass raw health 0–100; normalised here
        action_counts:           Dict[str, int],
        steps_taken:             int,
    ) -> EpisodeRecord:
        """
        Record one episode outcome.

        Parameters
        ----------
        health_final            : system_health at episode end (0–100 scale)
        resource_remaining_frac : fraction of SOC budget not spent (0.0–1.0)
        hidden_threats_remaining: threats still hidden at episode end
        action_counts           : raw action → count dict from the episode
        """
        threshold    = TASK_PASSING_SCORES.get(task, 0.55)
        passed       = score >= threshold
        health_norm  = max(0.0, min(1.0, health_final / 100.0))
        total_actions = max(1, sum(action_counts.values()))
        action_dist   = {k: round(v / total_actions, 4) for k, v in action_counts.items()}

        rec = EpisodeRecord(
            task                    = task,
            seed                    = seed,
            score                   = round(score, 4),
            passed                  = passed,
            attack_strategy         = attack_strategy,
            resource_condition      = _resource_condition(resource_remaining_frac),
            observability_condition = _observability_condition(hidden_threats_remaining),
            health_condition        = _health_condition(health_norm),
            scan_coverage           = round(scan_coverage, 3),
            health_final            = health_norm,
            steps_taken             = steps_taken,
            action_distribution     = action_dist,
        )
        self._records.append(rec)
        return rec

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def detect_weaknesses(self) -> List[WeaknessScenario]:
        """
        Group episodes by (attack_strategy, resource_condition, observability_condition)
        and flag groups with enough samples and a high fail rate.

        Returns a list of WeaknessScenario sorted by fail_rate descending.
        """
        # group key → list of records
        groups: Dict[Tuple, List[EpisodeRecord]] = defaultdict(list)
        for rec in self._records:
            key = (rec.attack_strategy, rec.resource_condition, rec.observability_condition)
            groups[key].append(rec)

        weaknesses: List[WeaknessScenario] = []
        for (strategy, resource, obs), recs in groups.items():
            if len(recs) < self._MIN_SAMPLE:
                continue
            failed    = [r for r in recs if not r.passed]
            fail_rate = len(failed) / len(recs)
            if fail_rate < self._FAIL_RATE_THRESHOLD:
                continue

            avg_score = sum(r.score for r in recs) / len(recs)
            seeds     = [r.seed for r in failed[:3]]

            label = _build_weakness_label(strategy, resource, obs)
            weaknesses.append(WeaknessScenario(
                label         = label,
                conditions    = {
                    "attack_strategy":         strategy,
                    "resource_condition":      resource,
                    "observability_condition": obs,
                },
                fail_count    = len(failed),
                total_count   = len(recs),
                fail_rate     = round(fail_rate, 3),
                avg_score     = round(avg_score, 4),
                example_seeds = seeds,
            ))

        return sorted(weaknesses, key=lambda w: -w.fail_rate)

    def dominant_failure_actions(self, top_n: int = 3) -> List[Tuple[str, float]]:
        """
        Return the top N actions by average frequency in FAILED episodes.
        Useful for identifying what the agent does wrong when it loses.
        """
        failed = [r for r in self._records if not r.passed]
        if not failed:
            return []
        totals: Dict[str, float] = defaultdict(float)
        for rec in failed:
            for action, frac in rec.action_distribution.items():
                totals[action] += frac
        avg = {a: v / len(failed) for a, v in totals.items()}
        return sorted(avg.items(), key=lambda x: -x[1])[:top_n]

    def task_summary(self) -> Dict[str, dict]:
        """Per-task pass/fail/avg-score breakdown."""
        by_task: Dict[str, List[EpisodeRecord]] = defaultdict(list)
        for rec in self._records:
            by_task[rec.task].append(rec)

        summary = {}
        for task, recs in sorted(by_task.items()):
            passed = [r for r in recs if r.passed]
            summary[task] = {
                "total":     len(recs),
                "passed":    len(passed),
                "failed":    len(recs) - len(passed),
                "pass_rate": round(len(passed) / len(recs), 3),
                "avg_score": round(sum(r.score for r in recs) / len(recs), 4),
                "threshold": TASK_PASSING_SCORES.get(task, "?"),
            }
        return summary

    def attack_strategy_impact(self) -> Dict[str, dict]:
        """
        For each attack strategy, compute how often the defender fails against it.
        Strategies with high fail_rate are the attacker's most effective tools.
        """
        by_strategy: Dict[str, List[EpisodeRecord]] = defaultdict(list)
        for rec in self._records:
            by_strategy[rec.attack_strategy].append(rec)

        impact = {}
        for strategy, recs in sorted(by_strategy.items()):
            failed = [r for r in recs if not r.passed]
            impact[strategy] = {
                "episodes":  len(recs),
                "fail_count": len(failed),
                "fail_rate": round(len(failed) / len(recs), 3),
                "avg_score": round(sum(r.score for r in recs) / len(recs), 4),
            }
        return impact

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """
        Full post-training failure analysis report.

        Sections
        --------
        1. Overview              — total episodes, pass rate, avg score
        2. Per-task breakdown    — pass/fail/avg per difficulty tier
        3. Attack strategy impact — which attacks the agent struggles with
        4. Identified weaknesses  — scenario combinations with high fail rate
        5. Failure action profile — what the agent over-does when losing
        6. Recommendations        — concrete next steps
        """
        if not self._records:
            return "FailureAnalyzer: no episodes recorded yet."

        total   = len(self._records)
        passed  = sum(1 for r in self._records if r.passed)
        avg_scr = sum(r.score for r in self._records) / total

        lines = [
            "=" * 65,
            "  POST-TRAINING FAILURE ANALYSIS REPORT",
            "=" * 65,
            "",
            "── 1. OVERVIEW ─────────────────────────────────────────────",
            f"   Total episodes   : {total}",
            f"   Passed           : {passed} ({passed/total:.1%})",
            f"   Failed           : {total - passed} ({(total-passed)/total:.1%})",
            f"   Average score    : {avg_scr:.4f}",
            "",
            "── 2. PER-TASK BREAKDOWN ────────────────────────────────────",
        ]

        for task, stats in self.task_summary().items():
            bar_pass = "█" * int(stats["pass_rate"] * 20)
            bar_fail = "░" * (20 - len(bar_pass))
            lines.append(
                f"   {task:<10} pass={stats['pass_rate']:.1%}  "
                f"avg={stats['avg_score']:.4f}  "
                f"threshold={stats['threshold']}  "
                f"|{bar_pass}{bar_fail}|  "
                f"({stats['passed']}/{stats['total']})"
            )

        lines += ["", "── 3. ATTACK STRATEGY IMPACT ───────────────────────────────"]
        for strategy, stats in sorted(
            self.attack_strategy_impact().items(),
            key=lambda x: -x[1]["fail_rate"],
        ):
            lines.append(
                f"   {strategy:<16}  fail_rate={stats['fail_rate']:.1%}  "
                f"avg_score={stats['avg_score']:.4f}  "
                f"({stats['fail_count']}/{stats['episodes']} episodes failed)"
            )

        weaknesses = self.detect_weaknesses()
        lines += ["", "── 4. IDENTIFIED WEAKNESSES ─────────────────────────────────"]
        if not weaknesses:
            lines.append("   No recurring weakness patterns detected.")
        else:
            for i, w in enumerate(weaknesses, 1):
                lines += [
                    f"   [{i}] {w.label}",
                    f"       Conditions : {w.conditions}",
                    f"       Fail rate  : {w.fail_rate:.1%}  ({w.fail_count}/{w.total_count} episodes)",
                    f"       Avg score  : {w.avg_score:.4f}",
                    f"       Seeds      : {w.example_seeds}  (use these to reproduce)",
                    "",
                ]

        dom_actions = self.dominant_failure_actions()
        lines += ["── 5. FAILURE ACTION PROFILE ────────────────────────────────"]
        if not dom_actions:
            lines.append("   No failed episodes recorded.")
        else:
            lines.append("   Top actions by average frequency in FAILED episodes:")
            for action, frac in dom_actions:
                bar = "█" * int(frac * 40)
                lines.append(f"   {action:<22}  {frac:.1%}  {bar}")

        lines += [
            "",
            "── 6. RECOMMENDATIONS ───────────────────────────────────────",
        ]
        lines += _build_recommendations(weaknesses, dom_actions, self.attack_strategy_impact())
        lines += ["", "=" * 65]
        return "\n".join(lines)

    def save_report(self, path: str = "outputs/failure_report.txt") -> Path:
        """Write the text report to disk."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.generate_report())
        print(f"  [failure_analyzer] Report saved → {out}")
        return out

    def save_records_json(self, path: str = "outputs/failure_records.json") -> Path:
        """Dump all episode records to JSON for external analysis."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [vars(r) for r in self._records]
        out.write_text(json.dumps(data, indent=2))
        print(f"  [failure_analyzer] Records saved → {out}")
        return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_weakness_label(strategy: str, resource: str, obs: str) -> str:
    parts = []
    if strategy in ("APT", "ZERO_DAY", "INSIDER_THREAT"):
        parts.append("stealth attack")
    else:
        parts.append(strategy.lower().replace("_", " "))
    if resource == "exhausted":
        parts.append("budget exhausted")
    elif resource == "tight":
        parts.append("low budget")
    if obs == "poor":
        parts.append("low observability")
    elif obs == "partial":
        parts.append("partial visibility")
    return " + ".join(parts) if parts else f"{strategy}/{resource}/{obs}"


def _build_recommendations(
    weaknesses: List[WeaknessScenario],
    dom_actions: List[Tuple[str, float]],
    strategy_impact: Dict[str, dict],
) -> List[str]:
    recs: List[str] = []

    # Resource exhaustion
    resource_weak = [w for w in weaknesses if w.conditions.get("resource_condition") in ("exhausted", "tight")]
    if resource_weak:
        recs.append(
            "   • Budget exhaustion is a recurring failure condition. "
            "Train the agent to scan less on confirmed-clean nodes and "
            "reserve budget for mitigation once a threat is visible."
        )

    # Poor observability
    obs_weak = [w for w in weaknesses if w.conditions.get("observability_condition") == "poor"]
    if obs_weak:
        recs.append(
            "   • Agent fails when too many threats remain hidden. "
            "Prioritise scanning hub nodes (node_2, node_5) early — "
            "they have the highest criticality and are primary lateral-spread targets."
        )

    # Stealth attack weakness
    stealth_weak = [w for w in weaknesses if w.conditions.get("attack_strategy") in ("APT", "ZERO_DAY", "INSIDER_THREAT")]
    if stealth_weak:
        recs.append(
            "   • Agent struggles against high-evasion strategies (APT/ZERO_DAY/INSIDER). "
            "Add verify_node_X before mitigating low-confidence threats — "
            "it confirms real threats and avoids false-positive budget drain."
        )

    # Dominant over-used action
    if dom_actions:
        top_action, top_frac = dom_actions[0]
        if top_frac > 0.40:
            recs.append(
                f"   • '{top_action}' accounts for {top_frac:.1%} of actions in failed episodes. "
                "Over-reliance on a single action is the primary signal the adaptive "
                "attacker exploits to switch strategy. Diversify the action policy."
            )

    # Highest-impact attack
    hardest = max(strategy_impact.items(), key=lambda x: x[1]["fail_rate"], default=None)
    if hardest and hardest[1]["fail_rate"] >= 0.5:
        recs.append(
            f"   • '{hardest[0]}' is the attacker's most effective strategy "
            f"(fail_rate={hardest[1]['fail_rate']:.1%}). "
            "Collect more episodes against this strategy and evaluate whether "
            "the agent's IOC classification correctly distinguishes it from similar threats."
        )

    if not recs:
        recs.append("   • No systemic weaknesses detected. Increase episode count for statistical confidence.")

    return recs
