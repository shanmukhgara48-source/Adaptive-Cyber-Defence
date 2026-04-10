"""
Curriculum Learning Scheduler
==============================
Automatically adjusts task difficulty based on rolling agent performance.

When the agent performs well (avg score >= promote_threshold), it advances
to the next difficulty level. When it struggles (avg score < demote_threshold),
it drops back. This keeps training in the productive zone where tasks are
challenging but achievable.

What it does NOT do
-------------------
- Does NOT modify scoring, grader.py, or thresholds
- Does NOT change any API schema or request format
- Does NOT affect the existing fixed-task run() flow in any way

Usage (opt-in via use_curriculum=True in inference.py)
-------------------------------------------------------
    from curriculum import CurriculumScheduler

    sched = CurriculumScheduler()
    for episode in range(20):
        task = sched.current_level
        result = run_task(task)
        next_task = sched.update(result["score"])
        print(f"  [curriculum] next → {next_task}")
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Task ordering — must match TASKS list in inference.py
# ---------------------------------------------------------------------------

TASK_ORDER: list[str] = ["easy", "medium", "hard", "nightmare", "elite"]

# Rolling window: how many recent scores to average
WINDOW: int = 3

# Score thresholds for promotion / demotion
PROMOTE_THRESHOLD: float = 0.65   # avg ≥ this → advance one level
DEMOTE_THRESHOLD: float  = 0.35   # avg < this  → drop one level


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Tracks rolling agent performance and recommends the next task difficulty.

    Parameters
    ----------
    start_level    : initial task name (default "easy")
    window         : number of recent scores to average (default 3)
    promote_thresh : avg score required to advance (default 0.65)
    demote_thresh  : avg score below which we drop back (default 0.35)
    """

    def __init__(
        self,
        start_level:    str   = "easy",
        window:         int   = WINDOW,
        promote_thresh: float = PROMOTE_THRESHOLD,
        demote_thresh:  float = DEMOTE_THRESHOLD,
    ) -> None:
        if start_level not in TASK_ORDER:
            raise ValueError(f"start_level must be one of {TASK_ORDER}")

        self.current_level:   str   = start_level
        self._window:         int   = window
        self._promote_thresh: float = promote_thresh
        self._demote_thresh:  float = demote_thresh
        self._scores:         list[float] = []   # rolling history (capped at window)

        # Human-readable history for logging / post-run inspection
        self.history: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, score: float) -> str:
        """
        Record the latest episode score and update current_level.

        Parameters
        ----------
        score : the grader score returned by run_task() (0–1 range)

        Returns
        -------
        The new (possibly unchanged) current_level after the update.
        """
        score = float(score)
        self._scores.append(score)
        if len(self._scores) > self._window:
            self._scores.pop(0)

        avg     = sum(self._scores) / len(self._scores)
        old     = self.current_level
        action  = "hold"

        if len(self._scores) == self._window:
            # Only act once we have a full window of data
            current_idx = TASK_ORDER.index(self.current_level)
            if avg >= self._promote_thresh and current_idx < len(TASK_ORDER) - 1:
                self.current_level = TASK_ORDER[current_idx + 1]
                action = "promote"
            elif avg < self._demote_thresh and current_idx > 0:
                self.current_level = TASK_ORDER[current_idx - 1]
                action = "demote"

        self.history.append({
            "score":    round(score, 4),
            "avg":      round(avg, 4),
            "action":   action,
            "from":     old,
            "to":       self.current_level,
        })

        if action != "hold":
            print(
                f"  [curriculum] {action}: {old} → {self.current_level} "
                f"(window_avg={avg:.3f})"
            )

        return self.current_level

    def next_task(self) -> str:
        """Return the recommended next task name (alias for current_level)."""
        return self.current_level

    def summary(self) -> dict:
        """
        Return a summary dict suitable for logging or JSON export.

        Keys
        ----
        current_level      : current task difficulty
        episodes_completed : total episodes recorded
        window_avg         : average score over the last window episodes (or all if fewer)
        history            : full list of {score, avg, action, from, to} entries
        """
        avg = (
            sum(self._scores) / len(self._scores)
            if self._scores else 0.0
        )
        return {
            "current_level":      self.current_level,
            "episodes_completed": len(self.history),
            "window_avg":         round(avg, 4),
            "history":            list(self.history),
        }

    def log(self) -> None:
        """Print a one-line curriculum status line."""
        s = self.summary()
        print(
            f"  [curriculum] level={s['current_level']} "
            f"episodes={s['episodes_completed']} "
            f"window_avg={s['window_avg']:.3f}"
        )

    def get_report(self) -> dict:
        """
        Return a structured report of the full curriculum run.

        Keys
        ----
        episodes    : total episodes completed
        final_level : task difficulty at end of run
        trajectory  : list of per-episode dicts — {episode, task, score, avg, action}
        promotions  : number of difficulty advances
        demotions   : number of difficulty drops
        """
        return {
            "episodes":    len(self.history),
            "final_level": self.current_level,
            "trajectory": [
                {
                    "episode": i + 1,
                    "task":    entry["to"],
                    "score":   entry["score"],
                    "avg":     entry["avg"],
                    "action":  entry["action"],
                }
                for i, entry in enumerate(self.history)
            ],
            "promotions": sum(1 for e in self.history if e["action"] == "promote"),
            "demotions":  sum(1 for e in self.history if e["action"] == "demote"),
        }

