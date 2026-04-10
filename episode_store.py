"""
Episode Replay + Post-Mortem System
=====================================
Records every step of an episode to structured JSON for replay and inspection.
Generates a lightweight post-mortem report at episode end.

This module is COMPLETELY ADDITIVE — it does not affect scoring, API schema,
existing endpoints, or the [START]/[STEP]/[END] output format.

Usage
-----
    from episode_store import EpisodeStore

    store = EpisodeStore()                              # base_dir defaults to outputs/episodes
    store.start_episode(session_id, task, seed)
    store.record_step(session_id, step, obs, action, reward, reasoning)
    store.finalize(session_id, final_score)

Replay
------
    from episode_store import load_episode
    episode = load_episode("outputs/episodes/elite_s42_abc123.json")
    for step in episode["steps"]:
        print(step["step"], step["action"], step["reward"])
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _obs_summary(obs: dict) -> dict:
    """Extract a compact snapshot of the observation — no threat-type leakage."""
    threats = obs.get("visible_threats", [])
    return {
        "visible_threats":   len(threats),
        "hidden_threats":    int(obs.get("hidden_threat_count", 0)),
        "health":            _safe_float(obs.get("system_health", 100)) / 100.0,
        "scan_coverage":     _safe_float(obs.get("scan_coverage", 0.0)),
        "max_severity":      max((_safe_float(t.get("severity")) for t in threats), default=0.0),
        "avg_confidence":    (
            sum(_safe_float(t.get("detection_confidence", 0.5)) for t in threats) / len(threats)
            if threats else 0.0
        ),
    }


# ---------------------------------------------------------------------------
# Post-mortem heuristics  (no threat_type access — no leakage)
# ---------------------------------------------------------------------------

def _compute_post_mortem(steps: list[dict]) -> dict:
    """
    Derive actionable insights from the recorded step list.

    low_reward_steps    : steps where reward < 0.2  (likely wrong mitigation or ignore)
    late_actions        : steps > 5 where hidden threats > 0 and action was scan or ignore
    missed_opportunities: consecutive scans on the same node (no containment progress)
    """
    low_reward_steps: list[dict] = []
    late_actions: list[dict]     = []
    scan_streak: dict[str, int]  = {}   # node → consecutive scan count

    missed_opportunities: list[dict] = []

    for entry in steps:
        step    = entry["step"]
        action  = entry["action"]
        reward  = _safe_float(entry["reward"])
        summary = entry.get("summary", {})
        hidden  = summary.get("hidden_threats", 0)

        # Low-reward steps
        if reward < 0.2:
            low_reward_steps.append({
                "step":   step,
                "action": action,
                "reward": round(reward, 3),
            })

        # Late actions: past step 5, threats still hidden, agent scanning or ignoring
        if step > 5 and hidden > 0 and (action.startswith("scan_") or action == "ignore"):
            late_actions.append({
                "step":           step,
                "action":         action,
                "hidden_threats": hidden,
            })

        # Missed opportunities: same node scanned twice in a row with no improvement
        if action.startswith("scan_node_"):
            node = action.replace("scan_", "")
            scan_streak[node] = scan_streak.get(node, 0) + 1
            if scan_streak[node] >= 2:
                missed_opportunities.append({
                    "step":   step,
                    "node":   node,
                    "streak": scan_streak[node],
                    "note":   "repeated scan — consider a containment action instead",
                })
        else:
            # Reset streaks for all nodes when a non-scan action is taken
            scan_streak.clear()

    return {
        "low_reward_steps":     low_reward_steps,
        "late_actions":         late_actions,
        "missed_opportunities": missed_opportunities,
        "summary": {
            "total_low_reward":      len(low_reward_steps),
            "total_late_actions":    len(late_actions),
            "total_missed_opps":     len(missed_opportunities),
        },
    }


# ---------------------------------------------------------------------------
# EpisodeStore
# ---------------------------------------------------------------------------

class EpisodeStore:
    """
    Lightweight in-memory recorder that flushes each episode to JSON on finalize().

    Thread-safety: one EpisodeStore instance per process is sufficient for the
    single-threaded inference.py loop. For concurrent use, instantiate per-thread.
    """

    def __init__(self, base_dir: str = "outputs/episodes") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: dict[str, dict] = {}   # session_id → episode dict

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def start_episode(self, session_id: str, task: str, seed: int = 0) -> None:
        """Initialise an empty episode record. Call once after /reset."""
        self._episodes[session_id] = {
            "task":       task,
            "seed":       seed,
            "session_id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "score":      None,
            "steps":      [],
            "post_mortem": {},
        }

    def record_step(
        self,
        session_id: str,
        step: int,
        obs: dict,
        action: str,
        reward: float,
        reasoning: str = "",
    ) -> None:
        """Append one step entry. Call after /step returns the reward."""
        episode = self._episodes.get(session_id)
        if episode is None:
            return   # start_episode was not called — silently skip

        episode["steps"].append({
            "step":      step,
            "action":    action,
            "reward":    round(_safe_float(reward), 4),
            "reasoning": reasoning,
            "summary":   _obs_summary(obs),
        })

    def finalize(self, session_id: str, final_score: float) -> Path | None:
        """
        Compute post-mortem, attach final score, and flush to JSON.

        Returns the Path of the saved file, or None if session_id was not found.
        """
        episode = self._episodes.pop(session_id, None)
        if episode is None:
            return None

        episode["score"]      = round(_safe_float(final_score), 4)
        episode["steps_taken"] = len(episode["steps"])
        episode["post_mortem"] = _compute_post_mortem(episode["steps"])
        episode["finished_at"] = datetime.now(timezone.utc).isoformat()

        # File name: <task>_s<seed>_<short_session_id>.json
        short_id = session_id[:8] if len(session_id) >= 8 else session_id
        filename = f"{episode['task']}_s{episode['seed']}_{short_id}.json"
        out_path = self.base_dir / filename

        try:
            out_path.write_text(json.dumps(episode, indent=2))
            print(f"  [replay] Episode saved → {out_path}")
        except OSError as exc:
            print(f"  [replay] WARNING: could not save episode: {exc}")
            return None

        return out_path


# ---------------------------------------------------------------------------
# Replay helper
# ---------------------------------------------------------------------------

def load_episode(file_path: str | Path) -> dict:
    """Load a saved episode JSON for replay or post-mortem inspection.

    Example
    -------
        episode = load_episode("outputs/episodes/elite_s42_abc123.json")
        for entry in episode["steps"]:
            print(entry["step"], entry["action"], entry["reward"])
        print(episode["post_mortem"])
    """
    return json.loads(Path(file_path).read_text())


def list_episodes(base_dir: str = "outputs/episodes") -> list[Path]:
    """Return all saved episode JSON files sorted by modification time (newest first)."""
    p = Path(base_dir)
    if not p.exists():
        return []
    return sorted(p.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
