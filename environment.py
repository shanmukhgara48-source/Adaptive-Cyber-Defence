"""
environment.py — OpenEnv-compliant CyberDefenseEnv wrapper.

Wraps AdaptiveCyberDefenseEnv and converts all returns to plain Python
dicts so the hackathon validator can work without importing internal types.

API:
    env = CyberDefenseEnv(task="easy", seed=42)
    obs          = env.reset()
    obs, r, done, info = env.step(Action.SCAN)
    full         = env.state()
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# path bootstrap so this file works when run directly
_HERE = Path(__file__).resolve().parent
for _p in (_HERE.parent, _HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from adaptive_cyber_defense.models.action import Action, ActionInput, ACTION_PROFILES
from adaptive_cyber_defense.models.state  import EnvironmentState
from adaptive_cyber_defense.tasks         import EasyTask, MediumTask, HardTask, NightmareTask

_TASK_MAP = {
    "easy":      EasyTask,
    "medium":    MediumTask,
    "hard":      HardTask,
    "nightmare": NightmareTask,
}


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _threat_to_dict(t) -> dict:
    return {
        "id":                   t.id,
        "stage":                t.stage.name,
        "origin_node":          t.origin_node,
        "current_node":         t.current_node,
        "severity":             round(t.severity, 4),
        "detection_confidence": round(t.detection_confidence, 4),
        "is_detected":          t.is_detected,
        "is_contained":         t.is_contained,
        "steps_active":         t.steps_active,
        "attack_type":          getattr(t, "attack_type", "generic"),
        # MITRE ATT&CK — prefer live fields, fall back to stage properties
        "technique_id":         getattr(t, "mitre_technique_id", "") or t.stage.technique_id,
        "technique_name":       getattr(t, "mitre_technique_name", "") or t.stage.technique_name,
        "tactic":               getattr(t, "mitre_tactic", "") or t.stage.tactic,
        "tactic_id":            getattr(t, "mitre_tactic_id", "") or t.stage.tactic_id,
    }


def _asset_to_dict(asset) -> dict:
    return {
        "type":           asset.asset_type.value,
        "health":         round(asset.health, 4),
        "is_compromised": asset.is_compromised,
        "is_isolated":    asset.is_isolated,
        "patch_level":    round(asset.patch_level, 4),
        "criticality":    round(asset.criticality, 4),
        "vulnerability":  round(asset.vulnerability_score(), 4),
    }


def _state_to_obs(env_state: EnvironmentState) -> dict:
    """Convert EnvironmentState → flat dict observation."""
    return {
        "active_threats": [_threat_to_dict(t) for t in env_state.active_threats],
        "network_state":  {nid: _asset_to_dict(a) for nid, a in env_state.assets.items()},
        "resources": {
            "scan_capacity":  round(env_state.resource_availability, 4),
            "response_slots": round(env_state.resource_availability, 4),
        },
        "step":  env_state.time_step,
        "score": round(env_state.episode_score, 4),
        # extra fields for richer agents
        "threat_severity":      round(env_state.threat_severity, 4),
        "network_load":         round(env_state.network_load, 4),
        "detection_confidence": round(env_state.detection_confidence, 4),
        "compromised_nodes":    list(env_state.compromised_nodes),
        "is_terminal":          env_state.is_terminal,
    }


def _best_target(env_state: EnvironmentState) -> Optional[str]:
    """Pick the highest-severity threat's node as a default action target."""
    if env_state.active_threats:
        return max(env_state.active_threats,
                   key=lambda t: t.effective_severity()).current_node
    if env_state.compromised_nodes:
        return env_state.compromised_nodes[0]
    return next(iter(env_state.assets), None)


def _to_action_input(action: Union[Action, str, dict, ActionInput],
                     env_state: EnvironmentState) -> ActionInput:
    """Normalise any action representation into an ActionInput."""
    if isinstance(action, ActionInput):
        return action

    # Resolve action enum
    if isinstance(action, str):
        action_enum = Action[action.upper()]
    elif isinstance(action, Action):
        action_enum = action
    elif isinstance(action, dict):
        action_enum = Action[str(action.get("action", "IGNORE")).upper()]
        target = action.get("target") or action.get("target_node")
        return ActionInput(action=action_enum, target_node=target)
    else:
        action_enum = Action.IGNORE

    # For non-IGNORE actions, supply a sensible default target
    target = None
    if action_enum != Action.IGNORE:
        target = _best_target(env_state)

    return ActionInput(action=action_enum, target_node=target)


# ---------------------------------------------------------------------------
# CyberDefenseEnv
# ---------------------------------------------------------------------------

class CyberDefenseEnv:
    """
    OpenEnv-compliant wrapper around AdaptiveCyberDefenseEnv.

    Differences from the raw env:
      - reset() / step() return plain dicts (not EnvironmentState objects)
      - step() accepts Action enum, action name strings, dicts, or ActionInput
      - state() returns a full structured dict with episode_info sub-dict
    """

    def __init__(self, task: str = "easy", seed: int = 0) -> None:
        task_cls = _TASK_MAP.get(task.lower())
        if task_cls is None:
            raise ValueError(f"Unknown task '{task}'. Choose from: {list(_TASK_MAP)}")
        self._task     = task_cls()
        self._env      = self._task.build_env()
        self._seed     = seed
        self._env_state: Optional[EnvironmentState] = None
        self._done     = False
        self._threats_seen:      int = 0
        self._threats_contained: int = 0

    # ── OpenEnv API ──────────────────────────────────────────────────────────

    def reset(self, seed: Optional[int] = None) -> dict:
        """
        Reset the episode and return the initial observation dict.

        Returns:
            dict with keys: active_threats, network_state, resources,
                            step (=0), score (=0.0), ...
        """
        use_seed = seed if seed is not None else self._seed
        self._env_state = self._env.reset(seed=use_seed)
        self._done = False
        self._threats_seen      = len(self._env_state.active_threats)
        self._threats_contained = 0
        obs = _state_to_obs(self._env_state)
        # Guarantee spec requirements
        assert obs["step"]  == 0,   "reset: step must be 0"
        assert obs["score"] == 0.0, "reset: score must be 0.0"
        return obs

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        """
        Advance one step.

        Args:
            action: Action enum | action name str | dict | ActionInput

        Returns:
            (observation, reward, done, info)
            where observation is a dict, reward ∈ [-1.0, 1.0],
            done is bool, info is dict.
        """
        if self._env_state is None:
            self._env_state = self._env.reset(seed=self._seed)

        action_input = _to_action_input(action, self._env_state)
        next_env_state, reward, done, info = self._env.step(action_input)

        self._env_state = next_env_state
        self._done      = done

        # Track containment across steps
        newly_contained = sum(
            1 for t in next_env_state.active_threats if t.is_contained
        )
        self._threats_contained = max(self._threats_contained, newly_contained)
        self._threats_seen = max(self._threats_seen,
                                 len(next_env_state.active_threats))

        obs = _state_to_obs(next_env_state)

        # Clamp reward to [-1, 1] as required by spec
        reward = float(max(-1.0, min(1.0, reward)))

        # Enrich info with mandatory keys
        info.setdefault("step",               next_env_state.time_step)
        info.setdefault("score",              round(next_env_state.episode_score, 4))
        info.setdefault("threats_contained",  self._threats_contained)

        return obs, reward, bool(done), info

    def state(self) -> dict:
        """
        Return the full current state as a structured dict.

        Returns:
            dict with keys: observation, step, score, done, episode_info
        """
        if self._env_state is None:
            self._env_state = self._env.reset(seed=self._seed)

        s = self._env_state
        critical_healths = [
            a.health for a in s.assets.values() if a.criticality >= 0.7
        ]
        critical_health = (
            sum(critical_healths) / len(critical_healths)
            if critical_healths else 1.0
        )

        return {
            "observation": _state_to_obs(s),
            "step":        s.time_step,
            "score":       round(s.episode_score, 4),
            "done":        self._done,
            "episode_info": {
                "total_threats":       self._threats_seen,
                "threats_contained":   self._threats_contained,
                "critical_health":     round(critical_health, 4),
                "resources_remaining": round(s.resource_availability, 4),
            },
        }

    # ── convenience ─────────────────────────────────────────────────────────

    def recommend(self):
        """Passthrough to underlying env recommendations."""
        return self._env.recommend()
