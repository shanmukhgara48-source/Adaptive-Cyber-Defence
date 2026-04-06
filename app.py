# ─── ARCHITECTURE NOTE ────────────────────────────────────────────────────────
# This file (app.py) is the single HTTP server for the OpenEnv evaluation path.
# It contains a complete, self-contained 5-node simulation used by all API calls.
#
# Design principles shared across the full project:
#   • One grader formula: 0.50×containment + 0.20×health + 0.15×resource + 0.15×speed
#   • One task configuration source: tasks/*.py — derived once into TASK_OVERRIDES
#   • One reward signal: MITRE ATT&CK correct-action lookup (_MITRE_CORRECT_ACTION)
#   • One adaptive attacker: AdaptiveAttacker — strategy overrides flow into effective_config
#   • One network topology: ADJACENCY — used by HTTP simulation and OOP simulation alike
#
# The project also contains an OOP simulation layer (env.py + engines/) validated by
# 760+ tests (test_phase*.py, test_tc*.py). It shares the same task configs, threat
# model, and reward formula as this HTTP layer — both layers are in active use.
# ──────────────────────────────────────────────────────────────────────────────
import random
import logging
import math
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from models import Observation
import importlib.util as _ilu, sys as _sys, os as _os


# ─── TASK CONFIG IMPORTS ──────────────────────────────────────────────────────
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from tasks.nightmare import NightmareTask
from tasks.elite import EliteTask
from tasks.impossible import ImpossibleTask
from grader import (
    TASK_PASSING_SCORES,
    compute_speed_bonus as _compute_speed_bonus,
    compute_grader_score as _grader_formula,
)

TASK_MAP = {
    "easy":       EasyTask,
    "medium":     MediumTask,
    "hard":       HardTask,
    "nightmare":  NightmareTask,
    "elite":      EliteTask,
    "impossible": ImpossibleTask,
}
_aa_spec = _ilu.spec_from_file_location(
    "adaptive_attacker",
    _os.path.join(_os.path.dirname(__file__), "engines", "adaptive_attacker.py"),
)
_aa_mod = _ilu.module_from_spec(_aa_spec)
_sys.modules["adaptive_attacker"] = _aa_mod   # must register before exec for @dataclass
_aa_spec.loader.exec_module(_aa_mod)
AdaptiveAttacker = _aa_mod.AdaptiveAttacker

# ─── DEBUG ────────────────────────────────────────────────────────────────────
import os as _os
DEBUG = _os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.WARNING)
log = logging.getLogger("cyber_defense")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATTACKS = ["phishing", "malware", "ddos", "ransomware", "lateral_movement"]

MITRE_MAP = {
    "phishing": "T1566",
    "malware": "T1204",
    "ddos": "T1499",
    "ransomware": "T1486",
    "lateral_movement": "T1021",
}

# Pure MITRE ATT&CK mitigation lookup — matches openenv.yaml and inference.py prompt exactly.
# phishing/T1566 → block_ip, malware/T1204 → isolate_machine,
# ransomware/T1486 → isolate_machine, ddos/T1499 → patch,
# lateral_movement/T1021 → block_ip
_MITRE_CORRECT_ACTION: dict[str, str] = {
    "phishing":         "block_ip",
    "malware":          "isolate_machine",
    "ransomware":       "isolate_machine",
    "ddos":             "patch",
    "lateral_movement": "block_ip",
}


def _get_correct_action(threat_type: str, severity: float, stage: str) -> str:
    """Return the correct MITRE-aligned mitigation action for a threat type.
    Severity and stage parameters are accepted for API compatibility but not used —
    the mapping is deterministic by type only, matching the spec and agent prompts.
    """
    return _MITRE_CORRECT_ACTION.get(threat_type, "ignore")

EXPLAIN = {
    "phishing": {
        "correct": "Phishing attack detected (T1566). Blocking the source IP prevents credential theft and halts the initial access vector.",
        "wrong": "Phishing requires blocking the source IP. Other mitigations do not stop credential harvesting.",
        "ignore": "Ignoring phishing allows the attacker to harvest credentials — severe health impact.",
    },
    "malware": {
        "correct": "Malware execution detected (T1204). Isolating the machine cuts off C2 communication and stops lateral spread.",
        "wrong": "Malware requires machine isolation. Blocking IPs or patching alone does not stop active execution.",
        "ignore": "Ignoring active malware allows it to spread to adjacent nodes — critical health loss.",
    },
    "ddos": {
        "correct": "DDoS attack detected (T1499). Patching the exposed service mitigates the volumetric impact.",
        "wrong": "DDoS requires patching the target service. Isolation or IP blocking does not absorb volumetric traffic.",
        "ignore": "Ignoring a DDoS degrades service availability rapidly.",
    },
    "ransomware": {
        "correct": "Ransomware detected (T1486). Isolating the machine prevents encryption from spreading to network shares.",
        "wrong": "Ransomware requires machine isolation to stop file encryption propagation.",
        "ignore": "Ignoring ransomware allows full disk encryption — catastrophic health loss.",
    },
    "lateral_movement": {
        "correct": "Lateral movement detected (T1021). Blocking the attacker IP stops traversal to new hosts.",
        "wrong": "Lateral movement requires IP blocking to cut the attacker's pivot path.",
        "ignore": "Ignoring lateral movement allows the attacker to compromise additional nodes.",
    },
}

TOTAL_NODES = 5
NODES = [f"node_{i}" for i in range(1, TOTAL_NODES + 1)]

# Network topology: linear chain with a hub.
# node_1 — node_2 — node_3 — node_4 — node_5
#                 \— node_5 (also adjacent to node_2 for a second path)
# Lateral movement can only spread along edges — attackers must traverse the network.
ADJACENCY: dict[str, set] = {
    "node_1": {"node_2"},
    "node_2": {"node_1", "node_3", "node_5"},
    "node_3": {"node_2", "node_4"},
    "node_4": {"node_3", "node_5"},
    "node_5": {"node_4", "node_2"},
}

VALID_ACTIONS = frozenset(
    ["block_ip", "isolate_machine", "patch", "ignore"]
    + [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)]
)

MAX_ACTION_LEN = 64
MAX_REWARD = 2.0
MIN_REWARD = -2.0

TECHNIQUE_DEFAULTS = {
    "phishing":         ("T1566", "Phishing",                       "Initial Access"),
    "malware":          ("T1204", "User Execution",                  "Execution"),
    "ddos":             ("T1499", "Endpoint Denial of Service",      "Impact"),
    "ransomware":       ("T1486", "Data Encrypted for Impact",       "Impact"),
    "lateral_movement": ("T1021", "Remote Services",                 "Lateral Movement"),
}

# ─── BEHAVIORAL IOC PROFILES ─────────────────────────────────────────────────
# Each value is a (min, max) integer range for the spawned value.
# Profiles are designed with non-overlapping high-signal axes so a reasoning
# agent can infer the threat class without being given the type directly:
#   DDoS        → uniquely extreme packets_per_second (900-9500)
#   Ransomware  → uniquely extreme outbound_data_bytes (9000-48000)
#   Lat. Move.  → uniquely high lateral_connection_count (12-38)
#   Phishing    → high failed_auth_attempts + low everything else
#   Malware     → high unusual_process_count + moderate outbound
_IOC_PROFILES: dict[str, dict] = {
    "phishing": {
        "packets_per_second":       (1, 8),
        "failed_auth_attempts":     (45, 200),
        "outbound_data_bytes":      (80, 600),
        "lateral_connection_count": (0, 2),
        "unusual_process_count":    (1, 4),
        "spread_rate":              (0.05, 0.25),
        "is_persistent":            False,
    },
    "malware": {
        "packets_per_second":       (15, 65),
        "failed_auth_attempts":     (0, 8),
        "outbound_data_bytes":      (900, 4500),
        "lateral_connection_count": (2, 10),
        "unusual_process_count":    (6, 20),
        "spread_rate":              (0.30, 0.65),
        "is_persistent":            True,
    },
    "ddos": {
        "packets_per_second":       (900, 9500),
        "failed_auth_attempts":     (0, 5),
        "outbound_data_bytes":      (0, 120),
        "lateral_connection_count": (0, 2),
        "unusual_process_count":    (1, 3),
        "spread_rate":              (0.0, 0.08),
        "is_persistent":            False,
    },
    "ransomware": {
        "packets_per_second":       (5, 25),
        "failed_auth_attempts":     (8, 45),
        "outbound_data_bytes":      (9000, 48000),
        "lateral_connection_count": (6, 20),
        "unusual_process_count":    (10, 28),
        "spread_rate":              (0.55, 0.95),
        "is_persistent":            True,
    },
    "lateral_movement": {
        "packets_per_second":       (20, 90),
        "failed_auth_attempts":     (25, 115),
        "outbound_data_bytes":      (200, 1200),
        "lateral_connection_count": (12, 38),
        "unusual_process_count":    (3, 10),
        "spread_rate":              (0.50, 0.90),
        "is_persistent":            True,
    },
}

# False-positive profile: ambiguous low-signal indicators — look real but weak
_FP_IOC_PROFILE: dict = {
    "packets_per_second":       (2, 18),
    "failed_auth_attempts":     (3, 22),
    "outbound_data_bytes":      (50, 350),
    "lateral_connection_count": (0, 4),
    "unusual_process_count":    (1, 5),
    "spread_rate":              (0.0, 0.15),
    "is_persistent":            False,
}


def _spawn_iocs(t_type: str, rng: random.Random, is_fp: bool = False) -> dict:
    """Generate initial behavioral IOC values for a new threat.
    These are stored on the internal threat dict and exposed in _visible_threats.
    The threat type itself is NOT exposed — agents must infer it from these signals.
    """
    profile = _FP_IOC_PROFILE if is_fp else _IOC_PROFILES.get(t_type, _IOC_PROFILES["malware"])
    pps_lo, pps_hi = profile["packets_per_second"]
    auth_lo, auth_hi = profile["failed_auth_attempts"]
    ob_lo, ob_hi   = profile["outbound_data_bytes"]
    lat_lo, lat_hi = profile["lateral_connection_count"]
    proc_lo, proc_hi = profile["unusual_process_count"]
    sr_lo, sr_hi   = profile["spread_rate"]
    return {
        "packets_per_second":       rng.randint(pps_lo, pps_hi),
        "failed_auth_attempts":     rng.randint(auth_lo, auth_hi),
        "outbound_data_bytes":      rng.randint(ob_lo, ob_hi),
        "lateral_connection_count": rng.randint(lat_lo, lat_hi),
        "unusual_process_count":    rng.randint(proc_lo, proc_hi),
        "spread_rate":              round(rng.uniform(sr_lo, sr_hi), 3),
        "is_persistent":            profile["is_persistent"],
        "affected_node_count":      1,
        "detection_confidence":     round(
            rng.uniform(0.3, 0.7) if is_fp else rng.uniform(0.6, 1.0), 3
        ),
    }


def _evolve_iocs(t: dict, rng: random.Random) -> None:
    """Grow IOC signals each step to simulate escalating threat activity.
    Growth is intentionally small to preserve profile separability across the episode.
    outbound_data_bytes is NOT grown — its initial value is the primary ransomware signal.
    """
    if t.get("is_false_positive"):
        return  # FP signals are static noise — they do not escalate
    # Auth failures accumulate gradually
    t["failed_auth_attempts"] = t.get("failed_auth_attempts", 0) + rng.randint(0, 2)
    # Lateral connections grow for high-spread threats
    if t.get("spread_rate", 0.0) > 0.3:
        t["lateral_connection_count"] = t.get("lateral_connection_count", 0) + rng.randint(0, 2)
    # Process count ticks up as payload executes
    t["unusual_process_count"] = t.get("unusual_process_count", 0) + rng.randint(0, 1)
    # Detection confidence increases as threat becomes more active
    t["detection_confidence"] = round(
        min(1.0, t.get("detection_confidence", 0.7) + 0.02), 3
    )

TASKS = [
    {"id": 1, "difficulty": "easy",       "passing_score": TASK_PASSING_SCORES["easy"],       "goal": "Three simultaneous attacks. High detection, generous resources. Contain all before lateral spread."},
    {"id": 2, "difficulty": "medium",     "passing_score": TASK_PASSING_SCORES["medium"],     "goal": "Two intrusions with limited resources, FP noise. Prioritise threats."},
    {"id": 3, "difficulty": "hard",       "passing_score": TASK_PASSING_SCORES["hard"],       "goal": "APT across 5 nodes. Low detection, scarce resources, fast progression."},
    {"id": 4, "difficulty": "nightmare",  "passing_score": TASK_PASSING_SCORES["nightmare"],  "goal": "Nation-state APT. Near-zero detection, 15 steps. Designed for frontier LLMs."},
    {"id": 5, "difficulty": "elite",      "passing_score": TASK_PASSING_SCORES["elite"],      "goal": "Persistent threat with insider access. All nodes pre-compromised. Kill chain advances every step."},
    {"id": 6, "difficulty": "impossible", "passing_score": TASK_PASSING_SCORES["impossible"], "goal": "AI-driven attacker with perfect counter-strategy. Exists to show environment has no ceiling."},
]


# Per-task config derived from task classes — all difficulty params now live there.
# age_visibility_threshold is HTTP-API-specific (not in TaskConfig).
def _derive_task_overrides() -> dict:
    out = {}
    for name, cls in TASK_MAP.items():
        cfg = cls.config
        out[name] = {
            "threat_count":              cfg.initial_threat_count,
            "max_steps":                 cfg.max_steps,
            "false_negative_rate":       cfg.false_negative_rate,
            "base_detection_prob":       cfg.base_detection_prob,
            "attack_progression_prob":   cfg.attack_progression_prob,
            "lateral_spread_base_prob":  cfg.lateral_spread_base_prob,
            "health_degradation_rate":   cfg.health_degradation_rate,
            # Previously missing — these are critical for difficulty scaling
            "resource_per_step":         cfg.resource_per_step,
            "natural_severity_growth":   cfg.natural_severity_growth,
            "false_positive_rate":       cfg.false_positive_rate,
            "passing_score":             cfg.passing_score,
            # Visibility threshold: nightmare/elite/impossible use 8 (longer hidden window)
            "age_visibility_threshold":  8 if name in ("nightmare", "elite", "impossible") else 5,
        }
    return out

TASK_OVERRIDES = _derive_task_overrides()

# Action translation: app.py lowercase → attacker uppercase
ACTION_TRANSLATION = {
    "block_ip":        "BLOCK_IP",
    "isolate_machine": "ISOLATE_NODE",
    "patch":           "PATCH_VULNERABILITY",
    "ignore":          "IGNORE",
}

def translate_action(raw_action: str) -> str:
    """Translate a lowercase app.py action to the uppercase name AdaptiveAttacker expects."""
    if raw_action.startswith("scan"):
        return "SCAN"
    return ACTION_TRANSLATION.get(raw_action, raw_action.upper())

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Adaptive Cyber Defense", version="2.0.0")

# ─── RED TEAM ─────────────────────────────────────────────────────────────────
# AdaptiveAttacker is instantiated per-session in /reset to prevent cross-session contamination.
_ATTACKER_SEED = int(_os.getenv("ATTACKER_SEED", "42"))

# ─── SESSION ──────────────────────────────────────────────────────────────────
# Each call to /reset creates an isolated Session.  Concurrent users/judges
# each hold their own session_id and never share state.

@dataclass
class Session:
    """All mutable per-episode state, isolated per user/judge."""
    task_name:   str
    task_config: dict
    # effective_config is task_config modified by the adaptive attacker's strategy overrides.
    # e.g. APT strategy reduces detection_prob by 60%, slows stage progression 3×.
    # Code that governs simulation mechanics uses effective_config, not task_config.
    effective_config: dict = field(default_factory=dict)
    state:       dict = field(default_factory=dict)
    history:     list = field(default_factory=list)
    episode_history: list = field(default_factory=list)
    episode_actions_taken: list = field(default_factory=list)
    episode_rewards:       list = field(default_factory=list)
    threats_detected:      set  = field(default_factory=set)
    threats_contained:     set  = field(default_factory=set)
    false_positive_actions: int = 0
    containment_events: list = field(default_factory=list)
    attack_plan: dict = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    attacker: object = field(default_factory=lambda: None)


# Session store — keyed by UUID string.
_SESSIONS: OrderedDict[str, Session] = OrderedDict()
# Maximum live sessions (evict oldest after this limit to prevent memory growth)
_MAX_SESSIONS = 256
# Tracks the most-recently created session ID so callers that omit session_id
# (e.g. the adversarial test suite helpers) automatically use the latest session.
_LATEST_SID: str | None = None

def _evict_oldest_sessions():
    """Keep session count under _MAX_SESSIONS by dropping the oldest entries."""
    global _LATEST_SID
    while len(_SESSIONS) >= _MAX_SESSIONS:
        oldest = next(iter(_SESSIONS))
        if oldest == _LATEST_SID:
            _LATEST_SID = None
        del _SESSIONS[oldest]


def _get_session(session_id: str | None) -> Session | None:
    """Return the session for the given id.
    When session_id is None or empty, falls back to the most-recently-created
    session so that clients that omit session_id still get a valid session.
    Moves accessed sessions to end for LRU eviction ordering.
    """
    if not session_id:
        # Fall back to latest session when no id is provided
        if _LATEST_SID and _LATEST_SID in _SESSIONS:
            _SESSIONS.move_to_end(_LATEST_SID)
            return _SESSIONS[_LATEST_SID]
        return None
    sid = session_id.strip()
    sess = _SESSIONS.get(sid)
    if sess is not None:
        _SESSIONS.move_to_end(sid)
    return sess


# ─── STATE HELPERS (session-scoped) ──────────────────────────────────────────

# Initial severity ranges per attack type — vary so severity-based logic fires
_SEVERITY_RANGES = {
    "phishing":         (0.3, 0.6),
    "malware":          (0.5, 0.8),
    "ransomware":       (0.7, 1.0),
    "ddos":             (0.4, 0.7),
    "lateral_movement": (0.6, 0.9),
}


def _initial_severity(t_type: str, rng: random.Random) -> float:
    lo, hi = _SEVERITY_RANGES.get(t_type, (0.4, 0.7))
    return round(rng.uniform(lo, hi), 3)


def _compute_grader_score(sess: "Session") -> float:
    """Extract components from session state, delegate formula to grader.py."""
    s = sess.state
    _contained = sum(1 for t in s["threats"] if t.get("contained") and not t.get("is_false_positive"))
    _total = max(1, sum(1 for t in s["threats"] if not t.get("is_false_positive")))
    containment_rate = _contained / _total
    critical_health = s["system_health"] / 100.0
    _task_budget = max(0.01, sess.task_config.get("resource_per_step", 1.0))
    _cost_raw = {"isolate_machine": 0.4, "block_ip": 0.3, "patch": 0.3}
    _total_spent = sum(_cost_raw.get(a, 0.2 if a.startswith("scan") else 0.0) for a in sess.episode_actions_taken)
    _total_budget = max(0.01, _task_budget * sess.task_config.get("max_steps", 50))
    resource_efficiency = max(0.0, 1.0 - _total_spent / _total_budget)
    speed_bonus = _compute_speed_bonus(sess.containment_events)
    return _grader_formula(containment_rate, critical_health, resource_efficiency, speed_bonus)


def _make_threats_fixed(task_config: dict, rng: random.Random, attacker=None) -> list:
    """Make threats for a new episode using the given task config."""
    threats = []
    count = task_config.get("threat_count", 3)
    for idx in range(count):
        t_type = rng.choice(ATTACKS)
        node   = rng.choice(NODES)
        iocs = _spawn_iocs(t_type, rng)
        threats.append({
            "id":              f"alert_{node}_{idx}",   # opaque — does not reveal threat type
            "type":            t_type,
            "original_type":   t_type,  # preserved even after stage escalation — INTERNAL ONLY
            "node":            node,
            "visible":         False,
            "age":             0,
            "stage":           "initial",
            "escalated":       False,
            "contained":       False,
            "spread_attempted": False,
            "mitre_id":        MITRE_MAP[t_type],
            "severity":        _initial_severity(t_type, rng),
            **iocs,
        })
    # If attacker has a strategy, bias threat types accordingly
    if attacker is not None:
        strategy = getattr(attacker, 'current_strategy', None)
        _STRATEGY_BIAS = {
            "APT":             ["phishing", "malware"],
            "RANSOMWARE":      ["ransomware"],
            "INSIDER_THREAT":  ["lateral_movement"],
            "SUPPLY_CHAIN":    ["malware"],
            "ZERO_DAY":        ["malware", "ransomware"],
        }
        preferred = _STRATEGY_BIAS.get(strategy, [])
        if preferred:
            for t in threats:
                # 60% chance to override type with a strategy-preferred type
                if rng.random() < 0.6:
                    new_type = rng.choice(preferred)
                    t["type"] = new_type
                    t["original_type"] = new_type
                    t["mitre_id"] = MITRE_MAP[new_type]
                    t["severity"] = _initial_severity(new_type, rng)
                    # Re-spawn IOCs for the overridden type; keep opaque id
                    new_iocs = _spawn_iocs(new_type, rng)
                    t.update(new_iocs)
    return threats


def _fresh_state(task_config: dict, rng: random.Random, attacker=None) -> dict:
    return {
        "threats": _make_threats_fixed(task_config, rng, attacker),
        "scanned_nodes": set(),
        "system_health": 100,
        "score": 0.0,
        "step": 0,
        "done": False,
        "false_positives_seen": 0,
    }


def _do_reset_session(sess: Session) -> None:
    """Reset all mutable fields on an existing Session object in-place."""
    sess.state                  = _fresh_state(sess.task_config, sess.rng, sess.attacker)
    sess.history                = []
    sess.episode_history        = []
    sess.episode_actions_taken  = []
    sess.episode_rewards        = []
    sess.threats_detected       = set()
    sess.threats_contained      = set()
    sess.false_positive_actions = 0
    sess.containment_events     = []
    sess.attack_plan            = {}


def _validate_session_state(sess: Session) -> None:
    s = sess.state
    try:
        assert isinstance(s["threats"], list)
        assert isinstance(s["scanned_nodes"], set)
        assert isinstance(s["system_health"], (int, float))
        assert isinstance(s["score"], (int, float))
        assert isinstance(s["step"], int)
        assert isinstance(s["done"], bool)
        if not math.isfinite(s["system_health"]):
            raise ValueError("system_health non-finite")
        if not math.isfinite(s["score"]):
            raise ValueError("score non-finite")
    except Exception as e:
        log.error(f"State corruption in session — auto-resetting: {e}")
        _do_reset_session(sess)
        raise RuntimeError("session_state_reset") from e


def _clamp_health(sess: Session) -> None:
    sess.state["system_health"] = int(
        max(0, min(100, round(sess.state["system_health"])))
    )


def _clamp_reward(r: float) -> float:
    if not math.isfinite(r):
        raise ValueError(f"non-finite reward passed to _clamp_reward: {r!r}")
    normalized_reward = (float(r) + 2.0) / 4.0
    return max(0.0, min(1.0, normalized_reward))


def _clamp_score(sess: Session) -> None:
    if not math.isfinite(sess.state["score"]):
        sess.state["score"] = 0.0
    sess.state["score"] = max(0.0, min(1.0, sess.state["score"]))


# ─── LOGIC ────────────────────────────────────────────────────────────────────
def enrich_threat(threat: dict) -> dict:
    """Ensure every visible threat has all required fields with safe defaults."""
    if not isinstance(threat, dict):
        return threat

    t_type = threat.get("type", "malware")

    tech_id, tech_name, tactic = TECHNIQUE_DEFAULTS.get(
        t_type,
        ("T1204", "User Execution", "Execution"),
    )

    threat["id"]    = str(threat.get("id", f"{t_type}_{threat.get('node', 'unknown')}"))
    threat["type"]  = str(t_type)
    threat["node"]  = str(threat.get("node", "node_1"))
    threat["stage"] = str(threat.get("stage", "initial"))

    threat["age"]      = int(threat.get("age", 0))
    threat["severity"] = float(threat.get("severity", 0.5))

    threat["technique_id"]   = threat.get("technique_id")   or tech_id
    threat["technique_name"] = threat.get("technique_name") or tech_name
    threat["tactic"]         = threat.get("tactic")         or tactic

    threat["detection_confidence"] = float(threat.get("detection_confidence", 1.0))

    return threat


def _update_visibility(sess: Session) -> None:
    """Auto-reveal threats based on age or lateral movement, gated by task difficulty.
    Uses effective_config so attacker detection_evasion actually reduces visibility."""
    eff          = sess.effective_config if sess.effective_config else sess.task_config
    age_thresh   = eff.get("age_visibility_threshold", sess.task_config.get("age_visibility_threshold", 5))
    detect_prob  = eff.get("base_detection_prob", 1.0)
    fn_rate      = eff.get("false_negative_rate", sess.task_config.get("false_negative_rate", 0.0))
    for t in sess.state["threats"]:
        if t.get("contained") or t.get("visible"):
            continue
        if t.get("is_false_positive"):
            continue  # FPs are immediately visible when created
        if t["stage"] == "lateral_movement":
            if sess.rng.random() > fn_rate:
                t["visible"] = True
        elif t["age"] >= age_thresh:
            if sess.rng.random() < detect_prob and sess.rng.random() > fn_rate:
                t["visible"] = True
    _maybe_generate_false_positive(sess)


def _maybe_generate_false_positive(sess: Session) -> None:
    """With probability false_positive_rate, add a ghost (false positive) threat."""
    fp_rate = sess.task_config.get("false_positive_rate", 0.0)
    if fp_rate <= 0:
        return
    if sess.rng.random() > fp_rate:
        return
    # Don't create more than 2 active false positives at a time
    existing_fps = sum(1 for t in sess.state["threats"] if t.get("is_false_positive") and t["visible"] and not t.get("contained"))
    if existing_fps >= 2:
        return
    fp_type = sess.rng.choice(ATTACKS)
    fp_node = sess.rng.choice(NODES)
    fp_idx = len(sess.state["threats"])
    fp_iocs = _spawn_iocs(fp_type, sess.rng, is_fp=True)
    fp_threat = {
        "id": f"alert_{fp_node}_{fp_idx}",   # opaque — does not reveal FP or threat type
        "type": fp_type,
        "original_type": fp_type,
        "node": fp_node,
        "visible": True,  # immediately visible — that's the point, it "appears" as an alert
        "age": 0,
        "stage": "initial",
        "escalated": False,
        "contained": False,
        "spread_attempted": False,
        "mitre_id": MITRE_MAP[fp_type],
        "severity": round(sess.rng.uniform(0.2, 0.5), 3),  # low-moderate severity
        "is_false_positive": True,
        **fp_iocs,
    }
    sess.state["threats"].append(fp_threat)
    sess.state["false_positives_seen"] = sess.state.get("false_positives_seen", 0) + 1


def _age_threats(sess: Session) -> None:
    # Use effective_config (task_config modified by adaptive attacker overrides).
    # Falls back to task_config for fields not overridden (e.g. natural_severity_growth).
    eff = sess.effective_config if sess.effective_config else sess.task_config
    prog_prob = eff.get("attack_progression_prob", 0.15)
    sev_growth = eff.get("natural_severity_growth", 0.05)
    for t in sess.state["threats"]:
        if t.get("is_false_positive"):
            continue  # false positives don't age, escalate, or spread
        if not t.get("contained"):
            t["age"] += 1
            # Severity grows each step — makes severity-based action logic meaningful
            t["severity"] = round(min(1.0, t.get("severity", 0.5) + sev_growth), 3)
            # Evolve behavioral IOC signals to simulate escalating threat activity
            _evolve_iocs(t, sess.rng)
            if t["stage"] == "initial" and sess.rng.random() < prog_prob:
                # Stage escalates to lateral_movement but original type is PRESERVED.
                # The correct mitigation is always determined by original_type so the
                # agent is never penalized for correctly identifying the threat before escalation.
                t["stage"] = "lateral_movement"
                # Keep t["type"] = original_type (do NOT mutate it)
                if "original_type" not in t:
                    t["original_type"] = t["type"]
                t["escalated"] = True
            # Lateral spread: when a threat escalates to lateral_movement,
            # it may spread to a new node with probability lateral_spread_base_prob
            if t["stage"] == "lateral_movement" and t.get("escalated") and not t.get("spread_attempted"):
                t["spread_attempted"] = True  # only attempt spread once per threat
                spread_prob = eff.get("lateral_spread_base_prob", 0.0)
                if spread_prob > 0 and sess.rng.random() < spread_prob:
                    # Cap total threats at 8 to prevent memory/performance issues
                    if len(sess.state["threats"]) < 8:
                        # Topology-constrained spread: only reach adjacent nodes.
                        # Attacker must traverse the network graph — not teleport.
                        occupied = {th["node"] for th in sess.state["threats"] if not th.get("contained")}
                        current_node = t.get("node", "node_1")
                        adjacent = ADJACENCY.get(current_node, set(NODES))
                        free_nodes = [n for n in adjacent if n not in occupied]
                        if free_nodes:
                            new_node = sess.rng.choice(free_nodes)
                            new_type = "lateral_movement"
                            new_idx = len(sess.state["threats"])
                            child_iocs = _spawn_iocs(new_type, sess.rng)
                            sess.state["threats"].append({
                                "id": f"alert_{new_node}_{new_idx}",   # opaque spread child
                                "type": new_type,
                                "original_type": new_type,
                                "node": new_node,
                                "visible": False,
                                "age": 0,
                                "stage": "initial",
                                "escalated": False,
                                "contained": False,
                                "spread_attempted": False,
                                "mitre_id": MITRE_MAP[new_type],
                                "severity": _initial_severity(new_type, sess.rng),
                                **child_iocs,
                            })
                            # Increment parent's affected_node_count to signal spread
                            t["affected_node_count"] = t.get("affected_node_count", 1) + 1


def _visible_threats(sess: Session) -> list:
    """Build the agent-facing threat list.

    Exposes only behavioral IOC signals — never the internal threat type,
    original_type, mitre_id, technique_id, technique_name, or tactic.
    Agents must infer the threat class (and therefore the correct action)
    from the observable signals, not from a direct type lookup.
    """
    out = []
    for t in sess.state["threats"]:
        if t["visible"] and not t.get("contained"):
            out.append({
                "id":                     str(t.get("id", "unknown")),
                "node":                   str(t["node"]),
                "stage":                  str(t["stage"]),
                "age":                    int(t["age"]),
                "dwell_time_steps":       int(t["age"]),      # alias for clarity
                "escalated":              bool(t.get("escalated", False)),
                "severity":               round(float(t.get("severity", 0.5)), 3),
                "detection_confidence":   round(float(t.get("detection_confidence", 0.8)), 3),
                "is_persistent":          bool(t.get("is_persistent", False)),
                "spread_rate":            round(float(t.get("spread_rate", 0.0)), 3),
                "affected_node_count":    int(t.get("affected_node_count", 1)),
                "packets_per_second":     int(t.get("packets_per_second", 0)),
                "failed_auth_attempts":   int(t.get("failed_auth_attempts", 0)),
                "outbound_data_bytes":    int(t.get("outbound_data_bytes", 0)),
                "lateral_connection_count": int(t.get("lateral_connection_count", 0)),
                "unusual_process_count":  int(t.get("unusual_process_count", 0)),
            })
    return out


def _obs(sess: Session) -> dict:
    scanned = sess.state["scanned_nodes"]
    hidden_count = sum(
        1 for t in sess.state["threats"] if not t["visible"] and not t.get("contained")
    )
    score = round(sess.state["score"], 4)
    # Single authoritative grader score — same formula used by /analytics and on_episode_end
    grader = _compute_grader_score(sess)
    _contained = sum(1 for t in sess.state["threats"] if t.get("contained") and not t.get("is_false_positive"))
    _total = max(1, sum(1 for t in sess.state["threats"] if not t.get("is_false_positive")))
    _containment_rate = round(_contained / _total, 4)
    _critical_health = round(sess.state["system_health"] / 100.0, 4)
    _task_budget = max(0.01, sess.task_config.get("resource_per_step", 1.0))
    _cost_raw = {"isolate_machine": 0.4, "block_ip": 0.3, "patch": 0.3}
    _total_spent = sum(_cost_raw.get(a, 0.2 if a.startswith("scan") else 0.0) for a in sess.episode_actions_taken)
    _total_budget = max(0.01, _task_budget * sess.task_config.get("max_steps", 50))
    _resources_remaining = round(max(0.0, 1.0 - _total_spent / _total_budget), 3)
    _speed_bonus = _compute_speed_bonus(sess.containment_events)
    return {
        "visible_threats":  _visible_threats(sess),
        "hidden_threat_count": hidden_count,
        "scan_coverage":    round(len(scanned) / TOTAL_NODES, 2),
        "system_health":    sess.state["system_health"],
        "score":            score,
        "grader_score":     grader,   # grader formula: 0.50×contain+0.20×health+0.15×resource+0.15×speed
        "step":             sess.state["step"],
        "done":             sess.state["done"],
        "episode_info": {
            "total_threats": sum(1 for t in sess.state["threats"] if not t.get("is_false_positive")),
            "threats_contained": _contained,
            "containment_rate": _containment_rate,
            "critical_health": _critical_health,
            "resources_remaining": _resources_remaining,
            "speed_bonus": _speed_bonus,
            "containment_events": list(sess.containment_events),
            "false_positives_seen": sess.state.get("false_positives_seen", 0),
            "false_positives_acted_on": sess.false_positive_actions,
            "grader_breakdown": {
                "containment_rate": _containment_rate,
                "critical_health": _critical_health,
                "resource_efficiency": _resources_remaining,
                "speed_bonus": _speed_bonus,
                "weighted_score": grader,
                "formula": "0.50×contain + 0.20×health + 0.15×resource + 0.15×speed",
                # score (running reward mean) vs grader_score (formula above) are distinct:
                # score tracks per-step reward signal for RL agents;
                # grader_score is the authoritative episode quality metric for ranking.
                "score_label": "running_reward_mean",
                "grader_score_label": "episode_quality_formula",
            },
            "network_topology": {
                # Live graph state — edges are fixed; node_status reflects current compromise/scan state.
                "nodes": NODES,
                "edges": {k: sorted(v) for k, v in ADJACENCY.items()},
                "node_status": {
                    n: (
                        "compromised" if any(
                            t["node"] == n and t.get("stage") == "lateral_movement"
                            and not t.get("contained")
                            for t in sess.state["threats"]
                        ) else "scanned" if n in sess.state["scanned_nodes"]
                        else "unknown"
                    )
                    for n in NODES
                },
            },
        },
    }


def _build_reason(action: str, matched: bool, threat_type: str | None, early: bool) -> tuple[str, float]:
    """Return (reason_string, confidence)."""
    if action.startswith("scan"):
        return ("Scanning reveals hidden threats before they escalate. Essential under partial observability.", 0.85)

    if threat_type is None:
        return ("No visible threat to act on. Scanning unexplored nodes is recommended.", 0.60)

    ex = EXPLAIN.get(threat_type, {})
    if matched:
        base_conf = 0.92
        reason = ex.get("correct", f"Correct mitigation for {threat_type} ({MITRE_MAP.get(threat_type, '')}).")
        if early:
            reason += " Early neutralization bonus applied."
            base_conf = min(1.0, base_conf + 0.05)
        return (reason, base_conf)
    elif action == "ignore":
        return (ex.get("ignore", f"Ignoring {threat_type} allows escalation."), 0.20)
    else:
        return (ex.get("wrong", f"Wrong mitigation for {threat_type}. Check MITRE technique {MITRE_MAP.get(threat_type, '')}."), 0.35)


def safe_response(obs, action, reward=0.0, reason="", confidence=0.0, error=None):
    score = obs.get("score", 0.0)
    resp = {
        "action":           action,
        "reward":           round(float(reward), 3),
        "visible_threats":  obs.get("visible_threats", []),
        "hidden_threat_count": obs.get("hidden_threat_count", TOTAL_NODES),
        "scan_coverage":    obs.get("scan_coverage", 0.0),
        "system_health":    obs.get("system_health", 100),
        "score":            score,
        "grader_score":     obs.get("grader_score", score),
        "step":             obs.get("step", 0),
        "done":             obs.get("done", False),
        "reason":           reason,
        "confidence":       round(float(confidence), 2),
    }
    # Propagate grader_breakdown so every /step response carries full score decomposition.
    ep = obs.get("episode_info") or {}
    if ep.get("grader_breakdown"):
        resp["grader_breakdown"] = ep["grader_breakdown"]
    if error is not None:
        resp["error"] = error
    return resp


# ─── EXCEPTION HANDLERS ───────────────────────────────────────────────────────
_EMPTY_OBS: dict = {
    "visible_threats": [], "hidden_threat_count": TOTAL_NODES,
    "scan_coverage": 0.0, "system_health": 100,
    "score": 0.0, "grader_score": 0.0, "step": 0, "done": False,
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if DEBUG:
        log.debug(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=200,
        content=safe_response(_EMPTY_OBS, action="", reward=0.0,
                               reason="Invalid input received. Action rejected.",
                               confidence=0.0, error="invalid action"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content=safe_response(_EMPTY_OBS, action="", reward=0.0,
                               reason="Internal error. State preserved.",
                               confidence=0.0, error="internal error"),
    )


# ─── INPUT MODELS ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task:       str = "easy"
    seed:       int = 0
    session_id: str | None = None   # optional; omit for auto-generated UUID


class StepRequest(BaseModel):
    """Extended step request that adds optional session_id."""
    action:     str
    session_id: str | None = None   # omit to use the most-recently-reset session

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v):
        if not isinstance(v, str):
            v = str(v)
        if len(v) > MAX_ACTION_LEN:
            v = v[:MAX_ACTION_LEN]
        return v


class StateRequest(BaseModel):
    session_id: str | None = None


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Adaptive Cyber Defense API v2.0 — OpenEnv compatible"}


@app.get("/tasks")
def get_tasks():
    return TASKS


@app.get("/history")
def get_history(session_id: str | None = None):
    sess = _get_session(session_id)
    if sess is None:
        return {"error": "session_id required. Call /reset first.", "episode_steps": [], "total_steps": 0}
    return {
        "episode_steps": sess.episode_history,
        "total_steps": len(sess.episode_history),
        "total_reward": round(sum(s["reward"] for s in sess.episode_history), 4),
        "final_status": "done" if sess.episode_history and sess.episode_history[-1]["done"] else "in_progress",
        "session_id": session_id,
    }


@app.get("/reset")
@app.get("/reset/")
@app.post("/reset")
@app.post("/reset/")
def reset(req: ResetRequest = None):
    task_name  = (req.task.lower().strip() if req and req.task else "easy")
    if task_name not in TASK_OVERRIDES:
        task_name = "easy"
    task_cfg   = TASK_OVERRIDES[task_name]
    seed       = int(req.seed) if req and req.seed is not None else random.randint(0, 2**31 - 1)

    # Determine session_id: use caller-provided or generate a new UUID
    sid = (req.session_id.strip() if req and req.session_id else None) or str(uuid.uuid4())

    # Evict old sessions if at capacity, then create or overwrite
    _evict_oldest_sessions()
    sess = Session(task_name=task_name, task_config=task_cfg)
    sess.rng = random.Random(seed)
    sess.attacker = AdaptiveAttacker(seed=_ATTACKER_SEED)
    _do_reset_session(sess)
    sess.attack_plan = sess.attacker.on_episode_start()

    # Apply adaptive attacker config overrides to create effective simulation parameters.
    # dwell_time_multiplier slows/speeds stage progression.
    # detection_evasion reduces detection probability.
    # spread_rate scales lateral spread probability.
    strategy = sess.attacker.current_strategy
    attacker_cfg = sess.attacker.get_attack_config_override(strategy)
    dwell  = attacker_cfg.get("dwell_time_multiplier", 1.0)
    evasion = attacker_cfg.get("detection_evasion", 0.0)
    spread  = attacker_cfg.get("spread_rate", 1.0)
    sess.effective_config = dict(task_cfg)
    sess.effective_config["attack_progression_prob"]  = round(
        task_cfg["attack_progression_prob"] * dwell, 4
    )
    sess.effective_config["base_detection_prob"] = round(
        task_cfg["base_detection_prob"] * (1.0 - evasion), 4
    )
    sess.effective_config["lateral_spread_base_prob"] = round(
        task_cfg["lateral_spread_base_prob"] * spread, 4
    )

    _SESSIONS[sid] = sess
    global _LATEST_SID
    _LATEST_SID = sid

    obs = _obs(sess)
    obs["task"]              = task_name
    obs["session_id"]        = sid          # always returned so callers can track it
    obs["attacker_strategy"] = strategy
    obs["attacker_config"]   = {            # expose what strategy changed
        "strategy":               strategy,
        "dwell_time_multiplier":  dwell,
        "detection_evasion":      evasion,
        "spread_rate":            spread,
    }
    # attack_plan kept internally for AdaptiveAttacker but never exposed to agents
    return obs


@app.get("/state", response_model=Observation)
@app.get("/state/", response_model=Observation)
def get_state_get(session_id: str | None = None):
    return _get_state_impl(session_id)


@app.post("/state", response_model=Observation)
@app.post("/state/", response_model=Observation)
def get_state_post(req: StateRequest = None):
    return _get_state_impl(req.session_id if req else None)


def _get_state_impl(session_id: str | None):
    sess = _get_session(session_id)
    if sess is None:
        return JSONResponse(status_code=200, content={
            "visible_threats": [], "hidden_threat_count": TOTAL_NODES,
            "scan_coverage": 0.0, "system_health": 100,
            "score": 0.0, "grader_score": 0.0, "step": 0, "done": False,
            "error": "session_id required. Call /reset first.",
        })
    try:
        return Observation(**_obs(sess))
    except Exception as e:
        log.warning(f"get_state() snapshot error (transient): {e}")
        return JSONResponse(status_code=200, content={
            "visible_threats": [], "hidden_threat_count": 0,
            "scan_coverage": 0.0, "system_health": sess.state.get("system_health", 100),
            "score": 0.0, "grader_score": 0.0,
            "step": sess.state.get("step", 0), "done": sess.state.get("done", False),
        })


@app.post("/step")
def step(req: StepRequest):
    # OpenEnv spec requires HTTP 200 always; errors go in the response body.
    # When session_id is omitted, fall back to the most-recently-created session
    # so that clients (e.g. test helpers) that don't track session_id still work.
    sess = _get_session(req.session_id)
    if sess is None:
        return JSONResponse(
            status_code=200,
            content={
                "action": "", "reward": 0.0, "reason": "No active session. Call /reset first.",
                "confidence": 0.0, "done": False, "error": "no_active_session",
                "score": 0.0, "grader_score": 0.0, "step": 0,
                "visible_threats": [], "hidden_threat_count": 5, "scan_coverage": 0.0, "system_health": 100,
            }
        )

    try:
        _validate_session_state(sess)
        s = sess.state  # local alias for brevity

        if s["done"]:
            if DEBUG:
                log.debug("step() called after done=True")
            return safe_response(
                _obs(sess), action="", reward=0.0,
                reason="Episode is over. Call /reset to start a new episode.",
                confidence=1.0, error="Episode over — call /reset",
            )

        raw_action = req.action.strip().lower()

        # Unknown action — safe fallback
        if raw_action not in VALID_ACTIONS:
            if DEBUG:
                log.debug(f"Unknown action: {raw_action!r}")
            reward = _clamp_reward(-0.5)
            s["system_health"] = max(0, s["system_health"] - 5)
            _age_threats(sess)
            _update_visibility(sess)
            _clamp_health(sess)
            _all_r = sess.episode_rewards + [reward]
            s["score"] = round(sum(_all_r) / len(_all_r), 4)
            s["step"] += 1
            if s["system_health"] <= 0 or s["step"] >= sess.task_config.get("max_steps", 50):
                s["done"] = True
            obs = _obs(sess)
            sess.history.append({"step": s["step"], "action": raw_action,
                                  "reward": round(reward, 3), "attack": None})
            return safe_response(
                obs, action=raw_action, reward=reward,
                reason=f"'{raw_action}' is not a recognised action. Valid: block_ip, isolate_machine, patch, ignore, scan_node_1..5.",
                confidence=0.0, error="invalid action",
            )

        reason = ""
        confidence = 0.5
        matched = False
        early_bonus = False
        matched_threat_type = None
        scan_found_nothing = False

        # ── RESOURCE CHECK ──
        # Compute current resources remaining to enforce budget constraints.
        _COST_RAW = {"isolate_machine": 0.4, "block_ip": 0.3, "patch": 0.3}
        _task_budget = max(0.01, sess.task_config.get("resource_per_step", 1.0))
        _total_spent_now = sum(_COST_RAW.get(a, 0.2 if a.startswith("scan") else 0.0) for a in sess.episode_actions_taken)
        _total_budget_now = max(0.01, _task_budget * sess.task_config.get("max_steps", 50))
        _resources_now = max(0.0, 1.0 - _total_spent_now / _total_budget_now)
        _resource_exhausted = _resources_now <= 0.0

        # ── SCAN ──
        if raw_action.startswith("scan"):
            node = raw_action[len("scan_"):] if raw_action.startswith("scan_") else ""
            if node in NODES:
                s["scanned_nodes"].add(node)
                # Scan effectiveness also degrades at 50% when budget exhausted
                false_neg = sess.task_config.get("false_negative_rate", 0.0)
                if _resource_exhausted:
                    false_neg = min(0.99, false_neg + 0.5)  # scan less reliable when overloaded
                revealed = False
                for t in s["threats"]:
                    if t["node"] == node and not t["visible"] and not t.get("contained"):
                        if sess.rng.random() > false_neg:
                            t["visible"] = True
                            revealed = True
                reason, confidence = _build_reason(raw_action, False, None, False)
                if revealed:
                    reason = f"Scan of {node} revealed a hidden threat. Partial observability lifted for this node."
                    confidence = 0.90
                else:
                    reason = f"Scan of {node} found no new threats. Coverage improved."
                    confidence = 0.75
                    scan_found_nothing = True
            else:
                reason = f"'{node}' is not a valid node. Valid nodes: node_1 through node_5."
                confidence = 0.10

        # ── DEFENSE ──
        else:
            for t in s["threats"]:
                if t["visible"] and not t.get("contained"):
                    # Use original_type for MITRE lookup — type is preserved even after
                    # stage escalation so agents are never penalized for correct identification.
                    correct = _get_correct_action(t.get("original_type", t["type"]), t.get("severity", 0.5), t.get("stage", "initial"))
                    if raw_action == correct:
                        if t.get("is_false_positive"):
                            # False positive: waste the action, remove ghost threat
                            t["contained"] = True
                            matched_threat_type = t["type"]
                            matched = False  # treated as wrong action for reward purposes
                            sess.false_positive_actions += 1
                        elif _resource_exhausted and sess.rng.random() < 0.5:
                            # Resource exhausted: 50% chance action fails — forces budget planning
                            matched_threat_type = t["type"]
                            matched = False  # action attempted but resource-starved response failed
                            reason = f"Resources exhausted. {raw_action} on {t['type']} failed (50% degraded effectiveness). Containment unsuccessful."
                            confidence = 0.30
                        else:
                            t["contained"] = True
                            matched_threat_type = t["type"]
                            matched = True
                            early_bonus = t["age"] < 3
                            sess.containment_events.append({
                                "threat_id": t["id"],
                                "age_at_containment": t["age"],
                                "threat_type": t.get("original_type", t["type"]),
                            })
                        break

            if not matched:
                for t in s["threats"]:
                    if t["visible"] and not t.get("contained"):
                        matched_threat_type = t["type"]
                        break
                if raw_action == "ignore":
                    s["system_health"] = max(0, s["system_health"] - 10)
                else:
                    s["system_health"] = max(0, s["system_health"] - 5)

            reason, confidence = _build_reason(raw_action, matched, matched_threat_type, early_bonus)

        # Passive health degradation — only real threats (not false positives) cause damage.
        # FPs are phantom alerts and must not inflate the active-threat count.
        _degrade_rate = sess.task_config.get("health_degradation_rate", 0.0)
        if _degrade_rate > 0:
            _active = sum(1 for t in s["threats"] if not t.get("contained") and not t.get("is_false_positive"))
            s["system_health"] = max(0, s["system_health"] - _degrade_rate * (_active / TOTAL_NODES) * 100)

        _age_threats(sess)
        _update_visibility(sess)
        _clamp_health(sess)

        # Reward authority: MITRE-aligned lookup table.
        # Normalized via _clamp_reward((r + 2.0) / 4.0) → [0.0, 1.0].
        # correct:      raw 1.0 (+0.1 early bonus if age<3) → 0.750 (0.775)
        # wrong:        raw -0.5                            → 0.375
        # ignore:       raw -1.5                            → 0.125
        # scan reveal:  raw 0.02                            → 0.505
        # scan empty:   raw -0.3                            → 0.425
        if raw_action.startswith("scan"):
            reward = _clamp_reward(0.02) if not scan_found_nothing else _clamp_reward(-0.3)
        elif raw_action == "ignore":
            reward = _clamp_reward(-1.5)
        elif raw_action in ("block_ip", "isolate_machine", "patch"):
            if matched:
                reward = _clamp_reward(1.1 if early_bonus else 1.0)
            else:
                reward = _clamp_reward(-0.5)
        else:
            reward = _clamp_reward(-0.5)

        # Running average score — never saturates (each reward ∈ [0,1], mean ∈ [0,1])
        _all_rewards = sess.episode_rewards + [reward]
        s["score"] = round(sum(_all_rewards) / len(_all_rewards), 4)
        s["step"] += 1

        if s["system_health"] <= 0 or s["step"] >= sess.task_config.get("max_steps", 50):
            s["done"] = True

        obs = _obs(sess)

        sess.history.append({"step": s["step"], "action": raw_action,
                              "reward": round(reward, 3), "attack": matched_threat_type})
        sess.episode_history.append({"step": s["step"], "action": raw_action,
                                     "reward": float(reward), "done": bool(s["done"]),
                                     "reason": reason})

        # Analytics tracking
        sess.episode_actions_taken.append(raw_action)
        sess.episode_rewards.append(reward)
        _MITIGATIONS = {"block_ip", "isolate_machine", "patch"}
        # Note: false_positive_actions is now tracked in the defense loop (FP threats)
        # and here for wrong mitigations on real threats (matched=False, action!=FP)
        if raw_action in _MITIGATIONS and not matched and matched_threat_type is not None:
            # Only count as FP if we targeted something but matched=False and it wasn't an FP threat
            # (FP threats already incremented false_positive_actions in the defense loop)
            pass  # false_positive_actions now tracked centrally in defense loop
        for threat in obs.get("visible_threats", []):
            tid = threat.get("id", "")
            if tid:
                sess.threats_detected.add(tid)
        for t in s["threats"]:
            if t.get("contained"):
                tid = str(t.get("id", f"{t['type']}_{t['node']}"))
                sess.threats_contained.add(tid)

        # Red team
        translated = translate_action(raw_action)
        threat_ctx = (matched_threat_type or "UNKNOWN").upper()
        sess.attacker.observe_defender_action(translated, threat_ctx)

        if s["done"]:
            # Use the single authoritative grader formula (all 4 components)
            final_grader = _compute_grader_score(sess)
            containment = sum(1 for t in s["threats"] if t.get("contained") and not t.get("is_false_positive")) / max(1, sum(1 for t in s["threats"] if not t.get("is_false_positive")))
            sess.attacker.on_episode_end(
                defender_won=containment >= 0.8,
                score=final_grader,
            )

        return safe_response(obs, action=raw_action, reward=reward,
                             reason=reason, confidence=confidence)

    except Exception as e:
        log.error(f"/step unhandled exception: {e}", exc_info=True)
        # sess is always defined here because session validation happens before try block
        try:
            _validate_session_state(sess)
            obs = _obs(sess)
        except Exception:
            _do_reset_session(sess)
            obs = _obs(sess)
        # Reward errors indicate engine failure — return explicit error response
        if isinstance(e, RuntimeError) and "session_state_reset" in str(e):
            error_msg = "session_state_reset"
        elif "Reward" in str(type(e).__name__):
            error_msg = f"reward_error: {str(e)[:64]}"
        else:
            error_msg = "internal_error"
        return safe_response(obs, action="", reward=0.0,
                             reason=f"Step failed: {error_msg}. State preserved.",
                             confidence=0.0, error=error_msg)


@app.get("/attacker-report")
def attacker_report(session_id: str | None = None):
    sess = _get_session(session_id)
    if sess is None:
        return {"error": "session_id required. Call /reset first.", "episode_count": 0}
    aa = sess.attacker
    p = aa.defender_profile
    return {
        "episode_count":     aa.episode_count,
        "current_strategy":  aa.current_strategy,
        "defender_profile": {
            "strategy_label":   p.get_defender_strategy_label(),
            "isolation_rate":   round(p.isolation_rate, 3),
            "block_rate":       round(p.block_rate,     3),
            "scan_rate":        round(p.scan_rate,      3),
            "patch_rate":       round(p.patch_rate,     3),
            "most_used_action": p.get_most_used_action(),
            "steps_observed":   p.steps_observed,
            "action_counts":    dict(p.action_counts),
        },
        "strategy_history": aa.strategy_history[-5:],
        "adaptation_log":   aa.adaptation_log[-10:],
        "full_report":      aa.get_full_adaptation_report(),
    }


MITRE_INTEL = {
    "phishing": {
        "technique_id":   "T1566",
        "technique_name": "Phishing",
        "tactic":         "Initial Access",
        "tactic_id":      "TA0001",
        "severity":       "HIGH",
        "recommended_action": "block_ip",
        "description": "Adversary sending malicious emails to gain initial access",
        "indicators": ["suspicious_email", "malicious_attachment", "fake_link"],
        "mitigation": "Block source IP, enable email filtering, user awareness training",
        "kill_chain_phase": "delivery",
        "similar_incidents": 3,
    },
    "malware": {
        "technique_id":   "T1204",
        "technique_name": "User Execution",
        "tactic":         "Execution",
        "tactic_id":      "TA0002",
        "severity":       "CRITICAL",
        "recommended_action": "isolate_machine",
        "description": "Malicious code executing on compromised endpoint",
        "indicators": ["unusual_process", "file_modification", "registry_change"],
        "mitigation": "Isolate affected machine, run forensic analysis, restore from backup",
        "kill_chain_phase": "exploitation",
        "similar_incidents": 7,
    },
    "ddos": {
        "technique_id":   "T1499",
        "technique_name": "Endpoint Denial of Service",
        "tactic":         "Impact",
        "tactic_id":      "TA0040",
        "severity":       "HIGH",
        "recommended_action": "patch",
        "description": "Flooding target service to cause denial of service",
        "indicators": ["high_traffic", "service_unavailable", "cpu_spike"],
        "mitigation": "Apply rate limiting patch, enable DDoS protection, null-route attacker",
        "kill_chain_phase": "actions_on_objectives",
        "similar_incidents": 2,
    },
    "ransomware": {
        "technique_id":   "T1486",
        "technique_name": "Data Encrypted for Impact",
        "tactic":         "Impact",
        "tactic_id":      "TA0040",
        "severity":       "CRITICAL",
        "recommended_action": "isolate_machine",
        "description": "Encrypting files to extort ransom from victim organization",
        "indicators": ["file_encryption", "ransom_note", "shadow_copy_deletion"],
        "mitigation": "Immediately isolate machine, do not pay ransom, restore from backup",
        "kill_chain_phase": "actions_on_objectives",
        "similar_incidents": 5,
    },
    "lateral_movement": {
        "technique_id":   "T1021",
        "technique_name": "Remote Services",
        "tactic":         "Lateral Movement",
        "tactic_id":      "TA0008",
        "severity":       "HIGH",
        "recommended_action": "block_ip",
        "description": "Adversary moving through network using remote services",
        "indicators": ["unusual_login", "remote_access", "credential_reuse"],
        "mitigation": "Block internal IP, reset credentials, enable MFA",
        "kill_chain_phase": "lateral_movement",
        "similar_incidents": 4,
    },
}

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}


@app.get("/threat-intel")
def threat_intel(session_id: str | None = None):
    """Returns MITRE ATT&CK enriched threat intelligence about active threats."""
    try:
        sess = _get_session(session_id)
        if sess is None:
            return {"error": "session_id required", "active_campaigns": [], "risk_level": "UNKNOWN",
                    "threat_summary": {}, "network_assessment": {}, "recommended_actions": []}
        _validate_session_state(sess)
        visible_threats = _visible_threats(sess)
        scan_coverage   = round(len(sess.state["scanned_nodes"]) / TOTAL_NODES, 3)
        system_health   = sess.state["system_health"]

        active_campaigns = []
        for threat in visible_threats:
            # Derive severity from IOC signals — NOT from internal threat type.
            # This keeps the threat classification challenge intact.
            pps     = threat.get("packets_per_second", 0)
            outbound = threat.get("outbound_data_bytes", 0)
            spread  = threat.get("spread_rate", 0.0)
            procs   = threat.get("unusual_process_count", 0)
            if outbound > 7000 or pps > 500 or spread > 0.7:
                ioc_severity = "CRITICAL"
            elif outbound > 2000 or procs > 8 or spread > 0.4:
                ioc_severity = "HIGH"
            else:
                ioc_severity = "MEDIUM"

            active_campaigns.append({
                "threat_id":    threat.get("id", "unknown"),
                "node":         threat.get("node", "unknown"),
                "stage":        threat.get("stage", "unknown"),
                "age":          threat.get("age", 0),
                "dwell_time_steps": threat.get("age", 0),
                "severity":     ioc_severity,
                "confidence":   round(threat.get("detection_confidence", 0.8), 3),
                "urgency":      "IMMEDIATE" if threat.get("age", 0) >= 3 else "MONITOR",
                "ioc_summary": {
                    "packets_per_second":       pps,
                    "failed_auth_attempts":     threat.get("failed_auth_attempts", 0),
                    "outbound_data_bytes":       outbound,
                    "lateral_connection_count": threat.get("lateral_connection_count", 0),
                    "unusual_process_count":    procs,
                    "spread_rate":              spread,
                    "affected_node_count":      threat.get("affected_node_count", 1),
                    "is_persistent":            threat.get("is_persistent", False),
                },
            })

        active_campaigns.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 4))

        # Derive compromised nodes from lateral_movement threats
        compromised = list({
            t["node"] for t in sess.state["threats"]
            if t.get("stage") == "lateral_movement" and not t.get("contained")
        })

        if system_health < 30 or len(compromised) >= 3:
            risk_level = "CRITICAL"
        elif system_health < 60 or len(compromised) >= 2:
            risk_level = "HIGH"
        elif system_health < 80 or len(compromised) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "timestamp": sess.state["step"],
            "risk_level": risk_level,
            "active_campaigns": active_campaigns,
            "threat_summary": {
                "total_visible":    len(visible_threats),
                "critical_count":   sum(1 for t in active_campaigns if t["severity"] == "CRITICAL"),
                "high_count":       sum(1 for t in active_campaigns if t["severity"] == "HIGH"),
                "immediate_action": sum(1 for t in active_campaigns if t["urgency"] == "IMMEDIATE"),
            },
            "network_assessment": {
                "risk_level":      risk_level,
                "compromised_nodes": compromised,
                "scan_coverage":   scan_coverage,
                "system_health":   system_health,
                "unscanned_nodes": max(0, TOTAL_NODES - len(sess.state["scanned_nodes"])),
            },
            # recommended_actions intentionally omitted — agents must infer
            # the correct response from IOC signals, not from a direct lookup.
            "recommended_actions": [],
            "mitre_framework": "ATT&CK v14.0",
        }

    except Exception as e:
        log.error(f"/threat-intel error: {e}", exc_info=True)
        return {
            "timestamp":          0,
            "risk_level":         "UNKNOWN",
            "active_campaigns":   [],
            "threat_summary":     {"total_visible": 0},
            "network_assessment": {},
            "recommended_actions": [],
            "error":              str(e),
            "mitre_framework":    "ATT&CK v14.0",
        }


@app.get("/analytics")
def get_analytics(session_id: str | None = None):
    """Returns real SOC metrics for the current episode."""
    try:
        sess = _get_session(session_id)
        if sess is None:
            return {"error": "session_id required", "soc_metrics": {}, "resources_remaining": 0.0}
        _validate_session_state(sess)
        visible_threats = _visible_threats(sess)
        total_steps   = sess.state["step"]
        system_health = sess.state["system_health"]
        scan_coverage = round(len(sess.state["scanned_nodes"]) / TOTAL_NODES, 3)

        n_detected  = len(sess.threats_detected)
        n_contained = len(sess.threats_contained)
        # Total threats spawned this episode (initial + lateral spreads).
        # Used as the denominator for containment_rate to match tasks/base.py exactly:
        #   containment_rate = threats_contained / threats_total_spawned
        threats_total_spawned = len(sess.state["threats"])

        # Avoid division by zero; return 0.0 rates when nothing detected yet
        avg_steps_per_detection   = round(total_steps / max(1, n_detected), 2)
        avg_steps_per_containment = round(total_steps / max(1, n_contained), 2)

        detection_rate = round(n_detected / max(1, threats_total_spawned), 3)

        # containment_rate: contained / total_spawned — matches tasks/base.py formula.
        # (Previously used n_detected as denominator, which diverged when threats were never detected.)
        containment_rate = round(n_contained / max(1, threats_total_spawned), 3)

        total_mitigations = sum(
            1 for a in sess.episode_actions_taken
            if a in {"block_ip", "isolate_machine", "patch"}
        )
        false_positive_rate = round(
            sess.false_positive_actions / max(1, total_mitigations), 3
        )

        avg_reward = round(
            sum(sess.episode_rewards) / max(1, len(sess.episode_rewards)), 4
        )

        if len(sess.episode_rewards) >= 10:
            first_5 = sum(sess.episode_rewards[:5]) / 5
            last_5  = sum(sess.episode_rewards[-5:]) / 5
            trend = "IMPROVING" if last_5 > first_5 else "DECLINING"
        else:
            trend = "INSUFFICIENT_DATA"

        action_counts: dict = {}
        for a in sess.episode_actions_taken:
            action_counts[a] = action_counts.get(a, 0) + 1
        most_used = max(action_counts, key=action_counts.get) if action_counts else "none"

        if containment_rate >= 0.8 and system_health >= 70:
            grade = "A"
        elif containment_rate >= 0.6 and system_health >= 50:
            grade = "B"
        elif containment_rate >= 0.4 and system_health >= 30:
            grade = "C"
        else:
            grade = "D"

        # Recommended next action — derived from IOC signals, never from internal type.
        # Agents receive a hint based on observable data, not a type→action lookup.
        _vt = visible_threats
        _any = lambda key, threshold, op=(lambda a, b: a > b): any(
            op(t.get(key, 0), threshold) for t in _vt
        )
        if _any("packets_per_second", 500):
            recommended = "patch"
        elif _any("outbound_data_bytes", 2000) or _any("unusual_process_count", 5):
            recommended = "isolate_machine"
        elif _any("failed_auth_attempts", 20) or _any("lateral_connection_count", 8):
            recommended = "block_ip"
        elif scan_coverage < 1.0:
            scanned = sess.state["scanned_nodes"]
            unscanned = [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)
                         if f"node_{i}" not in scanned]
            recommended = unscanned[0] if unscanned else "ignore"
        else:
            recommended = "ignore"

        compromised = list({
            t["node"] for t in sess.state["threats"]
            if t.get("stage") == "lateral_movement" and not t.get("contained")
        })

        # Resources remaining: fraction of cumulative budget not yet spent.
        # Total budget = resource_per_step × max_steps; total spent = sum of raw action costs.
        task_budget = max(0.01, sess.task_config.get("resource_per_step", 1.0))
        _COST_RAW = {"isolate_machine": 0.4, "block_ip": 0.3, "patch": 0.3}
        total_spent = sum(
            _COST_RAW.get(a, 0.2 if a.startswith("scan") else 0.0)
            for a in sess.episode_actions_taken
        )
        total_budget = max(0.01, task_budget * sess.task_config.get("max_steps", 50))
        resources_remaining = round(max(0.0, 1.0 - total_spent / total_budget), 3)

        return {
            "episode_step":      total_steps,
            "performance_grade": grade,
            "soc_metrics": {
                "avg_steps_per_detection":   avg_steps_per_detection,
                "avg_steps_per_containment": avg_steps_per_containment,
                "detection_rate":       detection_rate,
                "containment_rate":     containment_rate,
                "false_positive_rate":  false_positive_rate,
                "avg_reward_per_step":  avg_reward,
                "reward_trend":         trend,
            },
            "threat_tracking": {
                "threats_detected":      n_detected,
                "threats_contained":     n_contained,
                "threats_total_spawned": threats_total_spawned,
                "threats_active":        len(visible_threats),
                "threats_ids_detected":  list(sess.threats_detected),
            },
            "network_status": {
                "system_health":     system_health,
                "scan_coverage":     scan_coverage,
                "compromised_nodes": compromised,
                "nodes_at_risk":     len(compromised),
            },
            "agent_behavior": {
                "total_actions":    len(sess.episode_actions_taken),
                "action_breakdown": action_counts,
                "most_used_action": most_used,
                "false_positives":  sess.false_positive_actions,
            },
            "recommended_next_action": recommended,
            "resources_remaining":     resources_remaining,
            "attacker_strategy": sess.attacker.current_strategy,
            "grader_score": _compute_grader_score(sess),
        }

    except Exception as e:
        log.error(f"/analytics error: {e}", exc_info=True)
        return {
            "episode_step":      0,
            "performance_grade": "UNKNOWN",
            "soc_metrics":       {},
            "threat_tracking":   {},
            "network_status":    {},
            "agent_behavior":    {},
            "recommended_next_action": "ignore",
            "error": str(e),
        }


def main():
    import uvicorn
    # Port 7860 matches openenv.yaml docker.port and Dockerfile EXPOSE
    port = int(_os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
