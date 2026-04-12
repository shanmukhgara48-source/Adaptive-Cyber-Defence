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
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from models import Observation
import importlib.util as _ilu, sys as _sys, os as _os
from adversarial_generator import generator as _adv_generator
from constants import COST_RAW, get_scaled_costs


# ─── TASK CONFIG IMPORTS ──────────────────────────────────────────────────────
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from tasks.nightmare import NightmareTask
from tasks.elite import EliteTask
from grader import (
    TASK_PASSING_SCORES,
    compute_speed_bonus as _compute_speed_bonus,
    compute_grader_score as _grader_formula,
    compute_criticality_weighted_containment as _criticality_weighted_containment,
    safe_score,
)

TASK_MAP = {
    "easy":       EasyTask,
    "medium":     MediumTask,
    "hard":       HardTask,
    "nightmare":  NightmareTask,
    "elite":      EliteTask,
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

# Node criticality — determines the weighted impact of containment and health loss.
# node_2 and node_5 form the hub ring in the linear+hub topology; compromising them
# is disproportionately dangerous because they route traffic to all other nodes.
# Agents that greedy-contain the easiest-to-reach threat rather than the most
# critical one will receive a lower criticality_weighted_containment score.
# This forces strategic prioritisation, defeating both greedy-first and scan-all baselines.
NODE_CRITICALITY: dict[str, float] = {
    "node_1": 0.40,   # edge leaf — lower-value target
    "node_2": 0.90,   # primary hub (connects to node_1, node_3, node_5)
    "node_3": 0.60,   # mid-chain application server
    "node_4": 0.50,   # secondary endpoint
    "node_5": 0.80,   # alternate hub (connects to node_2, node_4)
}

VALID_ACTIONS = frozenset(
    ["block_ip", "isolate_machine", "patch", "ignore"]
    + [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)]
    + [f"verify_node_{i}" for i in range(1, TOTAL_NODES + 1)]
    + [f"monitor_node_{i}" for i in range(1, TOTAL_NODES + 1)]
)

# Difficulty-scaled scan cost.  On hard (budget=0.30/step × 30 steps = 9.0),
# scanning all 5 nodes costs 5 × 0.45 = 2.25 per step — 75 % of one step's
# budget.  Even one scan/step at 0.45 exceeds the per-step allowance, making
# scan-all-nodes strategy budget-prohibitive at hard+.
# Easy remains affordable (0.15/scan) so new agents are not immediately penalised.
_SCAN_COST_BY_DIFFICULTY: dict[str, float] = {
    "easy":       0.15,
    "medium":     0.28,
    "hard":       0.45,
    "nightmare":  0.55,
    "elite":      0.65,
}

# IOC noise applied when a scan first reveals a threat.  Scans give imperfect
# information — the reveal adds ±30% noise to each IOC field.  Agents must
# track IOC evolution over subsequent steps to reduce classification uncertainty,
# rather than relying on a single post-scan snapshot.
_SCAN_IOC_NOISE_FACTOR: float = 0.30

# ── SEQUENTIAL DECISION MECHANICS ────────────────────────────────────────────
# verify_node_X cost: same tier as scan — spending resources to confirm a threat
# before acting is the recommended practice but requires budget discipline.
_VERIFY_COST_BY_DIFFICULTY: dict[str, float] = {
    "easy":       0.12,
    "medium":     0.22,
    "hard":       0.38,
    "nightmare":  0.48,
    "elite":      0.58,
}

# Monitor cost is intentionally cheap: rewarding post-containment vigilance
# without making it budget-prohibitive.  Flat across difficulties.
_MONITOR_COST: float = 0.05

# patch takes 2 steps to deploy; block_ip and isolate_machine are instant.
# During the delay window the threat continues dealing health damage and can
# still escalate — agents must plan ahead and not re-act on mitigating threats.
_MITIGATION_DELAY: dict[str, int] = {
    "patch":           2,
    "block_ip":        0,
    "isolate_machine": 0,
}

# Resurface risk per threat type — probability weight (0-1) applied each step
# after _RESURFACE_START_STEP if no monitor action was taken post-containment.
# Persistent threats (malware, ransomware, lateral_movement) have meaningful risk;
# non-persistent threats (phishing, ddos) rarely resurface.
_RESURFACE_RISK_BY_TYPE: dict[str, float] = {
    "malware":          0.60,
    "ransomware":       0.70,
    "lateral_movement": 0.50,
    "phishing":         0.15,
    "ddos":             0.10,
}
_RESURFACE_START_STEP: int   = 4    # steps after containment before resurface risk activates
_BASE_RESURFACE_PROB:  float = 0.22 # per-step probability (scaled by resurface_risk)

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
#
# ADVERSARIAL DESIGN — profiles are intentionally overlapping so no single
# feature uniquely identifies a threat class.  Agents must reason across
# the full IOC vector; any single-axis threshold classifier will misclassify
# the overlap region (≈30-40% of spawns) and be penalised for wrong actions.
#
# Overlap map (shared value ranges between adjacent types):
#   pps:     malware(20-300) ∩ ddos(150-4000)      at 150-300
#            phishing(5-100) ∩ malware(20-300)      at 20-100
#   auth:    phishing(30-170) ∩ lateral(15-120)     at 30-120
#            lateral(15-120) ∩ ransomware(10-70)    at 10-70
#   ob:      malware(1000-7000) ∩ ransomware(3500-18000) at 3500-7000
#   lat:     ransomware(5-24) ∩ lateral(4-22)       at 5-22
#            malware(3-16) ∩ ransomware(5-24)       at 5-16
#   proc:    malware(6-28) ∩ ransomware(10-32)      at 10-28
#
# Correct classification requires combining ≥2 signals — mitigating the
# single-axis lookup exploit described in the adversarial evaluation.
_IOC_PROFILES: dict[str, dict] = {
    "phishing": {
        "packets_per_second":       (5, 100),    # overlaps malware (20-300) at 20-100
        "failed_auth_attempts":     (30, 170),   # still highest but lateral overlaps at 30-120
        "outbound_data_bytes":      (100, 2000), # was (80-600); overlap with malware bottom
        "lateral_connection_count": (0, 8),      # was (0-2); overlaps malware (3-16) at 3-8
        "unusual_process_count":    (2, 12),     # was (1-4); overlaps malware (6-28) at 6-12
        "spread_rate":              (0.05, 0.30),
        "is_persistent":            False,
    },
    "malware": {
        "packets_per_second":       (20, 300),   # was (15-65); overlaps ddos (150-4000) at 150-300
        "failed_auth_attempts":     (5, 50),     # was (0-8); overlaps phishing (30-170) at 5-50
        "outbound_data_bytes":      (1000, 7000),# was (900-4500); overlaps ransomware at 3500-7000
        "lateral_connection_count": (3, 16),     # was (2-10); overlaps ransomware (5-24) at 5-16
        "unusual_process_count":    (6, 28),     # was (6-20); overlaps ransomware (10-32) at 10-28
        "spread_rate":              (0.25, 0.65),
        "is_persistent":            True,
    },
    "ddos": {
        "packets_per_second":       (150, 4000), # was (900-9500); overlaps malware (20-300) at 150-300
        "failed_auth_attempts":     (0, 25),     # was (0-5); small lift creates phishing mimicry
        "outbound_data_bytes":      (50, 600),   # was (0-120)
        "lateral_connection_count": (0, 6),      # was (0-2)
        "unusual_process_count":    (1, 8),      # was (1-3)
        "spread_rate":              (0.0, 0.10),
        "is_persistent":            False,
    },
    "ransomware": {
        "packets_per_second":       (8, 60),     # was (5-25)
        "failed_auth_attempts":     (10, 70),    # was (8-45); overlaps lateral (15-120) at 15-70
        "outbound_data_bytes":      (3500, 18000),# was (9000-48000); still highest but overlaps malware at 3500-7000
        "lateral_connection_count": (5, 24),     # was (6-20); overlaps lateral (4-22) at 5-22
        "unusual_process_count":    (10, 32),    # was (10-28)
        "spread_rate":              (0.50, 0.92),
        "is_persistent":            True,
    },
    "lateral_movement": {
        "packets_per_second":       (20, 180),   # was (20-90)
        "failed_auth_attempts":     (15, 120),   # was (25-115); overlaps phishing (30-170) at 30-120
        "outbound_data_bytes":      (200, 3000), # was (200-1200); overlaps malware at 1000-3000
        "lateral_connection_count": (4, 22),     # was (12-38); still highest but overlaps ransomware at 5-22
        "unusual_process_count":    (3, 16),     # was (3-10); overlaps malware (6-28) at 6-16
        "spread_rate":              (0.45, 0.88),
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


# ── KILL CHAIN PROGRESSION ────────────────────────────────────────────────────
# Each attack type has a staged kill chain.  Threats start at stage_idx=0 and
# advance each step with probability _KILL_CHAIN_ADVANCE_PROB.  Later stages:
#   • emit amplified IOC signals (harder to classify, blend across attack types)
#   • deal more health damage per step (kill_chain_health_multiplier)
#   • have higher resurface risk on containment
#
# Stage format: (stage_name, health_drain_multiplier, ioc_amplifiers: dict[str, float])
# ioc_amplifiers multiply existing IOC values at transition — signals sharpen/shift.
#
# Research motivation: single-stage threats are solvable by a memoryless reactive
# policy (scan → classify → mitigate).  Kill chains force the agent to track stage
# history across multiple steps, plan ahead (patch early, don't wait for encryption),
# and reason about escalating damage vs. available budget — impossible for stateless
# single-step classifiers.
KILL_CHAIN_STAGES: dict[str, list] = {
    "phishing": [
        ("initial_access",    1.0, {}),
        ("credential_access", 1.8, {"failed_auth_attempts": 2.0, "outbound_data_bytes": 1.5}),
        ("lateral_movement",  2.5, {"lateral_connection_count": 3.0, "failed_auth_attempts": 1.5}),
        ("exfiltration",      3.5, {"outbound_data_bytes": 4.0, "lateral_connection_count": 2.0}),
    ],
    "malware": [
        ("initial_access",    1.0, {}),
        ("execution",         1.5, {"unusual_process_count": 2.0, "packets_per_second": 1.5}),
        ("persistence",       2.0, {"unusual_process_count": 1.5, "lateral_connection_count": 1.5}),
        ("lateral_movement",  3.0, {"lateral_connection_count": 2.5, "packets_per_second": 2.0}),
    ],
    "ransomware": [
        ("initial_access",    1.0, {}),
        ("execution",         2.0, {"unusual_process_count": 2.5}),
        ("encryption",        4.0, {"outbound_data_bytes": 3.5, "unusual_process_count": 2.0}),
        ("exfiltration",      5.0, {"outbound_data_bytes": 6.0, "lateral_connection_count": 2.0}),
    ],
    "ddos": [
        ("reconnaissance",    0.5, {}),
        ("amplification",     2.0, {"packets_per_second": 4.0}),
        ("saturation",        4.0, {"packets_per_second": 12.0, "unusual_process_count": 1.5}),
    ],
    "lateral_movement": [
        ("discovery",         1.0, {}),
        ("lateral_movement",  2.0, {"lateral_connection_count": 2.5}),
        ("privilege_escalation", 3.0, {"failed_auth_attempts": 3.0, "lateral_connection_count": 2.0}),
    ],
}

# Per-step probability of advancing one stage in the kill chain.
# Lower than attack_progression_prob (0.25) so kill chain and stage-escalation
# are distinct mechanics; each is independently tunable per difficulty.
_KILL_CHAIN_ADVANCE_PROB: float = 0.28

_HARD_DIFFICULTIES = {"hard", "nightmare", "elite"}


def _spawn_iocs(t_type: str, rng: random.Random, is_fp: bool = False,
                difficulty: str = "easy") -> dict:
    """Generate initial behavioral IOC values for a new threat.
    These are stored on the internal threat dict and exposed in _visible_threats.
    The threat type itself is NOT exposed — agents must infer it from these signals.

    Multiplicative noise (via session RNG) is applied to all IOC fields so identical
    profiles produce varied signals — agents cannot memorise a fixed threshold lookup.
    For hard+ difficulty, cross-signal contamination blends attack-class signatures
    to simulate real-world multi-technique intrusions.
    All randomness uses the session RNG exclusively to preserve seed determinism.
    """
    profile = _FP_IOC_PROFILE if is_fp else _IOC_PROFILES.get(t_type, _IOC_PROFILES["malware"])
    pps_lo, pps_hi = profile["packets_per_second"]
    auth_lo, auth_hi = profile["failed_auth_attempts"]
    ob_lo, ob_hi   = profile["outbound_data_bytes"]
    lat_lo, lat_hi = profile["lateral_connection_count"]
    proc_lo, proc_hi = profile["unusual_process_count"]
    sr_lo, sr_hi   = profile["spread_rate"]

    # Base values sampled from profile ranges
    pps  = rng.randint(pps_lo, pps_hi)
    auth = rng.randint(auth_lo, auth_hi)
    ob   = rng.randint(ob_lo, ob_hi)
    lat  = rng.randint(lat_lo, lat_hi)
    proc = rng.randint(proc_lo, proc_hi)
    sr   = round(rng.uniform(sr_lo, sr_hi), 3)

    # Apply per-field multiplicative noise (session RNG only — determinism preserved)
    def _ni(val: int, factor: float) -> int:
        return max(0, int(val * (1 + rng.uniform(-factor, factor))))

    pps  = _ni(pps,  0.35)
    auth = _ni(auth, 0.35)
    ob   = _ni(ob,   0.40)
    lat  = _ni(lat,  0.30)
    proc = _ni(proc, 0.30)
    sr   = round(max(0.0, min(1.0, sr * (1 + rng.uniform(-0.20, 0.20)))), 3)

    # NO noise floors — removing the per-type minimum guarantees broke the single-feature
    # threshold exploit.  Without floors, the overlap regions in _IOC_PROFILES can produce
    # low values on the historically "dominant" axis, forcing multi-signal reasoning.
    # (The old floors guaranteed DDoS pps≥200, ransomware ob≥500, etc. — enough
    #  to make a one-feature lookup correct 100% of the time.)

    iocs = {
        "packets_per_second":       pps,
        "failed_auth_attempts":     auth,
        "outbound_data_bytes":      ob,
        "lateral_connection_count": lat,
        "unusual_process_count":    proc,
        "spread_rate":              sr,
        "is_persistent":            profile["is_persistent"],
        "affected_node_count":      1,
        "detection_confidence":     round(
            rng.uniform(0.3, 0.7) if is_fp else rng.uniform(0.6, 1.0), 3
        ),
    }

    # Cross-signal contamination — now applied at ALL difficulties, not just hard+.
    # Strength scales with difficulty so easy episodes remain learnable but not exploitable.
    # Every branch consumes exactly 3 RNG calls to keep the sequence length uniform
    # across threat types and difficulty levels (seed determinism preserved).
    if not is_fp:
        # Contamination strength: easy=weak, medium=moderate, hard+=strong
        if difficulty in _HARD_DIFFICULTIES:
            contamination  = rng.uniform(0.30, 0.60)   # call 1
            contamination2 = rng.uniform(0.20, 0.45)   # call 2
        elif difficulty == "medium":
            contamination  = rng.uniform(0.15, 0.35)   # call 1
            contamination2 = rng.uniform(0.10, 0.25)   # call 2
        else:  # easy
            contamination  = rng.uniform(0.05, 0.20)   # call 1
            contamination2 = rng.uniform(0.05, 0.15)   # call 2
        if t_type == "lateral_movement":
            # Stolen credentials add auth-failure signal — blends with phishing
            iocs["failed_auth_attempts"] += int(rng.randint(15, 120) * contamination)
        elif t_type == "ransomware":
            # Data staging adds lateral-connection signal — blends with lateral_movement
            iocs["lateral_connection_count"] += int(rng.randint(5, 24) * contamination)
        elif t_type == "malware":
            # C2 beaconing elevates packet rate — blends into lower DDoS range
            iocs["packets_per_second"] += int(rng.randint(50, 500) * contamination2)
        elif t_type == "ddos":
            # Amplification reflectors spawn unusual processes — blends with malware
            iocs["unusual_process_count"] += int(rng.randint(2, 10) * contamination)
        else:  # phishing
            # Credential harvesting exfiltrates small data — blends with malware outbound
            iocs["outbound_data_bytes"] += int(rng.randint(100, 1200) * contamination)
    else:
        # False positives: still consume 3 RNG calls to keep sequence length identical
        _ = rng.uniform(0.0, 1.0)   # call 1
        _ = rng.uniform(0.0, 1.0)   # call 2
        _ = rng.randint(0, 1)        # call 3

    return iocs


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
    {"id": "easy",       "name": "Easy Defense",      "difficulty": "easy",       "passing_score": TASK_PASSING_SCORES["easy"],       "max_steps": 30,  "description": "Three simultaneous attacks. High detection, generous resources. Contain all before lateral spread."},
    {"id": "medium",     "name": "Medium Defense",    "difficulty": "medium",     "passing_score": TASK_PASSING_SCORES["medium"],     "max_steps": 50,  "description": "Two intrusions with limited resources, FP noise. Requires threat prioritisation."},
    {"id": "hard",       "name": "Hard Defense",      "difficulty": "hard",       "passing_score": TASK_PASSING_SCORES["hard"],       "max_steps": 30,  "description": "APT across 5 nodes. Low detection, scarce resources, fast progression."},
    {"id": "nightmare",  "name": "Nightmare Defense", "difficulty": "nightmare",  "passing_score": TASK_PASSING_SCORES["nightmare"],  "max_steps": 15,  "description": "Nation-state APT. Near-zero detection, 15 steps. Designed to challenge frontier LLMs."},
    {"id": "elite",      "name": "Elite Defense",     "difficulty": "elite",      "passing_score": TASK_PASSING_SCORES["elite"],      "max_steps": 15,  "description": "Persistent threat with insider access. All nodes pre-compromised. Kill chain advances every step."},
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
            # Visibility threshold: nightmare/elite use 8 (longer hidden window)
            "age_visibility_threshold":  8 if name in ("nightmare", "elite") else 5,
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
    action_counts: dict = field(default_factory=dict)
    containment_events: list = field(default_factory=list)
    attack_plan: dict = field(default_factory=dict)
    rng: random.Random = field(default_factory=lambda: random.Random(0))  # overwritten at reset()
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
    """Extract components from session state, delegate formula to grader.py.

    Resource efficiency uses get_scaled_costs() so no MITRE-correct action
    (e.g. isolate_machine on hard task) is systematically penalised for
    exceeding the per-step budget.  Step-level _resource_exhausted tracking
    deliberately retains COST_RAW to keep gameplay mechanics unchanged.

    FP penalty = min(1.0, false_positive_actions / max(1, real_threats_seen)).
    Max deduction is 0.10 — enough to matter at score thresholds without
    punishing early exploratory over-action catastrophically.
    """
    s = sess.state
    _contained = sum(1 for t in s["threats"] if t.get("contained") and not t.get("is_false_positive"))
    _total_real = max(1, sum(1 for t in s["threats"] if not t.get("is_false_positive")))
    containment_rate = _contained / _total_real
    critical_health = s["system_health"] / 100.0
    _task_budget = max(0.01, sess.task_config.get("resource_per_step", 1.0))
    # Scaled action costs: no correct mitigation exceeds resource_per_step
    _scaled = get_scaled_costs(_task_budget)
    # Difficulty-scaled scan/verify cost (same as used in step resource check)
    _scan_cost_grader   = _SCAN_COST_BY_DIFFICULTY.get(sess.task_name, 0.20)
    _verify_cost_grader = _VERIFY_COST_BY_DIFFICULTY.get(sess.task_name, 0.15)
    def _action_cost_grader(a: str) -> float:
        if a.startswith("scan"):    return _scan_cost_grader
        if a.startswith("verify"):  return _verify_cost_grader
        if a.startswith("monitor"): return _MONITOR_COST
        return _scaled.get(a, 0.0)
    _total_spent = sum(_action_cost_grader(a) for a in sess.episode_actions_taken)
    _total_budget = max(0.01, _task_budget * sess.task_config.get("max_steps", 50))
    raw_efficiency = max(0.0, 1.0 - _total_spent / _total_budget)

    # Resource efficiency damping — closes the under-spending exploit.
    # An agent that ignores all threats spends nothing (raw_efficiency=1.0) but
    # also contains nothing (containment_rate=0).  Frugality is only efficient
    # when threats are being contained.  When threats remain active, unspent
    # budget represents missed interventions, not genuine resource discipline.
    #
    # Damping formula:
    #   uncontained_fraction = (total_real - contained) / total_real ∈ [0, 1]
    #   damping              = 1.0 - 0.5 × uncontained_fraction ∈ [0.5, 1.0]
    #   resource_efficiency  = raw_efficiency × damping
    #
    # Effect on exploit:
    #   ignore-all agent:   raw=1.0, damping=0.5 → efficiency=0.50  (was 1.0)
    #   contain-all agent:  raw=variable, damping=1.0 → no penalty
    #   contain-half agent: raw=variable, damping=0.75
    #
    # The 0.5 floor ensures partial credit for genuinely tight-budget scenarios
    # where an agent correctly prioritises surviving with unspent budget over
    # wasting resources on low-value actions.
    _uncontained_fraction = (_total_real - _contained) / _total_real
    _efficiency_damping   = 1.0 - 0.5 * _uncontained_fraction
    resource_efficiency   = raw_efficiency * _efficiency_damping

    speed_bonus = _compute_speed_bonus(sess.containment_events)
    # FP penalty: over-action on ghost alerts, bounded to [0, 1]
    fp_penalty = min(1.0, sess.false_positive_actions / _total_real)
    # Criticality-weighted containment: high-value hub nodes (node_2, node_5) matter more.
    # Blended 60/40 with raw containment_rate inside _grader_formula.
    cwc = _criticality_weighted_containment(s["threats"], NODE_CRITICALITY)
    return _grader_formula(
        containment_rate=containment_rate,
        critical_health=critical_health,
        resource_efficiency=resource_efficiency,
        speed_bonus=speed_bonus,
        fp_penalty=fp_penalty,
        criticality_weighted_containment=cwc,
    )


def _make_threats_fixed(task_config: dict, rng: random.Random, attacker=None,
                        task_name: str = "easy") -> list:
    """Make threats for a new episode using the given task config."""
    threats = []
    count = task_config.get("threat_count", 3)
    for idx in range(count):
        t_type = rng.choice(ATTACKS)
        node   = rng.choice(NODES)
        iocs = _spawn_iocs(t_type, rng, difficulty=task_name)
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
            # ── Sequential decision fields ───────────────────────────────────
            # is_verified: set to True after verify_node_X is called on this threat's node.
            # Verify is optional — it earns +0.15 reward when it confirms a new threat.
            # Mitigating without verify is fully valid; there is no penalty for skipping it.
            "is_verified":           False,
            # pending_action/mitigation_progress: patch is a 2-step delayed action.
            # During the delay the threat keeps escalating; agents must track it.
            "pending_action":        None,
            "mitigation_progress":   0,
            # resurface_risk: probability weight for post-containment resurgence.
            # Rises each time the threat resurfaces.  Reduced by monitor actions.
            "resurface_risk":        _RESURFACE_RISK_BY_TYPE.get(t_type, 0.30),
            # steps_since_contained: clock for resurface check start.
            "steps_since_contained": 0,
            # ── Kill chain fields ────────────────────────────────────────────
            # kill_chain_stage_idx: current position in KILL_CHAIN_STAGES[type].
            # Advances each step with _KILL_CHAIN_ADVANCE_PROB.  Higher index =
            # more damage, more ambiguous IOC signals, harder to contain.
            "kill_chain_stage_idx":        0,
            "kill_chain_health_multiplier": 1.0,  # from current stage entry
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
                    new_iocs = _spawn_iocs(new_type, rng, difficulty=task_name)
                    t.update(new_iocs)
    return threats


def _fresh_state(task_config: dict, rng: random.Random, attacker=None,
                 task_name: str = "easy") -> dict:
    return {
        "threats": _make_threats_fixed(task_config, rng, attacker, task_name=task_name),
        "scanned_nodes": set(),
        "system_health": 100,
        "score": 0.0001,
        "step": 0,
        "done": False,
        "false_positives_seen": 0,
    }


def _do_reset_session(sess: Session) -> None:
    """Reset all mutable fields on an existing Session object in-place."""
    sess.state                  = _fresh_state(sess.task_config, sess.rng, sess.attacker,
                                               task_name=sess.task_name)
    sess.history                = []
    sess.episode_history        = []
    sess.episode_actions_taken  = []
    sess.episode_rewards        = []
    sess.threats_detected       = set()
    sess.threats_contained      = set()
    sess.false_positive_actions = 0
    sess.action_counts          = {}
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
    # Use 0.001/0.999 (not 0.0/1.0) so the result is always strictly inside (0, 1),
    # matching the Phase 2 validator requirement even at extreme r values.
    return max(0.001, min(0.999, normalized_reward))


def _clamp_score(sess: Session) -> None:
    if not math.isfinite(sess.state["score"]):
        sess.state["score"] = safe_score(0.0)
    sess.state["score"] = safe_score(sess.state["score"])


# ─── LOGIC ────────────────────────────────────────────────────────────────────

def _update_visibility(sess: Session) -> None:
    """Auto-reveal threats based on age or lateral movement, gated by task difficulty.
    Uses effective_config so attacker detection_evasion actually reduces visibility.

    RNG discipline: exactly 2 calls are consumed per eligible (non-contained, non-visible,
    non-FP) threat regardless of which branch is taken.  This keeps the RNG sequence
    length uniform as a threat transitions between the "below age threshold" state and the
    "lateral_movement" state, preserving seed-based reproducibility across all episodes.

    Bug fixed (v2.1): the original lateral_movement path consumed only 1 RNG call and
    did NOT apply detect_prob.  On elite (detect_prob=0.08, fn_rate=0.15) a threat that
    escalated to lateral_movement stage had an 85% reveal chance — 10× higher than the
    intended 8%.  This made elite difficulty trivially solvable by ignoring threats until
    they spread (improving observability) rather than acting early.
    """
    eff          = sess.effective_config if sess.effective_config else sess.task_config
    age_thresh   = eff.get("age_visibility_threshold", sess.task_config.get("age_visibility_threshold", 5))
    detect_prob  = eff.get("base_detection_prob", 1.0)
    fn_rate      = eff.get("false_negative_rate", sess.task_config.get("false_negative_rate", 0.0))
    for t in sess.state["threats"]:
        if t.get("contained") or t.get("visible"):
            continue
        if t.get("is_false_positive"):
            continue  # FPs are immediately visible when created
        # Always consume exactly 2 RNG calls so the sequence length is invariant
        # over all threat states (lateral_movement vs age-threshold vs below-threshold).
        _roll1 = sess.rng.random()
        _roll2 = sess.rng.random()
        if t["stage"] == "lateral_movement":
            # Lateral movement now obeys detect_prob just like age-based detection.
            # The original code skipped detect_prob here, making lateral-movement
            # threats 10× easier to detect on elite than intended.
            if _roll1 < detect_prob and _roll2 > fn_rate:
                t["visible"] = True
        elif t["age"] >= age_thresh:
            if _roll1 < detect_prob and _roll2 > fn_rate:
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
    fp_iocs = _spawn_iocs(fp_type, sess.rng, is_fp=True, difficulty=sess.task_name)
    # With 40% probability, elevate one IOC to 60-80% of a real threat's level.
    # This creates genuine decision risk — the FP looks like a real threat on one signal
    # and forces the agent to weigh acting versus scanning for more evidence.
    if sess.rng.random() < 0.4:
        ioc_keys = ["packets_per_second", "failed_auth_attempts", "outbound_data_bytes",
                    "lateral_connection_count", "unusual_process_count"]
        chosen_ioc = sess.rng.choice(ioc_keys)
        real_type  = sess.rng.choice(ATTACKS)
        lo, hi     = _IOC_PROFILES[real_type][chosen_ioc]
        elevation  = sess.rng.uniform(0.6, 0.8)
        fp_iocs[chosen_ioc] = int(((lo + hi) / 2) * elevation)
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
                            child_iocs = _spawn_iocs(new_type, sess.rng,
                                                         difficulty=sess.task_name)
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
                            # Lateral-spread children inherit new-threat sequential + kill-chain fields
                            sess.state["threats"][-1].update({
                                "is_verified":                 False,
                                "pending_action":              None,
                                "mitigation_progress":         0,
                                "resurface_risk":              _RESURFACE_RISK_BY_TYPE.get(new_type, 0.30),
                                "steps_since_contained":       0,
                                "kill_chain_stage_idx":        0,
                                "kill_chain_health_multiplier": 1.0,
                            })

    # ── Kill chain advancement ────────────────────────────────────────────────
    # Each active, non-FP threat rolls each step to advance one stage in its
    # attack-type-specific kill chain.  Advancing applies IOC amplifiers (making
    # the threat harder to classify) and raises the health drain multiplier.
    # Stage transitions consume one RNG call per active threat for determinism.
    for t in sess.state["threats"]:
        if t.get("contained") or t.get("is_false_positive"):
            continue
        chain = KILL_CHAIN_STAGES.get(t.get("original_type", t["type"]), [])
        if not chain:
            _ = sess.rng.random()   # consume RNG call to preserve sequence length
            continue
        idx = int(t.get("kill_chain_stage_idx", 0))
        if idx >= len(chain) - 1:
            _ = sess.rng.random()   # already at final stage — consume for determinism
            continue
        if sess.rng.random() < _KILL_CHAIN_ADVANCE_PROB:
            idx += 1
            stage_name, health_mult, ioc_amps = chain[idx]
            t["kill_chain_stage_idx"]        = idx
            t["kill_chain_health_multiplier"] = health_mult
            # Apply IOC amplifiers — signals shift to reflect the new stage's behaviour.
            # Amplification blends signals across attack classes (e.g. phishing at
            # lateral_movement stage now looks partly like a lateral_movement threat).
            for ioc_key, factor in ioc_amps.items():
                if ioc_key in t and isinstance(t[ioc_key], (int, float)):
                    new_val = int(t[ioc_key] * factor)
                    t[ioc_key] = max(0, new_val)
            # Stage advancement auto-reveals the threat — hard to miss a threat in encryption
            if idx >= 2 and not t.get("visible"):
                t["visible"] = True

    # ── Pending mitigation resolution (delayed patch deployment) ─────────────
    # patch actions queue a 2-step delayed containment.  Each step we decrement
    # the counter; when it reaches 0 the threat is marked contained and a
    # containment_event is recorded for the speed_bonus calculation.
    for t in sess.state["threats"]:
        if not t.get("contained") and t.get("pending_action") and t.get("mitigation_progress", 0) > 0:
            t["mitigation_progress"] -= 1
            if t["mitigation_progress"] == 0:
                t["contained"] = True
                t["pending_action"] = None
                sess.containment_events.append({
                    "threat_id":          t["id"],
                    "age_at_containment": t["age"],
                    "threat_type":        t.get("original_type", t["type"]),
                })

    # ── Resurface check (post-containment persistent threats) ────────────────
    # Contained persistent threats that haven't been monitored within
    # _RESURFACE_START_STEP steps have a chance to re-activate each step.
    # Re-activated threats reset to "detected" phase with elevated severity,
    # deal an immediate health penalty, and raise their own resurface_risk so
    # each recurrence is harder to permanently eliminate.
    for t in sess.state["threats"]:
        if not t.get("contained"):
            continue
        if t.get("resurface_risk", 0.0) <= 0.0:
            continue
        t["steps_since_contained"] = t.get("steps_since_contained", 0) + 1
        if t["steps_since_contained"] < _RESURFACE_START_STEP:
            continue
        if sess.rng.random() < _BASE_RESURFACE_PROB * t["resurface_risk"]:
            # Threat resurfaces — revert to active visible state
            t["contained"]             = False
            t["visible"]               = True
            t["is_verified"]           = False
            t["pending_action"]        = None
            t["mitigation_progress"]   = 0
            t["steps_since_contained"] = 0
            t["age"]                   = 0
            t["stage"]                 = "initial"
            t["escalated"]             = False
            t["spread_attempted"]      = False
            # Each resurface makes the threat harder to fully contain
            t["resurface_risk"] = round(min(1.0, t["resurface_risk"] + 0.20), 3)
            t["severity"]       = round(min(1.0, t.get("severity", 0.5) + 0.15), 3)
            # Immediate health cost — resurfaced threat already did damage during dormancy
            sess.state["system_health"] = max(0, sess.state["system_health"] - 12)


def _compute_ambiguity(t: dict) -> tuple[float, bool]:
    """Compute (signal_ambiguity_score, contradicting_ioc) for a visible threat.

    signal_ambiguity_score ∈ [0, 1]: fraction of IOC fields that fall into overlap
    regions shared by two or more attack classes.  High values indicate the agent
    cannot reliably classify the threat from a single IOC axis and must reason
    across the full vector — or verify before acting.

    contradicting_ioc: True when the IOC vector simultaneously suggests attack classes
    with DIFFERENT correct mitigations.  For example, high pps (ddos → patch) combined
    with high auth failures (phishing/lateral → block_ip) is a genuine contradiction
    that no threshold classifier can resolve — the agent must use multi-signal inference.

    These fields are exposed in visible_threats so agents can explicitly reason about
    uncertainty and decide whether to scan/verify before committing a mitigation.
    """
    pps  = t.get("packets_per_second",       0)
    auth = t.get("failed_auth_attempts",     0)
    ob   = t.get("outbound_data_bytes",      0)
    lat  = t.get("lateral_connection_count", 0)
    proc = t.get("unusual_process_count",    0)

    overlap_count = 0
    # pps: malware(20-300) ∩ ddos(150-4000) at 150-300
    if 150 <= pps <= 300:
        overlap_count += 1
    # pps: phishing(5-100) ∩ malware(20-300) at 20-100
    elif 20 <= pps <= 100:
        overlap_count += 1
    # auth: phishing(30-170) ∩ lateral(15-120) at 30-120
    if 30 <= auth <= 120:
        overlap_count += 1
    # ob: malware(1000-7000) ∩ ransomware(3500-18000) at 3500-7000
    if 3500 <= ob <= 7000:
        overlap_count += 1
    # lat: ransomware(5-24) ∩ lateral(4-22) at 5-22
    if 5 <= lat <= 22:
        overlap_count += 1
    # proc: malware(6-28) ∩ ransomware(10-32) at 10-28
    if 10 <= proc <= 28:
        overlap_count += 1

    ambiguity = round(min(1.0, overlap_count / 5), 3)

    # Contradicting signals: the vector simultaneously implicates classes
    # requiring DIFFERENT correct actions (patch vs block vs isolate).
    suggests_patch   = pps > 500                        # high pps → ddos → patch
    suggests_block   = auth > 80 or (15 <= lat <= 80)  # phishing/lateral → block_ip
    suggests_isolate = (proc >= 10 and ob >= 1000)     # malware/ransomware → isolate
    contradicting = sum([suggests_patch, suggests_block, suggests_isolate]) >= 2

    return ambiguity, contradicting


def _visible_threats(sess: Session) -> list:
    """Build the agent-facing threat list.

    Exposes ONLY behavioral IOC signals. type/original_type are intentionally
    withheld per spec — agents must classify from packet rates, auth failures,
    lateral connections, and outbound bytes, not from ground-truth labels.
    """
    out = []
    for t in sess.state["threats"]:
        if t["visible"] and not t.get("contained"):
            mp = int(t.get("mitigation_progress", 0))
            # Kill chain stage name (exposes activity label, NOT attack type)
            _kc_chain = KILL_CHAIN_STAGES.get(t.get("original_type", t["type"]), [("initial_access", 1.0, {})])
            _kc_idx   = int(t.get("kill_chain_stage_idx", 0))
            _kc_stage_name = _kc_chain[min(_kc_idx, len(_kc_chain) - 1)][0]
            # Ambiguity and contradicting IOC signals
            _ambiguity, _contradicting = _compute_ambiguity(t)
            out.append({
                "id":                     str(t.get("id", "unknown")),
                "node":                   str(t["node"]),
                "stage":                  str(t["stage"]),
                # type/original_type intentionally withheld per spec — agents must
                # classify from behavioral IOC signals, not from a label lookup.
                "age":                    int(t["age"]),
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
                # ── Sequential decision signals ──────────────────────────────
                # is_verified: True after verify_node_X was called on this threat's node.
                # Verify is optional — earns +0.15 reward; no penalty for skipping it.
                "is_verified":            bool(t.get("is_verified", False)),
                # mitigation_in_progress: > 0 means a delayed patch is running.
                # Sending another mitigation action to this threat is wasted.
                "mitigation_in_progress": mp > 0,
                "mitigation_steps_left":  mp,
                # resurface_risk: visible after re-detection so agents know
                # which contained threats need monitoring (shown in episode_info).
                "resurface_risk":         round(float(t.get("resurface_risk", 0.0)), 3),
                # ── Kill chain signals ───────────────────────────────────────
                # kill_chain_stage: current named stage in the attack progression.
                # Exposes WHAT the attack is doing — not the threat type.
                # Agents must map stage behaviour to threat class via IOC signals.
                "kill_chain_stage":   _kc_stage_name,
                "kill_chain_depth":   int(t.get("kill_chain_stage_idx", 0)),
                # health_drain_multiplier: how much damage this threat deals per step
                # relative to baseline.  Rises as kill chain advances.
                # Agents should prioritise high-multiplier threats on critical nodes.
                "health_drain_multiplier": round(float(t.get("kill_chain_health_multiplier", 1.0)), 2),
                # ── Partial observability: ambiguity signals ─────────────────
                # signal_ambiguity_score: fraction of IOC fields in cross-class
                # overlap regions [0, 1].  High = harder to classify without verify.
                "signal_ambiguity_score": _ambiguity,
                # contradicting_ioc: True when IOC fields point to 2+ different
                # correct mitigations — e.g. high pps (patch) + high auth (block).
                # Agents should verify before acting on contradicting threats.
                "contradicting_ioc":      _contradicting,
                # node_criticality: how important this node is to total system health.
                # High criticality → act here first when resources are scarce.
                "node_criticality":   NODE_CRITICALITY.get(t["node"], 0.6),
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
    # Use scaled costs so resources_remaining matches the grader's view.
    # Scan/verify costs are difficulty-scaled; monitor is flat.
    _scaled_obs       = get_scaled_costs(_task_budget)
    _scan_cost_obs    = _SCAN_COST_BY_DIFFICULTY.get(sess.task_name, 0.20)
    _verify_cost_obs  = _VERIFY_COST_BY_DIFFICULTY.get(sess.task_name, 0.15)
    def _action_cost_obs(a: str) -> float:
        if a.startswith("scan"):    return _scan_cost_obs
        if a.startswith("verify"):  return _verify_cost_obs
        if a.startswith("monitor"): return _MONITOR_COST
        return _scaled_obs.get(a, 0.0)
    _total_spent = sum(_action_cost_obs(a) for a in sess.episode_actions_taken)
    _total_budget = max(0.01, _task_budget * sess.task_config.get("max_steps", 50))
    _resources_remaining = round(max(0.0, 1.0 - _total_spent / _total_budget), 3)
    _speed_bonus = _compute_speed_bonus(sess.containment_events)
    _fp_penalty_obs = min(1.0, sess.false_positive_actions / max(1, _total))
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
            # ── Sequential decision state summary ────────────────────────────
            # Agents should use these to plan verify/monitor/patch-chain actions.
            "pending_mitigations": [
                {"threat_id": t["id"], "action": t["pending_action"],
                 "steps_left": t["mitigation_progress"]}
                for t in sess.state["threats"]
                if t.get("pending_action") and not t.get("contained")
            ],
            "resurfaceable_threats": [
                {"threat_id": t["id"], "node": t["node"],
                 "resurface_risk": round(t.get("resurface_risk", 0.0), 3),
                 "steps_since_contained": t.get("steps_since_contained", 0)}
                for t in sess.state["threats"]
                if t.get("contained") and t.get("resurface_risk", 0.0) > 0
            ],
            "unconfirmed_visible_count": sum(
                1 for t in sess.state["threats"]
                if t.get("visible") and not t.get("contained")
                and not t.get("is_false_positive") and not t.get("is_verified")
            ),
            "grader_breakdown": {
                "containment": _containment_rate,
                "health": _critical_health,
                "resource": _resources_remaining,
                "speed": _speed_bonus,
                "fp_penalty": round(_fp_penalty_obs, 4),
            },
            "network_topology": {
                # Live graph state — edges are fixed; node_status reflects current compromise/scan state.
                "nodes": NODES,
                "edges": {k: sorted(v) for k, v in ADJACENCY.items()},
                "node_criticality": dict(NODE_CRITICALITY),  # static; agents can use for prioritisation
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
                # Kill chain depth per node — how far the attack has progressed.
                # 0 = initial_access, 3+ = critical stage.  -1 = no active threat.
                "node_kill_chain_depth": {
                    n: max(
                        (int(t.get("kill_chain_stage_idx", 0))
                         for t in sess.state["threats"]
                         if t["node"] == n and not t.get("contained") and not t.get("is_false_positive")),
                        default=-1
                    )
                    for n in NODES
                },
            },
        },
    }


def _build_reason(action: str, matched: bool, threat_type: str | None, early: bool) -> tuple[str, float]:
    """Return (reason_string, confidence) — no ground-truth threat information leaked."""
    if action.startswith("scan"):
        return ("Action processed.", 0.85)

    if threat_type is None:
        return ("Action processed.", 0.60)

    if matched:
        return ("Action applied. Monitoring threat indicators.", 0.92)
    elif action == "ignore":
        return ("No action taken. System health degraded.", 0.20)
    else:
        return ("Action did not contain the threat. Review behavioral signals.", 0.35)


def safe_response(obs, action, reward=0.001, reason="", confidence=0.0, error=None):
    score = safe_score(obs.get("score", 0.0001))
    resp = {
        "action":           action,
        "reward":           max(0.001, min(0.999, round(float(reward), 3))),
        "visible_threats":  obs.get("visible_threats", []),
        "hidden_threat_count": obs.get("hidden_threat_count", TOTAL_NODES),
        "scan_coverage":    obs.get("scan_coverage", 0.0),
        "system_health":    obs.get("system_health", 100),
        "score":            score,
        "grader_score":     safe_score(obs.get("grader_score", score)),
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
    "score": 0.0001, "grader_score": 0.0001, "step": 0, "done": False,
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if DEBUG:
        log.debug(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=200,
        content=safe_response(_EMPTY_OBS, action="", reward=0.001,
                               reason="Invalid input received. Action rejected.",
                               confidence=0.0, error="invalid action"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=200,
        content=safe_response(_EMPTY_OBS, action="", reward=0.001,
                               reason="Internal error. State preserved.",
                               confidence=0.0, error="internal error"),
    )


# ─── INPUT MODELS ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task:         str  = "easy"
    seed:         int  = 0
    session_id:   str | None = None   # optional; omit for auto-generated UUID
    adversarial:  bool = False        # if True, apply adversarial scenario generator


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


@app.get("/_stcore/health")
def health():
    """Liveness probe for HF Spaces and OpenEnv evaluators. Must return plain-text 'ok'."""
    return Response(content="ok", media_type="text/plain")


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
    # XOR the global attacker seed with the episode seed so every session gets a
    # distinct AdaptiveAttacker initial state.  Using a constant seed (42) for all
    # sessions allowed an adversarial agent to predict the attacker's episode-2 strategy
    # pivot across concurrent evaluation sessions — defeating the adaptive red-team goal.
    # XOR preserves the global seed's contribution while adding per-session variation.
    sess.attacker = AdaptiveAttacker(seed=_ATTACKER_SEED ^ (seed & 0xFFFFFFFF))
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

    # Apply adversarial scenario — only when caller explicitly opts in (default=False)
    if req and req.adversarial:
        _adv_config = _adv_generator.generate(seed)
        _adv_generator.apply_to_session(sess, _adv_config)

    _SESSIONS[sid] = sess
    global _LATEST_SID
    _LATEST_SID = sid

    obs = _obs(sess)
    obs["task"]              = task_name
    obs["session_id"]        = sid          # always returned so callers can track it
    obs["attacker_strategy"] = strategy
    obs["adversarial_mode"]  = bool(req and req.adversarial)
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
            "score": 0.0001, "grader_score": 0.0001, "step": 0, "done": False,
            "error": "session_id required. Call /reset first.",
        })
    try:
        return Observation(**_obs(sess))
    except Exception as e:
        log.warning(f"get_state() snapshot error (transient): {e}")
        return JSONResponse(status_code=200, content={
            "visible_threats": [], "hidden_threat_count": 0,
            "scan_coverage": 0.0, "system_health": sess.state.get("system_health", 100),
            "score": 0.0001, "grader_score": 0.0001,
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
                "action": "", "reward": 0.001, "reason": "No active session. Call /reset first.",
                "confidence": 0.0, "done": False, "error": "no_active_session",
                "score": 0.0001, "grader_score": 0.0001, "step": 0,
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
                _obs(sess), action="", reward=0.001,
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
            s["score"] = safe_score(round(sum(_all_r) / len(_all_r), 4))
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
        scan_revealed_real = False       # scan uncovered at least one genuine threat
        scan_revealed_fp   = False       # scan uncovered only false-positive alerts
        scan_node_already_contained = False  # node has only already-contained threats
        # ── Sequential decision tracking ─────────────────────────────────────
        verify_found_threat  = False     # verify_node revealed an unverified threat
        verify_already_done  = False     # verify on a node where threats already verified
        monitor_reduced_risk = False     # monitor_node reduced a resurface risk
        monitor_nothing      = False     # monitor on node with no resurfaceable threats
        patch_queued         = False     # patch action queued (delayed — not instant)

        # ── RESOURCE CHECK ──
        # Compute current resources remaining to enforce budget constraints.
        # Scan/verify costs are difficulty-scaled; monitor is flat.
        _task_budget      = max(0.01, sess.task_config.get("resource_per_step", 1.0))
        _scan_cost_now    = _SCAN_COST_BY_DIFFICULTY.get(sess.task_name, 0.20)
        _verify_cost_now  = _VERIFY_COST_BY_DIFFICULTY.get(sess.task_name, 0.15)
        def _action_cost_step(a: str) -> float:
            if a.startswith("scan"):    return _scan_cost_now
            if a.startswith("verify"):  return _verify_cost_now
            if a.startswith("monitor"): return _MONITOR_COST
            return COST_RAW.get(a, 0.0)
        _total_spent_now   = sum(_action_cost_step(a) for a in sess.episode_actions_taken)
        _total_budget_now  = max(0.01, _task_budget * sess.task_config.get("max_steps", 50))
        _resources_now     = max(0.0, 1.0 - _total_spent_now / _total_budget_now)
        _resource_exhausted = _resources_now <= 0.0

        # ── SCAN ──
        if raw_action.startswith("scan"):
            node = raw_action[len("scan_"):] if raw_action.startswith("scan_") else ""
            if node in NODES:
                s["scanned_nodes"].add(node)
                # Scan effectiveness degrades at 50% when budget exhausted
                false_neg = sess.task_config.get("false_negative_rate", 0.0)
                if _resource_exhausted:
                    false_neg = min(0.99, false_neg + 0.5)
                for t in s["threats"]:
                    if t["node"] == node and not t["visible"] and not t.get("contained"):
                        if sess.rng.random() > false_neg:
                            t["visible"] = True
                            # Apply scan-reveal IOC noise: the initial scan gives an imperfect
                            # snapshot — ±SCAN_IOC_NOISE_FACTOR noise on every numeric IOC.
                            # Without this, one scan gives a perfect classification signal;
                            # agents must track IOC evolution over subsequent steps to
                            # reduce uncertainty and confirm the correct mitigation.
                            if not t.get("is_false_positive"):
                                _nf = _SCAN_IOC_NOISE_FACTOR
                                for _ioc_key in ("packets_per_second", "failed_auth_attempts",
                                                  "outbound_data_bytes", "lateral_connection_count",
                                                  "unusual_process_count"):
                                    _val = t.get(_ioc_key, 0)
                                    t[_ioc_key] = max(0, int(_val * (1 + sess.rng.uniform(-_nf, _nf))))
                            if t.get("is_false_positive"):
                                scan_revealed_fp = True
                            else:
                                scan_revealed_real = True
                reason, confidence = _build_reason(raw_action, False, None, False)
                if scan_revealed_fp:
                    reason = "Scan complete. Low-severity anomaly detected."
                    confidence = 0.90
                elif scan_revealed_real:
                    reason = "Scan complete. Threat indicators present."
                    confidence = 0.90
                else:
                    # Nothing revealed — check if node already fully dealt with
                    scan_node_already_contained = any(
                        t["node"] == node and t.get("contained") and not t.get("is_false_positive")
                        for t in s["threats"]
                    )
                    reason = "Scan complete. No active indicators found."
                    confidence = 0.75
                    scan_found_nothing = True
            else:
                reason = f"'{node}' is not a valid node. Valid nodes: node_1 through node_5."
                confidence = 0.10

        # ── VERIFY ──
        elif raw_action.startswith("verify"):
            node = raw_action[len("verify_"):] if raw_action.startswith("verify_") else ""
            if node in NODES:
                # Mark all unverified visible real threats on this node as verified.
                # Verify does not reveal hidden threats — use scan for that.
                found_unverified = False
                for t in s["threats"]:
                    if (t["node"] == node and t.get("visible") and not t.get("contained")
                            and not t.get("is_false_positive") and not t.get("is_verified")):
                        t["is_verified"] = True
                        found_unverified = True
                if found_unverified:
                    verify_found_threat = True
                    reason = "Verification complete. Threat confirmed. Safe to mitigate."
                    confidence = 0.92
                else:
                    # Nothing to verify — already verified or no visible threat
                    verify_already_done = True
                    reason = "Verification complete. No unverified threats on this node."
                    confidence = 0.70
            else:
                reason = f"'{node}' is not a valid node. Valid nodes: node_1 through node_5."
                confidence = 0.10

        # ── MONITOR ──
        elif raw_action.startswith("monitor"):
            node = raw_action[len("monitor_"):] if raw_action.startswith("monitor_") else ""
            if node in NODES:
                # Reduce resurface_risk for contained persistent threats on this node.
                found_resurfaceable = False
                for t in s["threats"]:
                    if t["node"] == node and t.get("contained") and t.get("resurface_risk", 0.0) > 0:
                        # Each monitor step cuts resurface_risk by 40%, flooring at 0
                        t["resurface_risk"] = round(max(0.0, t["resurface_risk"] - 0.40), 3)
                        t["steps_since_contained"] = 0  # reset the resurface clock
                        found_resurfaceable = True
                if found_resurfaceable:
                    monitor_reduced_risk = True
                    reason = "Monitoring active. Resurface risk reduced."
                    confidence = 0.88
                else:
                    monitor_nothing = True
                    reason = "Monitoring complete. No persistent threats require attention on this node."
                    confidence = 0.65
            else:
                reason = f"'{node}' is not a valid node. Valid nodes: node_1 through node_5."
                confidence = 0.10

        # ── DEFENSE ──
        else:
            # Pre-roll resource-exhaustion dice unconditionally so the session RNG
            # advances exactly once per defense step, keeping the sequence stable
            # regardless of whether _resource_exhausted is True this step.
            _exhaust_roll = sess.rng.random()
            _ = sess.rng.random()  # consume slot for RNG-sequence stability (was: unverified penalty roll)

            for t in s["threats"]:
                if t["visible"] and not t.get("contained"):
                    # Skip threats already being mitigated by a queued patch.
                    # Use continue (not break) so subsequent visible threats can still be targeted.
                    # A break here would silently block the agent from acting on a second visible
                    # threat that does NOT have a pending action — the worst-case scenario is two
                    # visible threats where the first has a pending patch and the second is actively
                    # spreading to a hub node: the defender could never contain the second one.
                    if t.get("pending_action"):
                        continue
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
                        elif _resource_exhausted and _exhaust_roll < 0.5:
                            # Resource exhausted: 50% chance action fails — forces budget planning
                            matched_threat_type = t["type"]
                            matched = False  # action attempted but resource-starved response failed
                            reason = "Action failed. Resource budget depleted."
                            confidence = 0.30
                        else:
                            # ── Successful mitigation ────────────────────────────────────
                            delay = _MITIGATION_DELAY.get(raw_action, 0)
                            matched_threat_type = t.get("original_type", t["type"])
                            matched = True
                            early_bonus = t["age"] < 3
                            if delay > 0:
                                # Delayed mitigation (patch): mark in-progress, don't contain yet
                                t["pending_action"]      = raw_action
                                t["mitigation_progress"] = delay
                                patch_queued = True
                                reason = (f"Patch deployment started. Completes in {delay} step(s). "
                                          "Threat remains active during deployment.")
                                confidence = 0.80
                            else:
                                # Instant mitigation (block_ip, isolate_machine)
                                t["contained"] = True
                                sess.containment_events.append({
                                    "threat_id":          t["id"],
                                    "age_at_containment": t["age"],
                                    "threat_type":        t.get("original_type", t["type"]),
                                })
                        break

            if not matched:
                for t in s["threats"]:
                    if t["visible"] and not t.get("contained"):
                        matched_threat_type = t.get("original_type", t.get("type"))
                        break
                # Only penalise ignore when there was an active visible threat to act on.
                # Penalising ignore when no threats are visible punishes a correct no-op
                # (e.g. step 0 before any threat becomes visible) and corrupts health
                # baselines for zero-threat and early-episode edge cases.
                if raw_action == "ignore" and matched_threat_type is not None:
                    s["system_health"] = max(0, s["system_health"] - 10)
                elif not patch_queued:
                    s["system_health"] = max(0, s["system_health"] - 5)

            if not reason:
                reason, confidence = _build_reason(raw_action, matched, matched_threat_type, early_bonus)

        # Criticality-amplified, kill-chain-weighted health degradation.
        # Each active real threat contributes damage proportional to:
        #   degrade_rate × node_criticality × kill_chain_health_multiplier × 20
        # (×20 normalises across 5 nodes so base damage matches the old uniform formula
        #  when criticality=0.6 and multiplier=1.0 ≈ 0.6×20=12 HP per 5 threats).
        # This forces agents to prioritise high-criticality hub nodes and act early —
        # ransomware at "encryption" stage on node_2 deals 4× normal damage per step.
        _degrade_rate = sess.task_config.get("health_degradation_rate", 0.0)
        if _degrade_rate > 0:
            _health_loss = 0.0
            for _t in s["threats"]:
                if not _t.get("contained") and not _t.get("is_false_positive"):
                    _node_crit = NODE_CRITICALITY.get(_t.get("node", "node_1"), 0.6)
                    _kc_mult   = float(_t.get("kill_chain_health_multiplier", 1.0))
                    _health_loss += _degrade_rate * _node_crit * _kc_mult * 20
            s["system_health"] = max(0, s["system_health"] - _health_loss)

        _age_threats(sess)
        _update_visibility(sess)
        _clamp_health(sess)

        # Reward authority: MITRE-aligned lookup table extended for sequential chain.
        #
        # Scan (direct, not normalized — genuine cost for wasted scans):
        #   real threat revealed               → +0.25
        #   false positive revealed            → -0.05
        #   clean node                         → -0.10
        #   already-contained node             → -0.20
        #   scan while resources exhausted     → -0.30
        #
        # Verify (direct — rewards disciplined confirmation before acting):
        #   unconfirmed threat confirmed       → +0.15
        #   nothing new to verify              → -0.08  (wasted step)
        #
        # Monitor (direct — rewards post-containment vigilance):
        #   resurface risk reduced             → +0.10
        #   nothing resurfaceable              → -0.08  (wasted step)
        #
        # Mitigation (normalized via _clamp_reward((r + 2.0) / 4.0) → (0, 1)):
        #   correct + instant (age < 3 early)  → raw 1.1 → 0.775
        #   correct + instant                  → raw 1.0 → 0.750
        #   correct + patch queued             → raw 0.80 → 0.700  (delay accepted)
        #   wrong action                       → raw -0.50 → 0.375
        #   ignore with visible threat         → raw -1.50 → 0.125
        if raw_action.startswith("scan"):
            if _resource_exhausted:
                reward = -0.30
            elif scan_revealed_real:
                reward = +0.25
            elif scan_revealed_fp:
                reward = -0.05
            elif scan_node_already_contained:
                reward = -0.20
            else:
                reward = -0.10  # clean node, nothing to find

        elif raw_action.startswith("verify"):
            if verify_found_threat:
                reward = +0.15
            else:
                reward = -0.08  # wasted — already verified or no visible threat

        elif raw_action.startswith("monitor"):
            if monitor_reduced_risk:
                reward = +0.10
            else:
                reward = -0.08  # wasted — nothing resurfaceable here

        elif raw_action == "ignore":
            reward = _clamp_reward(-1.5)

        elif raw_action in ("block_ip", "isolate_machine", "patch"):
            if matched and patch_queued:
                # Delayed mitigation accepted: moderate reward — agent must follow through
                reward = _clamp_reward(0.80)
            elif matched:
                # Instant containment: full reward with early-act bonus
                reward = _clamp_reward(1.1 if early_bonus else 1.0)
            else:
                reward = _clamp_reward(-0.50)
        else:
            reward = _clamp_reward(-0.50)

        # Running average score — clamped to strict (0, 1) via safe_score
        _all_rewards = sess.episode_rewards + [reward]
        s["score"] = safe_score(round(sum(_all_rewards) / len(_all_rewards), 4))
        s["step"] += 1

        if s["system_health"] <= 0 or s["step"] >= sess.task_config.get("max_steps", 50):
            s["done"] = True

        # Append the current action and reward BEFORE calling _obs() so that
        # _compute_grader_score() sees the complete action history including this step.
        # The original code appended after _obs(), causing resource_efficiency in the
        # step response to be one action stale — the grader under-counted spent resources,
        # making resource_efficiency artificially inflated on every non-final step.
        sess.episode_actions_taken.append(raw_action)
        sess.action_counts[raw_action] = sess.action_counts.get(raw_action, 0) + 1
        sess.episode_rewards.append(reward)

        obs = _obs(sess)

        sess.history.append({"step": s["step"], "action": raw_action,
                              "reward": round(reward, 3), "attack": matched_threat_type})
        sess.episode_history.append({"step": s["step"], "action": raw_action,
                                     "reward": float(reward), "done": bool(s["done"]),
                                     "reason": reason})
        for threat in obs.get("visible_threats", []):
            tid = threat.get("id", "")
            if tid:
                sess.threats_detected.add(tid)
        for t in s["threats"]:
            # Exclude false positives so analytics containment_rate matches the grader formula.
            if t.get("contained") and not t.get("is_false_positive"):
                tid = str(t.get("id", f"{t['type']}_{t['node']}"))
                sess.threats_contained.add(tid)

        # Red team
        translated = translate_action(raw_action)
        threat_ctx = (matched_threat_type or "UNKNOWN").upper()
        sess.attacker.observe_defender_action(translated, threat_ctx)
        # Adversarial mid-episode strategy switch — no-op for normal sessions
        _adv_generator.maybe_switch_strategy(sess)

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
        return safe_response(obs, action="", reward=0.001,
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

        action_counts: dict = dict(sess.action_counts)
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
        # Must use difficulty-scaled scan/verify costs (same as grader and step layer)
        # so that resources_remaining here matches what the agent sees in /step responses.
        # The original code used a hardcoded 0.2 scan fallback — on hard (scan_cost=0.45)
        # this understated scan spending by 2.25× and inflated resources_remaining.
        task_budget = max(0.01, sess.task_config.get("resource_per_step", 1.0))
        _scan_cost_anal   = _SCAN_COST_BY_DIFFICULTY.get(sess.task_name, 0.20)
        _verify_cost_anal = _VERIFY_COST_BY_DIFFICULTY.get(sess.task_name, 0.15)
        total_spent = sum(
            (_scan_cost_anal   if a.startswith("scan")    else
             _verify_cost_anal if a.startswith("verify")  else
             _MONITOR_COST     if a.startswith("monitor") else
             COST_RAW.get(a, 0.0))
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


@app.get("/observe")
def observe(session_id: str | None = None):
    """Single-call enriched observation: state + threat-intel + analytics in one round trip."""
    sess = _get_session(session_id)
    if sess is None:
        return JSONResponse(status_code=200, content={
            "error": "session_id required. Call /reset first.",
            "observation": {}, "threat_intel": {}, "analytics": {}, "recommended": "ignore",
        })
    try:
        _validate_session_state(sess)
        observation  = _obs(sess)
        visible      = observation.get("visible_threats", [])
        scan_cov     = observation.get("scan_coverage", 0.0)
        sys_health   = sess.state["system_health"]

        # ── threat_intel (inline — no second network call) ───────────────────
        active_campaigns = []
        for threat in visible:
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
                "threat_id": threat.get("id", "unknown"),
                "node":      threat.get("node", "unknown"),
                "stage":     threat.get("stage", "unknown"),
                "age":       threat.get("age", 0),
                "severity":  ioc_severity,
                "confidence": round(threat.get("detection_confidence", 0.8), 3),
                "urgency":   "IMMEDIATE" if threat.get("age", 0) >= 3 else "MONITOR",
            })
        active_campaigns.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 4))
        compromised_ti = list({
            t["node"] for t in sess.state["threats"]
            if t.get("stage") == "lateral_movement" and not t.get("contained")
        })
        if sys_health < 30 or len(compromised_ti) >= 3:
            risk_level = "CRITICAL"
        elif sys_health < 60 or len(compromised_ti) >= 2:
            risk_level = "HIGH"
        elif sys_health < 80 or len(compromised_ti) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        threat_intel = {
            "risk_level":     risk_level,
            "active_campaigns": active_campaigns,
            "threat_summary": {
                "total_visible":    len(visible),
                "critical_count":   sum(1 for t in active_campaigns if t["severity"] == "CRITICAL"),
                "immediate_action": sum(1 for t in active_campaigns if t["urgency"] == "IMMEDIATE"),
            },
        }

        # ── analytics (inline — no third network call) ───────────────────────
        action_counts = dict(sess.action_counts)
        most_used     = max(action_counts, key=action_counts.get) if action_counts else "none"
        n_contained   = len(sess.threats_contained)
        spawned       = max(1, len(sess.state["threats"]))
        containment_rate = round(n_contained / spawned, 3)
        task_budget   = max(0.01, sess.task_config.get("resource_per_step", 1.0))
        _scan_cost_obs2   = _SCAN_COST_BY_DIFFICULTY.get(sess.task_name, 0.20)
        _verify_cost_obs2 = _VERIFY_COST_BY_DIFFICULTY.get(sess.task_name, 0.15)
        total_spent   = sum(
            (_scan_cost_obs2   if a.startswith("scan")    else
             _verify_cost_obs2 if a.startswith("verify")  else
             _MONITOR_COST     if a.startswith("monitor") else
             COST_RAW.get(a, 0.0))
            for a in sess.episode_actions_taken)
        total_budget  = max(0.01, task_budget * sess.task_config.get("max_steps", 50))
        resources_remaining = round(max(0.0, 1.0 - total_spent / total_budget), 3)
        if containment_rate >= 0.8 and sys_health >= 70:
            grade = "A"
        elif containment_rate >= 0.6 and sys_health >= 50:
            grade = "B"
        elif containment_rate >= 0.4 and sys_health >= 30:
            grade = "C"
        else:
            grade = "D"
        # Recommended action from IOC signals
        _any = lambda key, thr: any(t.get(key, 0) > thr for t in visible)
        if _any("packets_per_second", 500):
            recommended = "patch"
        elif _any("outbound_data_bytes", 2000) or _any("unusual_process_count", 5):
            recommended = "isolate_machine"
        elif _any("failed_auth_attempts", 20) or _any("lateral_connection_count", 8):
            recommended = "block_ip"
        elif scan_cov < 1.0:
            unscanned = [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)
                         if f"node_{i}" not in sess.state["scanned_nodes"]]
            recommended = unscanned[0] if unscanned else "ignore"
        else:
            recommended = "ignore"
        analytics = {
            "performance_grade":    grade,
            "soc_metrics":          {"containment_rate": containment_rate},
            "resources_remaining":  resources_remaining,
            "attacker_strategy":    sess.attacker.current_strategy,
            "agent_behavior":       {"action_breakdown": action_counts, "most_used_action": most_used},
            "network_status":       {"system_health": sys_health, "scan_coverage": scan_cov},
        }

        return {
            "observation": observation,
            "threat_intel": threat_intel,
            "analytics":    analytics,
            "recommended":  recommended,
        }
    except Exception as e:
        log.error(f"/observe error: {e}", exc_info=True)
        return JSONResponse(status_code=200, content={
            "error": str(e), "observation": {}, "threat_intel": {}, "analytics": {}, "recommended": "ignore",
        })


def main():
    import uvicorn
    # Port 7860 matches openenv.yaml docker.port and Dockerfile EXPOSE
    port = int(_os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
