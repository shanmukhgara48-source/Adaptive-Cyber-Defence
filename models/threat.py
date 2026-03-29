"""
models/threat.py — MITRE ATT&CK mapping for the kill-chain stages.
"""
from __future__ import annotations
from typing import Dict

# Full MITRE ATT&CK map keyed by canonical stage name and common aliases
MITRE_ATTACK_MAP: Dict[str, Dict[str, str]] = {
    "PHISHING": {
        "technique_id":   "T1566",
        "technique_name": "Phishing",
        "tactic":         "Initial Access",
        "tactic_id":      "TA0001",
    },
    "CREDENTIAL_ACCESS": {
        "technique_id":   "T1078",
        "technique_name": "Valid Accounts",
        "tactic":         "Credential Access",
        "tactic_id":      "TA0006",
    },
    "MALWARE_INSTALL": {
        "technique_id":   "T1204",
        "technique_name": "User Execution",
        "tactic":         "Execution",
        "tactic_id":      "TA0002",
    },
    "LATERAL_SPREAD": {
        "technique_id":   "T1021",
        "technique_name": "Remote Services",
        "tactic":         "Lateral Movement",
        "tactic_id":      "TA0008",
    },
    "EXFILTRATION": {
        "technique_id":   "T1041",
        "technique_name": "Exfiltration Over C2 Channel",
        "tactic":         "Exfiltration",
        "tactic_id":      "TA0010",
    },
}

# Aliases so callers can use short names or alternate spellings
_ALIASES: Dict[str, str] = {
    "ACCESS":           "CREDENTIAL_ACCESS",
    "CREDENTIAL":       "CREDENTIAL_ACCESS",
    "MALWARE":          "MALWARE_INSTALL",
    "INSTALL":          "MALWARE_INSTALL",
    "LATERAL":          "LATERAL_SPREAD",
    "LATERAL_MOVEMENT": "LATERAL_SPREAD",
    "SPREAD":           "LATERAL_SPREAD",
    "EXFIL":            "EXFILTRATION",
}

_UNKNOWN = {
    "technique_id":   "T0000",
    "technique_name": "Unknown Technique",
    "tactic":         "Unknown",
    "tactic_id":      "TA0000",
}


def get_mitre_info(stage_name: str) -> Dict[str, str]:
    """
    Return the MITRE ATT&CK info dict for a given kill-chain stage name.

    Accepts canonical names (e.g. "PHISHING") and common aliases
    (e.g. "LATERAL_MOVEMENT" → LATERAL_SPREAD).

    Returns a dict with keys: technique_id, technique_name, tactic, tactic_id.
    Never raises — returns unknown sentinel if stage_name is unrecognised.
    """
    key = stage_name.upper().strip()
    key = _ALIASES.get(key, key)
    return dict(MITRE_ATTACK_MAP.get(key, _UNKNOWN))


def generate_mitre_summary(episode_threats) -> Dict[str, int]:
    """
    Given a list of Threat objects seen during an episode, return a dict
    mapping technique_id → count of threats that used that technique.
    """
    counts: Dict[str, int] = {}
    for t in episode_threats:
        tid = getattr(t, "mitre_technique_id", None) or t.stage.technique_id
        counts[tid] = counts.get(tid, 0) + 1
    return counts
