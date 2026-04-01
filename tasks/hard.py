"""
HARD Task: Multiple simultaneous attacks, low detection confidence, resource constraints.

Scenario
--------
An advanced persistent threat (APT) group has launched a coordinated campaign
against five entry points simultaneously.  The attacker uses evasion techniques
(high false-negative rate) and generates decoy traffic (high false-positive rate)
to confuse the SOC.  Resources are scarce — the team is already stretched thin.

The attack progresses rapidly.  Without decisive action the database will be
exfiltrated within ~8 steps.

Win condition: prevent exfiltration from any critical asset and contain at least
2 of 5 threats.
Passing score: 0.45  (lower bar — this is genuinely hard)
"""

from .base import BaseTask, TaskConfig


class HardTask(BaseTask):
    config = TaskConfig(
        name="hard_apt_campaign",
        difficulty="hard",
        description=(
            "Coordinated APT across 5 entry points. Low detection, high false-positive "
            "noise, scarce resources, fast progression. "
            "Priority: protect critical database assets."
        ),
        max_steps=30,
        initial_threat_count=5,
        resource_per_step=0.55,         # severely limited budget
        attack_progression_prob=0.35,   # fast kill-chain (reduced from 0.38 — old value caused PHISHING→EXFIL in 5 steps)
        lateral_spread_base_prob=0.35,  # aggressive spread
        natural_severity_growth=0.05,
        health_degradation_rate=0.07,
        false_positive_rate=0.20,       # high noise — many ghost alerts
        false_negative_rate=0.45,       # attacker evades detection often
        base_detection_prob=0.25,       # low baseline detection
        passing_score=0.45,
    )
