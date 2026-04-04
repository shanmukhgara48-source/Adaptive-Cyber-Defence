"""
EASY Task: Three-threat scenario with mixed attack types.

Scenario
--------
Three workstations are under simultaneous attack with mixed
threat types (phishing, malware, ransomware, ddos, or
lateral_movement — selected randomly at episode start).
The SOC has high-quality logging and ample resources.
The attacks progress slowly, giving the defender time to respond.

Win condition: contain all threats before they reach LATERAL_SPREAD.
Passing score: 0.50
"""

from .base import BaseTask, TaskConfig


class EasyTask(BaseTask):
    config = TaskConfig(
        name="easy_single_stage",
        difficulty="easy",
        description=(
            "Three simultaneous attacks with mixed threat types. "
            "High detection probability, generous resources, slow progression. "
            "Goal: contain all threats before lateral spread."
        ),
        max_steps=30,
        initial_threat_count=3,
        resource_per_step=1.5,          # 50% more resources than default
        attack_progression_prob=0.15,   # slightly faster kill-chain
        lateral_spread_base_prob=0.12,  # low-moderate spread risk
        natural_severity_growth=0.02,
        health_degradation_rate=0.03,
        false_positive_rate=0.05,       # minimal noise
        false_negative_rate=0.20,       # some threats slip detection
        base_detection_prob=0.70,       # good but not perfect detection
        passing_score=0.50,
    )
