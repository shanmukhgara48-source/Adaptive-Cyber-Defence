"""
IMPOSSIBLE Task: AI-driven attacker with perfect counter-strategy.

Scenario
--------
Full network compromise by an AI-driven attacker that perfectly counters
every defensive action. Maximum threats, near-zero detection, instant
progression, no resources.

A random agent scores ~0.05. Even a perfect agent scores ~0.15.
This task exists to demonstrate the environment has no ceiling.

Win condition: survive — do not let all assets reach zero health.
Passing score: 0.10 (exists to show environment has no ceiling)
"""

from .base import BaseTask, TaskConfig


class ImpossibleTask(BaseTask):
    config = TaskConfig(
        name="impossible_ai_attacker",
        difficulty="impossible",
        description=(
            "AI-driven attacker with perfect counter-strategy. "
            "Maximum evasion, instant kill chain, zero resources. "
            "Random agent scores ~0.05. Exists to show environment has no ceiling."
        ),
        max_steps=10,
        initial_threat_count=5,
        resource_per_step=0.10,
        attack_progression_prob=0.70,
        lateral_spread_base_prob=0.80,
        natural_severity_growth=0.15,
        health_degradation_rate=0.20,
        false_positive_rate=0.50,
        false_negative_rate=0.80,
        base_detection_prob=0.05,
        passing_score=0.10,
    )
