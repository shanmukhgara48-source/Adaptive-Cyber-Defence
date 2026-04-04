"""
ELITE Task: Nation-state persistent threat with insider access.

Scenario
--------
A coordinated nation-state attack with insider threat and supply chain
compromise. The attacker has already bypassed perimeter defenses.
All nodes are pre-compromised with dormant malware. Detection is near
impossible. The kill chain advances every single step.

Win condition: prevent total network collapse — contain at least 3 of 5
threats before critical asset health reaches zero.
Passing score: 0.20 (designed for frontier models only)
"""

from .base import BaseTask, TaskConfig


class EliteTask(BaseTask):
    config = TaskConfig(
        name="elite_persistent_threat",
        difficulty="elite",
        description=(
            "Nation-state persistent threat with insider access. "
            "All nodes pre-compromised. Near-zero detection window. "
            "Kill chain advances every step. Designed for frontier models."
        ),
        max_steps=15,
        initial_threat_count=5,
        resource_per_step=0.20,
        attack_progression_prob=0.55,
        lateral_spread_base_prob=0.65,
        natural_severity_growth=0.10,
        health_degradation_rate=0.15,
        false_positive_rate=0.40,
        false_negative_rate=0.70,
        base_detection_prob=0.10,
        passing_score=0.20,
    )
