"""
NIGHTMARE Task: Maximum difficulty — 5 simultaneous threats,
near-zero detection, fastest possible kill chain.

Scenario
--------
A nation-state APT group has pre-positioned inside the network.
All 5 nodes are under simultaneous attack. Detection is almost
impossible — high evasion, maximum false positives, minimum
scan window. The attacker adapts faster than the defender can
respond.

This task is designed to genuinely challenge frontier LLMs.
A random agent scores ~0.10. A perfect agent scores ~0.70.

Win condition: contain at least 3 of 5 threats before
critical asset health reaches zero.
Passing score: 0.25 (intentionally very hard)
"""

from .base import BaseTask, TaskConfig


class NightmareTask(BaseTask):
    config = TaskConfig(
        name="nightmare_apt",
        difficulty="nightmare",
        description=(
            "Nation-state APT. All 5 nodes compromised simultaneously. "
            "Near-zero detection probability, maximum evasion, fastest "
            "kill chain. Designed to challenge frontier LLMs. "
            "Random agent scores ~0.10."
        ),
        max_steps=15,
        initial_threat_count=5,
        resource_per_step=0.25,
        attack_progression_prob=0.45,
        lateral_spread_base_prob=0.50,
        natural_severity_growth=0.08,
        health_degradation_rate=0.12,
        false_positive_rate=0.35,
        false_negative_rate=0.70,
        base_detection_prob=0.10,
        passing_score=0.25,
    )
