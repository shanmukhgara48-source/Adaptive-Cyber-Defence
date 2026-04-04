"""
MEDIUM Task: Multi-stage attack, limited resources.

Scenario
--------
Two simultaneous intrusions — one via phishing, one via a known vulnerability
on the web server.  Resources are constrained: the SOC must prioritise which
threat to address first.  Detection confidence is moderate and false positives
add noise to the alert queue.

Win condition: contain both threats before exfiltration occurs on any critical
asset, without exhausting resources mid-episode.
Passing score: 0.55
"""

from .base import BaseTask, TaskConfig


class MediumTask(BaseTask):
    config = TaskConfig(
        name="medium_multi_stage",
        difficulty="medium",
        description=(
            "Two simultaneous intrusions with limited SOC resources. "
            "Moderate detection with false-positive noise. "
            "Requires threat prioritisation — can't address everything."
        ),
        max_steps=50,
        initial_threat_count=2,
        resource_per_step=0.60,         # tighter budget
        attack_progression_prob=0.25,
        lateral_spread_base_prob=0.30,
        natural_severity_growth=0.03,
        health_degradation_rate=0.05,
        false_positive_rate=0.12,       # moderate alert noise
        false_negative_rate=0.30,
        base_detection_prob=0.45,
        passing_score=0.55,
    )
