"""
EASY Task: Single-stage attack, high detection probability.

Scenario
--------
A single workstation has been targeted by a phishing campaign.
The SOC has high-quality logging and ample resources.
The attack progresses slowly, giving the defender time to respond.

Win condition: contain the threat before it reaches LATERAL_SPREAD.
Passing score: 0.55
"""

from .base import BaseTask, TaskConfig


class EasyTask(BaseTask):
    config = TaskConfig(
        name="easy_single_stage",
        difficulty="easy",
        description=(
            "Single phishing attack on one workstation. "
            "High detection probability, generous resources, slow progression. "
            "Goal: contain before lateral spread."
        ),
        max_steps=30,
        initial_threat_count=3,
        resource_per_step=1.5,          # 50% more resources than default
        attack_progression_prob=0.08,   # slower kill-chain — agent has time to respond
        lateral_spread_base_prob=0.05,  # very low spread risk
        natural_severity_growth=0.02,
        health_degradation_rate=0.03,
        false_positive_rate=0.05,       # minimal noise
        false_negative_rate=0.08,
        base_detection_prob=0.95,       # near-perfect detection — threats visible fast
        passing_score=0.55,
    )
