"""
Base task infrastructure for the Adaptive Cyber Defense Simulator.

A Task wraps the environment with difficulty-specific configuration and
produces a normalised episode score [0.0, 1.0] after a full run.

Episode score formula
---------------------
    score = (
        0.50 × containment_rate        # fraction of threats eventually contained
      + 0.20 × survival_rate           # fraction of critical assets still healthy at end
      + 0.15 × resource_efficiency     # average resource leftover fraction per step
      + 0.15 × speed_bonus             # average containment speed (early = better)
    )
    score = clamp(score, 0.0, 1.0)

TaskResult holds all per-episode metrics for benchmarking and comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..env import AdaptiveCyberDefenseEnv


# ---------------------------------------------------------------------------
# Task configuration
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """
    Difficulty-specific configuration for one task variant.
    Applied to EnvConfig when building the environment.
    """
    name: str
    difficulty: str                    # "easy" | "medium" | "hard"
    description: str

    # Episode parameters
    max_steps: int = 50
    initial_threat_count: int = 1

    # Resource budget per step
    resource_per_step: float = 1.0

    # Attack engine parameters
    attack_progression_prob: float = 0.25
    lateral_spread_base_prob: float = 0.20
    natural_severity_growth: float = 0.03
    health_degradation_rate: float = 0.05

    # Detection parameters
    false_positive_rate: float = 0.10
    false_negative_rate: float = 0.15
    base_detection_prob: float = 0.55

    # Scoring: minimum score to "pass" this task level
    passing_score: float = 0.60


# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """
    Full record of one episode run on a task.

    Attributes:
        task_name:           Name of the task.
        seed:                RNG seed used.
        episode_score:       Normalised score [0.0, 1.0].
        passed:              True if score >= task.passing_score.
        steps_taken:         How many steps the episode lasted.
        threats_total:       Total threats spawned (initial + lateral).
        threats_contained:   How many were contained by episode end.
        containment_rate:    threats_contained / threats_total.
        critical_health_end: Average health of critical assets at episode end.
        avg_resource_left:   Average fraction of resources unused per step.
        total_reward:        Sum of all per-step rewards.
        step_rewards:        Per-step reward trace.
        reward_breakdowns:   Per-step RewardBreakdown dicts (optional).
        terminal_reason:     Why the episode ended.
    """
    task_name: str
    seed: int
    episode_score: float
    passed: bool
    steps_taken: int
    threats_total: int
    threats_contained: int
    containment_rate: float
    critical_health_end: float
    avg_resource_left: float
    total_reward: float
    step_rewards: List[float] = field(default_factory=list)
    reward_breakdowns: List[Dict[str, Any]] = field(default_factory=list)
    terminal_reason: str = "max_steps"

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.task_name} seed={self.seed} "
            f"score={self.episode_score:.3f} "
            f"containment={self.containment_rate:.0%} "
            f"steps={self.steps_taken} "
            f"critical_health={self.critical_health_end:.2f}"
        )


# ---------------------------------------------------------------------------
# Base task
# ---------------------------------------------------------------------------

class BaseTask:
    """
    Base class for all task variants.

    Subclasses set `config` and optionally override `build_env()`.

    Usage::

        task = EasyTask()
        result = task.run(agent, seed=42)
        print(result.summary())
    """

    config: TaskConfig = None   # set by subclass

    def build_env(self) -> "AdaptiveCyberDefenseEnv":
        """
        Build an environment configured for this task.
        Subclasses may override to inject custom detection configs.
        """
        from ..env import AdaptiveCyberDefenseEnv, EnvConfig
        from ..engines.detection import DetectionConfig

        cfg = EnvConfig()
        cfg.max_steps              = self.config.max_steps
        cfg.resource_per_step      = self.config.resource_per_step
        cfg.initial_threat_count   = self.config.initial_threat_count
        cfg.attack_progression_prob= self.config.attack_progression_prob
        cfg.lateral_spread_base_prob= self.config.lateral_spread_base_prob
        cfg.natural_severity_growth= self.config.natural_severity_growth
        cfg.health_degradation_rate= self.config.health_degradation_rate
        cfg.false_positive_rate    = self.config.false_positive_rate
        cfg.false_negative_rate    = self.config.false_negative_rate

        env = AdaptiveCyberDefenseEnv(config=cfg)

        # Override detection system with task-specific base detection prob
        from ..engines.detection import DetectionSystem, DetectionConfig
        env._detection_system = DetectionSystem(DetectionConfig(
            base_detection_prob=self.config.base_detection_prob,
            false_positive_rate=self.config.false_positive_rate,
        ))
        env._response_engine.set_detection_system(env._detection_system)
        return env

    def run(self, agent, seed: int = 0) -> TaskResult:
        """
        Run one complete episode with the given agent.

        Args:
            agent: Any object with a `.choose(state) -> ActionInput` method.
            seed:  RNG seed for reproducibility.

        Returns:
            TaskResult with full episode metrics.
        """
        env = self.build_env()
        state = env.reset(seed=seed)

        step_rewards: list[float] = []
        breakdowns: list[dict] = []
        resource_leftovers: list[float] = []
        threats_seen: set[str] = set()
        terminal_reason = "max_steps"

        for t in state.active_threats:
            threats_seen.add(t.id)

        done = False
        while not done:
            action = agent.choose(state)
            state, reward, done, info = env.step(action)

            step_rewards.append(reward)
            if "reward_breakdown" in info:
                breakdowns.append(info["reward_breakdown"])
            resource_leftovers.append(info.get("resource_utilisation", 0.0))

            # Track all threat IDs seen (including lateral movement children)
            for t in state.active_threats:
                threats_seen.add(t.id)

        # ---- Episode metrics -----------------------------------------------
        final_state = env.state()
        # active_threats only contains non-contained threats (is_contained=False).
        # threats_contained = threats we saw minus those still active.
        threats_total = len(threats_seen)
        threats_still_active = len(final_state.active_threats)
        threats_contained_count = max(0, threats_total - threats_still_active)
        containment_rate = (
            threats_contained_count / threats_total if threats_total > 0 else 1.0
        )

        critical_assets = [
            a for a in final_state.assets.values() if a.criticality >= 0.7
        ]
        if critical_assets:
            critical_health = sum(a.health for a in critical_assets) / len(critical_assets)
        else:
            critical_health = 1.0

        avg_resource_left = (
            1.0 - (sum(resource_leftovers) / len(resource_leftovers))
            if resource_leftovers else 1.0
        )

        # Terminal reason detection
        if threats_still_active == 0 and threats_total > 0:
            terminal_reason = "all_contained"
        elif any(a.criticality >= 0.9 and a.health <= 0.0
                 for a in final_state.assets.values()):
            terminal_reason = "critical_asset_failure"

        episode_score = self._compute_episode_score(
            containment_rate=containment_rate,
            critical_health=critical_health,
            avg_resource_left=avg_resource_left,
            step_rewards=step_rewards,
        )

        return TaskResult(
            task_name=self.config.name,
            seed=seed,
            episode_score=round(episode_score, 4),
            passed=episode_score >= self.config.passing_score,
            steps_taken=len(step_rewards),
            threats_total=threats_total,
            threats_contained=threats_contained_count,
            containment_rate=round(containment_rate, 4),
            critical_health_end=round(critical_health, 4),
            avg_resource_left=round(avg_resource_left, 4),
            total_reward=round(sum(step_rewards), 4),
            step_rewards=step_rewards,
            reward_breakdowns=breakdowns,
            terminal_reason=terminal_reason,
        )

    def _compute_episode_score(
        self,
        containment_rate: float,
        critical_health: float,
        avg_resource_left: float,
        step_rewards: list[float],
    ) -> float:
        """
        Weighted episode score [0.0, 1.0].

        Weights:
          0.50 × containment_rate  — primary objective
          0.20 × critical_health   — asset preservation
          0.15 × avg_resource_left — efficiency
          0.15 × avg_step_reward   — quality of play each step
        """
        avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        score = (
            0.50 * containment_rate
            + 0.20 * critical_health
            + 0.15 * avg_resource_left
            + 0.15 * avg_reward
        )
        return max(0.0, min(1.0, score))
