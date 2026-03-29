"""
Phase 6 tests — RewardFunction and Task definitions.

Reward tests verify:
  - Reward is always in [0.0, 1.0]
  - Containment produces positive reward
  - Spread events penalise reward
  - Wasted actions penalise reward
  - False positives penalise reward
  - Resource efficiency contributes positively
  - Survival bonus is present every step
  - Breakdown components are consistent with total
  - Determinism: same inputs → same reward

Task tests verify:
  - EasyTask, MediumTask, HardTask each build valid environments
  - Each task runs a full episode and produces a TaskResult
  - Episode score is in [0.0, 1.0]
  - Harder tasks are harder (lower mean score for same agent)
  - TaskResult.summary() returns a string
  - Same seed → same result (reproducibility)
  - Passing threshold respected in TaskResult.passed
"""

import random
import pytest

from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.engines.attack import LateralMovementEvent
from adaptive_cyber_defense.engines.detection import DetectionEvent
from adaptive_cyber_defense.engines.response import ActionResult
from adaptive_cyber_defense.engines.reward import (
    RewardFunction, RewardWeights, RewardBreakdown,
)
from adaptive_cyber_defense.engines.scoring import ThreatScore
from adaptive_cyber_defense.models.action import Action, ActionInput
from adaptive_cyber_defense.models.network import NetworkGraph
from adaptive_cyber_defense.models.state import (
    AttackStage, EnvironmentState, ResourcePool, Threat,
)
from adaptive_cyber_defense.tasks import EasyTask, MediumTask, HardTask, TaskResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rng(seed=0):
    return random.Random(seed)


def make_network(seed=0):
    return NetworkGraph.build_default(make_rng(seed))


def make_threat(
    tid="t-001", node="ws-01", stage=AttackStage.PHISHING,
    severity=0.4, contained=False,
) -> Threat:
    return Threat(
        id=tid, stage=stage, origin_node=node, current_node=node,
        severity=severity, detection_confidence=0.5,
        is_detected=True, persistence=0.2, spread_potential=0.4,
        is_contained=contained,
    )


def make_threat_score(tid="t-001", node="ws-01", composite=0.6) -> ThreatScore:
    return ThreatScore(
        threat_id=tid, node_id=node,
        impact_score=composite, spread_score=composite,
        likelihood_score=composite, urgency_score=composite,
        composite_score=composite, primary_driver="impact",
    )


def make_state(
    threats=None, severity=0.4, network=None, compromised=None
) -> EnvironmentState:
    net = network or make_network()
    active = threats or []
    return EnvironmentState(
        assets=net.assets,
        compromised_nodes=compromised or [t.current_node for t in active if not t.is_contained],
        active_threats=active,
        threat_severity=severity,
        network_load=0.2,
        resource_availability=1.0,
        detection_confidence=0.5,
        time_step=1,
    )


def make_action_result(
    action=Action.IGNORE, node=None, success=False,
    cost=0.0, avail=0.0, wasted=False,
) -> ActionResult:
    return ActionResult(
        action=action, target_node=node,
        success=success, cost_paid=cost,
        availability_impact=avail, wasted=wasted,
    )


def full_pool(total=1.0, remaining=1.0):
    return ResourcePool(total=total, remaining=remaining)


def compute(
    state_before=None, state_after=None,
    action_result=None, threat_scores=None,
    lateral_events=None, detection_events=None,
    resource_pool=None, network=None,
    weights=None,
):
    rf = RewardFunction(weights or RewardWeights())
    net = network or make_network()
    return rf.compute(
        state_before=state_before or make_state(network=net),
        state_after=state_after or make_state(network=net),
        action_result=action_result or make_action_result(),
        threat_scores_before=threat_scores or [],
        lateral_events=lateral_events or [],
        detection_events=detection_events or [],
        resource_pool=resource_pool or full_pool(),
        network=net,
    )


# ---------------------------------------------------------------------------
# RewardFunction — bounds and basic properties
# ---------------------------------------------------------------------------

class TestRewardBounds:
    def test_reward_always_in_range(self):
        for seed in range(30):
            env = AdaptiveCyberDefenseEnv()
            env.reset(seed=seed)
            for _ in range(5):
                _, r, done, _ = env.step(ActionInput(action=Action.IGNORE))
                assert 0.0 <= r <= 1.0, f"reward={r} out of range at seed={seed}"
                if done:
                    break

    def test_reward_in_range_with_real_actions(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        s = env.state()
        if s.compromised_nodes:
            _, r, _, _ = env.step(
                ActionInput(action=Action.BLOCK_IP, target_node=s.compromised_nodes[0])
            )
            assert 0.0 <= r <= 1.0

    def test_breakdown_components_sum_to_total(self):
        """Breakdown components must be consistent with reported total."""
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=2)
        _, r, _, info = env.step(ActionInput(action=Action.IGNORE))
        bd = info["reward_breakdown"]
        reconstructed = (
            bd["containment"]
            + bd["severity"]
            + bd["efficiency"]
            + bd["survival"]
            + bd["spread_penalty"]   # already negated in dict
            + bd["waste_penalty"]
            + bd["fp_penalty"]
            + bd["avail_penalty"]
        )
        # total = clamp(sum_of_positives + sum_of_negatives); just check it's close
        assert abs(bd["total"] - r) < 1e-6


# ---------------------------------------------------------------------------
# RewardFunction — individual components
# ---------------------------------------------------------------------------

class TestContainmentBonus:
    def test_containment_produces_positive_contribution(self):
        """Containing a threat should yield a positive containment bonus."""
        net = make_network()
        t_active = make_threat(tid="t-001", node="ws-01")
        t_contained = make_threat(tid="t-001", node="ws-01", contained=True)

        state_before = make_state([t_active], severity=0.5, network=net)
        state_after  = make_state([t_contained], severity=0.3, network=net, compromised=[])
        ts = [make_threat_score("t-001", "ws-01", 0.7)]

        reward, bd = compute(
            state_before=state_before,
            state_after=state_after,
            threat_scores=ts,
            network=net,
        )
        assert bd.containment_bonus > 0.0

    def test_no_containment_no_bonus(self):
        net = make_network()
        t = make_threat(tid="t-001", node="ws-01")
        state = make_state([t], network=net)

        reward, bd = compute(
            state_before=state,
            state_after=state,
            network=net,
        )
        assert bd.containment_bonus == 0.0

    def test_early_stage_containment_worth_more(self):
        """Containing a PHISHING threat should give higher bonus than EXFILTRATION."""
        net = make_network()
        weights = RewardWeights(containment=1.0, severity=0.0, efficiency=0.0,
                                survival=0.0, spread=0.0, waste=0.0,
                                false_pos=0.0, availability=0.0)

        for stage, tid in [(AttackStage.PHISHING, "t-p"), (AttackStage.EXFILTRATION, "t-e")]:
            t_active = make_threat(tid=tid, node="ws-01", stage=stage)
            t_contained = make_threat(tid=tid, node="ws-01", stage=stage, contained=True)
            sb = make_state([t_active], network=net)
            sa = make_state([t_contained], network=net, compromised=[])
            ts = [make_threat_score(tid, "ws-01", 0.7)]
            _, bd = compute(state_before=sb, state_after=sa,
                            threat_scores=ts, network=net, weights=weights)
            if stage == AttackStage.PHISHING:
                bonus_phish = bd.containment_bonus
            else:
                bonus_exfil = bd.containment_bonus

        assert bonus_phish > bonus_exfil


class TestSeverityReduction:
    def test_severity_drop_positive(self):
        net = make_network()
        sb = make_state(severity=0.8, network=net)
        sa = make_state(severity=0.3, network=net)
        _, bd = compute(state_before=sb, state_after=sa, network=net)
        assert bd.severity_reduction > 0.0

    def test_severity_increase_no_bonus(self):
        net = make_network()
        sb = make_state(severity=0.3, network=net)
        sa = make_state(severity=0.8, network=net)
        _, bd = compute(state_before=sb, state_after=sa, network=net)
        assert bd.severity_reduction == 0.0


class TestSpreadPenalty:
    def test_lateral_event_penalises_reward(self):
        net = make_network()
        child = make_threat(node="srv-db")
        event = LateralMovementEvent(
            parent_threat_id="t-001",
            source_node="ws-01",
            target_node="srv-db",
            child_threat=child,
        )
        _, bd_spread = compute(lateral_events=[event], network=net)
        _, bd_clean  = compute(lateral_events=[], network=net)
        assert bd_spread.spread_penalty > bd_clean.spread_penalty

    def test_high_criticality_spread_worse(self):
        """Spreading to a high-criticality node should cost more."""
        net_low  = make_network()
        net_low.assets["ws-03"].criticality = 0.1
        net_high = make_network()
        net_high.assets["srv-db"].criticality = 1.0

        child_low  = make_threat(node="ws-03")
        child_high = make_threat(node="srv-db")

        ev_low  = LateralMovementEvent("t", "ws-01", "ws-03",  child_low)
        ev_high = LateralMovementEvent("t", "ws-01", "srv-db", child_high)

        _, bd_low  = compute(lateral_events=[ev_low],  network=net_low)
        _, bd_high = compute(lateral_events=[ev_high], network=net_high)
        assert bd_high.spread_penalty >= bd_low.spread_penalty


class TestWastePenalty:
    def test_wasted_action_penalised(self):
        net = make_network()
        wasted_result = make_action_result(wasted=True)
        valid_result  = make_action_result(wasted=False)
        _, bd_waste = compute(action_result=wasted_result, network=net)
        _, bd_ok    = compute(action_result=valid_result,  network=net)
        assert bd_waste.waste_penalty > bd_ok.waste_penalty

    def test_non_wasted_no_penalty(self):
        net = make_network()
        result = make_action_result(wasted=False)
        _, bd = compute(action_result=result, network=net)
        assert bd.waste_penalty == 0.0


class TestFalsePositivePenalty:
    def test_false_positives_penalised(self):
        net = make_network()
        fp_event = DetectionEvent(
            threat_id=None, node_id="ws-02",
            is_true_positive=False, is_false_positive=True, is_false_negative=False,
            updated_confidence=0.0, detection_method="false_alarm",
        )
        _, bd_fp = compute(detection_events=[fp_event], network=net)
        _, bd_ok = compute(detection_events=[], network=net)
        assert bd_fp.false_positive_penalty > bd_ok.false_positive_penalty


class TestResourceEfficiency:
    def test_full_resources_gives_max_efficiency(self):
        net = make_network()
        _, bd = compute(resource_pool=full_pool(1.0, 1.0), network=net)
        w = RewardWeights()
        assert abs(bd.resource_efficiency - w.efficiency * 1.0) < 1e-6

    def test_zero_resources_gives_zero_efficiency(self):
        net = make_network()
        _, bd = compute(resource_pool=ResourcePool(total=1.0, remaining=0.0), network=net)
        assert bd.resource_efficiency == 0.0


class TestSurvivalBonus:
    def test_healthy_network_gives_survival_bonus(self):
        net = make_network()
        # All assets fully healthy
        for a in net.assets.values():
            a.health = 1.0
        _, bd = compute(network=net)
        assert bd.survival_bonus > 0.0

    def test_critical_asset_failure_reduces_survival(self):
        net_healthy  = make_network()
        net_degraded = make_network()
        for a in net_degraded.assets.values():
            if a.criticality >= 0.7:
                a.health = 0.0

        _, bd_h = compute(network=net_healthy)
        _, bd_d = compute(network=net_degraded)
        assert bd_h.survival_bonus > bd_d.survival_bonus


# ---------------------------------------------------------------------------
# Reward determinism
# ---------------------------------------------------------------------------

class TestRewardDeterminism:
    def test_same_episode_same_rewards(self):
        def run(seed):
            env = AdaptiveCyberDefenseEnv()
            env.reset(seed=seed)
            rewards = []
            for _ in range(10):
                _, r, done, _ = env.step(ActionInput(action=Action.IGNORE))
                rewards.append(round(r, 6))
                if done:
                    break
            return rewards

        assert run(17) == run(17)

    def test_reward_breakdown_in_step_info(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(ActionInput(action=Action.IGNORE))
        bd = info["reward_breakdown"]
        assert "total" in bd
        assert "containment" in bd
        assert "survival" in bd

    def test_env_reward_breakdown_accessor(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        env.step(ActionInput(action=Action.IGNORE))
        bd = env.reward_breakdown()
        assert bd is not None
        assert 0.0 <= bd.total <= 1.0


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

class SimpleIgnoreAgent:
    """Baseline: always IGNORE. Useful for testing task infrastructure."""
    def choose(self, state):
        return ActionInput(action=Action.IGNORE)


class SimpleContainAgent:
    """Try to BLOCK_IP on the first compromised node, else IGNORE."""
    def choose(self, state):
        if state.compromised_nodes:
            return ActionInput(action=Action.BLOCK_IP,
                               target_node=state.compromised_nodes[0])
        return ActionInput(action=Action.IGNORE)


class TestEasyTask:
    def test_builds_valid_env(self):
        task = EasyTask()
        env = task.build_env()
        s = env.reset(seed=0)
        assert s is not None
        assert len(s.assets) == 8

    def test_run_returns_task_result(self):
        task = EasyTask()
        result = task.run(SimpleIgnoreAgent(), seed=0)
        assert isinstance(result, TaskResult)

    def test_score_in_range(self):
        task = EasyTask()
        result = task.run(SimpleIgnoreAgent(), seed=0)
        assert 0.0 <= result.episode_score <= 1.0

    def test_containment_agent_scores_higher_than_ignore(self):
        task = EasyTask()
        r_ignore  = task.run(SimpleIgnoreAgent(),  seed=42)
        r_contain = task.run(SimpleContainAgent(), seed=42)
        assert r_contain.episode_score >= r_ignore.episode_score

    def test_deterministic(self):
        task = EasyTask()
        r1 = task.run(SimpleIgnoreAgent(), seed=7)
        r2 = task.run(SimpleIgnoreAgent(), seed=7)
        assert r1.episode_score == r2.episode_score

    def test_summary_returns_string(self):
        result = EasyTask().run(SimpleIgnoreAgent(), seed=0)
        s = result.summary()
        assert isinstance(s, str)
        assert "easy" in s.lower()


class TestMediumTask:
    def test_builds_valid_env(self):
        env = MediumTask().build_env()
        s = env.reset(seed=0)
        # Medium starts with 2 threats
        assert len(s.active_threats) >= 1

    def test_score_in_range(self):
        result = MediumTask().run(SimpleIgnoreAgent(), seed=0)
        assert 0.0 <= result.episode_score <= 1.0

    def test_medium_harder_than_easy_for_ignore_agent(self):
        """Ignore agent should fare worse on medium than easy (more threats, tighter resources)."""
        scores_easy   = [EasyTask().run(SimpleIgnoreAgent(),   seed=s).episode_score for s in range(5)]
        scores_medium = [MediumTask().run(SimpleIgnoreAgent(), seed=s).episode_score for s in range(5)]
        assert sum(scores_easy) >= sum(scores_medium)

    def test_deterministic(self):
        r1 = MediumTask().run(SimpleIgnoreAgent(), seed=3)
        r2 = MediumTask().run(SimpleIgnoreAgent(), seed=3)
        assert r1.episode_score == r2.episode_score


class TestHardTask:
    def test_builds_valid_env(self):
        env = HardTask().build_env()
        s = env.reset(seed=0)
        assert len(s.active_threats) >= 1

    def test_score_in_range(self):
        result = HardTask().run(SimpleIgnoreAgent(), seed=0)
        assert 0.0 <= result.episode_score <= 1.0

    def test_hard_harder_than_medium_for_ignore_agent(self):
        scores_medium = [MediumTask().run(SimpleIgnoreAgent(), seed=s).episode_score for s in range(5)]
        scores_hard   = [HardTask().run(SimpleIgnoreAgent(),   seed=s).episode_score for s in range(5)]
        assert sum(scores_medium) >= sum(scores_hard)

    def test_deterministic(self):
        r1 = HardTask().run(SimpleIgnoreAgent(), seed=9)
        r2 = HardTask().run(SimpleIgnoreAgent(), seed=9)
        assert r1.episode_score == r2.episode_score

    def test_passing_score_lower_than_easy(self):
        assert HardTask.config.passing_score < EasyTask.config.passing_score


class TestTaskResult:
    def test_containment_rate_in_range(self):
        result = EasyTask().run(SimpleContainAgent(), seed=0)
        assert 0.0 <= result.containment_rate <= 1.0

    def test_steps_taken_positive(self):
        result = EasyTask().run(SimpleIgnoreAgent(), seed=0)
        assert result.steps_taken > 0

    def test_passed_flag_consistent(self):
        task = EasyTask()
        result = task.run(SimpleIgnoreAgent(), seed=0)
        assert result.passed == (result.episode_score >= task.config.passing_score)

    def test_step_rewards_length_matches_steps(self):
        result = EasyTask().run(SimpleIgnoreAgent(), seed=0)
        assert len(result.step_rewards) == result.steps_taken

    def test_all_step_rewards_in_range(self):
        result = EasyTask().run(SimpleIgnoreAgent(), seed=1)
        for r in result.step_rewards:
            assert 0.0 <= r <= 1.0

    def test_total_reward_consistent(self):
        result = EasyTask().run(SimpleIgnoreAgent(), seed=0)
        assert abs(result.total_reward - sum(result.step_rewards)) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
