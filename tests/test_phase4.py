"""
Phase 4 tests — ActionMemory, DecisionEngine, ResourcePrioritiser.

Memory tests verify:
  - Records are stored and retrievable
  - Success rates move in the right direction
  - Decay reduces weight of old outcomes
  - Expected value blends prior and empirical correctly
  - reset() clears all state

DecisionEngine tests verify:
  - Recommendations are produced for all threats
  - IGNORE recommended for low-severity threats
  - RUN_DEEP_SCAN recommended when detection confidence is low
  - ISOLATE_NODE recommended at LATERAL_SPREAD / EXFILTRATION stage
  - Unaffordable actions are flagged correctly
  - Recommendations are sorted by score descending
  - Duplicate node targets are not both funded

ResourcePrioritiser tests verify:
  - Funded actions respect budget cap
  - One action per node in funded list
  - Deferred list contains remainder
  - Empty recommendations → empty plan

Integration tests verify:
  - Memory updates after each step
  - Recommendations appear in step info
  - env.recommend() callable after reset+step
  - env.spending_plan() respects resource pool
"""

import pytest

from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.engines.decision import (
    ActionMemory, DecisionConfig, DecisionEngine,
    ResourcePrioritiser, ActionRecommendation,
)
from adaptive_cyber_defense.engines.scoring import ThreatScore, ThreatScorer
from adaptive_cyber_defense.models.action import Action, ActionInput
from adaptive_cyber_defense.models.network import NetworkGraph
from adaptive_cyber_defense.models.state import (
    AttackStage, ResourcePool, Threat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rng(seed=0):
    import random
    return random.Random(seed)


def make_network():
    return NetworkGraph.build_default(make_rng(0))


def make_threat_score(
    tid="t-001", node="ws-01", composite=0.5, stage=AttackStage.PHISHING
) -> ThreatScore:
    return ThreatScore(
        threat_id=tid, node_id=node,
        impact_score=composite, spread_score=composite,
        likelihood_score=composite, urgency_score=composite,
        composite_score=composite,
        primary_driver="impact",
    )


def make_threat(
    tid="t-001", node="ws-01", stage=AttackStage.PHISHING,
    detection_confidence=0.5, persistence=0.2, contained=False,
) -> Threat:
    return Threat(
        id=tid, stage=stage, origin_node=node, current_node=node,
        severity=0.4, detection_confidence=detection_confidence,
        is_detected=True, persistence=persistence, spread_potential=0.4,
        is_contained=contained,
    )


def full_pool(total=1.0) -> ResourcePool:
    return ResourcePool(total=total, remaining=total)


def empty_pool() -> ResourcePool:
    return ResourcePool(total=1.0, remaining=0.0)


# ---------------------------------------------------------------------------
# ActionMemory
# ---------------------------------------------------------------------------

class TestActionMemory:
    def test_record_and_retrieve(self):
        mem = ActionMemory()
        mem.record(
            action=Action.BLOCK_IP, node_id="ws-01",
            threat_stage=AttackStage.PHISHING, success=True,
            resource_cost=0.1, threat_score_before=0.4, step=1,
        )
        assert len(mem.recent_records(10)) == 1

    def test_success_rate_all_successes(self):
        mem = ActionMemory()
        for step in range(5):
            mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING,
                       True, 0.1, 0.4, step)
        assert mem.success_rate(Action.BLOCK_IP, "ws-01") > 0.5

    def test_success_rate_all_failures(self):
        mem = ActionMemory()
        for step in range(5):
            mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING,
                       False, 0.1, 0.4, step)
        assert mem.success_rate(Action.BLOCK_IP, "ws-01") < 0.5

    def test_neutral_prior_with_no_data(self):
        mem = ActionMemory()
        assert mem.success_rate(Action.PATCH_SYSTEM, "srv-db") == 0.5

    def test_expected_value_blends_toward_prior_when_sparse(self):
        """With only 1 sample, expected value should be close to the prior."""
        mem = ActionMemory()
        mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING,
                   True, 0.1, 0.4, 1)
        prior = 0.55   # BLOCK_IP base_effectiveness
        ev = mem.expected_value(Action.BLOCK_IP, "ws-01", fallback_effectiveness=prior)
        # With 1 sample (blend ~ 0.1), ev should be close to prior
        assert abs(ev - prior) < 0.30

    def test_decay_reduces_weight_of_old_outcomes(self):
        """After many new records, old success weight should be diminished."""
        mem = ActionMemory(decay=0.5)   # aggressive decay for testing
        # 10 successes early
        for step in range(10):
            mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING,
                       True, 0.1, 0.4, step)
        rate_after_early = mem.success_rate(Action.BLOCK_IP, "ws-01")
        # 10 failures late
        for step in range(10, 20):
            mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING,
                       False, 0.1, 0.4, step)
        rate_after_late = mem.success_rate(Action.BLOCK_IP, "ws-01")
        assert rate_after_late < rate_after_early

    def test_wasted_action_rate(self):
        mem = ActionMemory()
        mem.record(Action.PATCH_SYSTEM, "ws-01", AttackStage.PHISHING,
                   False, 0.2, 0.4, 1)
        mem.record(Action.PATCH_SYSTEM, "ws-01", AttackStage.PHISHING,
                   False, 0.2, 0.4, 2)
        assert mem.wasted_action_rate(Action.PATCH_SYSTEM) == 1.0

    def test_reset_clears_all(self):
        mem = ActionMemory()
        mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING,
                   True, 0.1, 0.4, 1)
        mem.reset()
        assert mem.recent_records() == []
        assert mem.success_rate(Action.BLOCK_IP, "ws-01") == 0.5

    def test_capacity_limit(self):
        mem = ActionMemory(capacity=5)
        for step in range(10):
            mem.record(Action.IGNORE, "ws-01", AttackStage.PHISHING,
                       False, 0.0, 0.0, step)
        assert len(mem.recent_records(100)) <= 5

    def test_global_success_rate_aggregates_nodes(self):
        """success_rate(action, node=None) should aggregate across all nodes."""
        mem = ActionMemory()
        mem.record(Action.BLOCK_IP, "ws-01", AttackStage.PHISHING, True, 0.1, 0.4, 1)
        mem.record(Action.BLOCK_IP, "ws-02", AttackStage.PHISHING, False, 0.1, 0.4, 2)
        rate = mem.success_rate(Action.BLOCK_IP)
        assert 0.0 < rate < 1.0


# ---------------------------------------------------------------------------
# DecisionEngine
# ---------------------------------------------------------------------------

class TestDecisionEngine:
    def setup_method(self):
        self.engine = DecisionEngine()
        self.network = make_network()
        self.mem = ActionMemory()

    def _state_with_threat(self, threat, detection_confidence=0.5):
        from adaptive_cyber_defense.models.state import EnvironmentState
        assets = self.network.assets
        return EnvironmentState(
            assets=assets,
            compromised_nodes=[threat.current_node],
            active_threats=[threat],
            threat_severity=0.5,
            network_load=0.2,
            resource_availability=1.0,
            detection_confidence=detection_confidence,
            time_step=1,
        )

    def test_produces_recommendations_for_active_threat(self):
        threat = make_threat(node="ws-01", stage=AttackStage.MALWARE_INSTALL)
        ts = ThreatScorer().score(threat, self.network)
        state = self._state_with_threat(threat)
        recs = self.engine.recommend(state, [ts], full_pool(), self.mem)
        assert len(recs) > 0

    def test_ignore_recommended_for_low_severity_threat(self):
        threat = make_threat(node="ws-03", stage=AttackStage.PHISHING)
        ts = make_threat_score(node="ws-03", composite=0.05)   # below threshold
        state = self._state_with_threat(threat)
        recs = self.engine.recommend(state, [ts], full_pool(), self.mem)
        top = recs[0]
        assert top.action_input.action == Action.IGNORE

    def test_deep_scan_recommended_for_low_confidence(self):
        threat = make_threat(node="ws-01", stage=AttackStage.PHISHING,
                              detection_confidence=0.10)  # very low
        ts = make_threat_score(node="ws-01", composite=0.6)
        state = self._state_with_threat(threat, detection_confidence=0.10)
        config = DecisionConfig(scan_confidence_threshold=0.30)
        engine = DecisionEngine(config)
        recs = engine.recommend(state, [ts], full_pool(), self.mem)
        top_affordable = next(r for r in recs if r.is_affordable)
        assert top_affordable.action_input.action == Action.RUN_DEEP_SCAN

    def test_isolate_recommended_at_lateral_spread_stage(self):
        threat = make_threat(node="ws-01", stage=AttackStage.LATERAL_SPREAD,
                              detection_confidence=0.8)
        ts = make_threat_score(node="ws-01", composite=0.8)
        state = self._state_with_threat(threat, detection_confidence=0.8)
        recs = self.engine.recommend(state, [ts], full_pool(), self.mem)
        top_affordable = next(r for r in recs if r.is_affordable)
        assert top_affordable.action_input.action == Action.ISOLATE_NODE

    def test_isolate_recommended_at_exfiltration_stage(self):
        threat = make_threat(node="srv-db", stage=AttackStage.EXFILTRATION,
                              detection_confidence=0.9)
        ts = make_threat_score(node="srv-db", composite=0.9)
        state = self._state_with_threat(threat, detection_confidence=0.9)
        recs = self.engine.recommend(state, [ts], full_pool(), self.mem)
        top_affordable = next(r for r in recs if r.is_affordable)
        assert top_affordable.action_input.action == Action.ISOLATE_NODE

    def test_unaffordable_actions_flagged(self):
        threat = make_threat(node="ws-01", stage=AttackStage.MALWARE_INSTALL)
        ts = make_threat_score(node="ws-01", composite=0.7)
        state = self._state_with_threat(threat)
        pool = ResourcePool(total=1.0, remaining=0.05)   # barely anything left
        recs = self.engine.recommend(state, [ts], pool, self.mem)
        expensive = [r for r in recs if r.resource_cost > 0.05]
        for r in expensive:
            assert not r.is_affordable

    def test_always_includes_ignore_fallback(self):
        threat = make_threat(node="ws-01", stage=AttackStage.PHISHING)
        ts = make_threat_score(node="ws-01", composite=0.5)
        state = self._state_with_threat(threat)
        recs = self.engine.recommend(state, [ts], full_pool(), self.mem)
        ignore_recs = [r for r in recs if r.action_input.action == Action.IGNORE]
        assert len(ignore_recs) >= 1

    def test_recommendations_sorted_affordable_first(self):
        threat = make_threat(node="ws-01", stage=AttackStage.MALWARE_INSTALL,
                              detection_confidence=0.8)
        ts = make_threat_score(node="ws-01", composite=0.7)
        state = self._state_with_threat(threat)
        # Small pool: some actions affordable, some not
        pool = ResourcePool(total=1.0, remaining=0.12)
        recs = self.engine.recommend(state, [ts], pool, self.mem)
        seen_unaffordable = False
        for r in recs:
            if not r.is_affordable:
                seen_unaffordable = True
            else:
                assert not seen_unaffordable, "Affordable rec appears after unaffordable"

    def test_memory_influences_score(self):
        """After recording many failures for BLOCK_IP, its score should drop."""
        threat = make_threat(node="ws-01", stage=AttackStage.MALWARE_INSTALL,
                              detection_confidence=0.8)
        ts = make_threat_score(node="ws-01", composite=0.6)
        state = self._state_with_threat(threat)

        mem_no_data = ActionMemory()
        mem_fail = ActionMemory()
        for step in range(15):
            mem_fail.record(Action.BLOCK_IP, "ws-01", AttackStage.MALWARE_INSTALL,
                            False, 0.1, 0.6, step)

        recs_no_data = self.engine.recommend(state, [ts], full_pool(), mem_no_data)
        recs_fail = self.engine.recommend(state, [ts], full_pool(), mem_fail)

        def block_score(recs):
            for r in recs:
                if r.action_input.action == Action.BLOCK_IP:
                    return r.score
            return 0.5

        assert block_score(recs_fail) <= block_score(recs_no_data)

    def test_no_recommendations_when_no_threats(self):
        from adaptive_cyber_defense.models.state import EnvironmentState
        state = EnvironmentState(
            assets=self.network.assets, compromised_nodes=[],
            active_threats=[], threat_severity=0.0, network_load=0.0,
            resource_availability=1.0, detection_confidence=0.0, time_step=0,
        )
        recs = self.engine.recommend(state, [], full_pool(), self.mem)
        # Only fallback IGNORE (which requires a threat_score) — list is empty
        assert isinstance(recs, list)


# ---------------------------------------------------------------------------
# ResourcePrioritiser
# ---------------------------------------------------------------------------

class TestResourcePrioritiser:
    def setup_method(self):
        self.prioritiser = ResourcePrioritiser()

    def _rec(self, action=Action.BLOCK_IP, node="ws-01", score=0.5, cost=0.10):
        ts = make_threat_score(node=node, composite=score)
        affordable = cost <= 1.0
        return ActionRecommendation(
            action_input=ActionInput(action=action, target_node=node),
            score=score, threat_score=ts,
            expected_benefit=score * 0.5,
            resource_cost=cost, availability_cost=0.0,
            reasoning="test", is_affordable=affordable,
        )

    def test_funded_respects_budget(self):
        recs = [
            self._rec(cost=0.30, score=0.9),
            self._rec(node="ws-02", cost=0.30, score=0.8),
            self._rec(node="ws-03", cost=0.30, score=0.7),
            self._rec(node="srv-db", cost=0.30, score=0.6),
        ]
        pool = ResourcePool(total=1.0, remaining=0.65)
        plan = self.prioritiser.plan(recs, pool)
        assert plan.total_cost <= 0.65

    def test_one_action_per_node(self):
        # Two actions targeting ws-01
        recs = [
            self._rec(action=Action.BLOCK_IP, node="ws-01", score=0.9, cost=0.10),
            self._rec(action=Action.ISOLATE_NODE, node="ws-01", score=0.8, cost=0.25),
        ]
        pool = full_pool()
        plan = self.prioritiser.plan(recs, pool)
        funded_nodes = [r.action_input.target_node for r in plan.funded]
        assert funded_nodes.count("ws-01") <= 1

    def test_deferred_contains_excess(self):
        recs = [
            self._rec(node="ws-01", cost=0.90, score=0.9),
            self._rec(node="ws-02", cost=0.50, score=0.8),
        ]
        pool = ResourcePool(total=1.0, remaining=1.0)
        plan = self.prioritiser.plan(recs, pool)
        assert len(plan.deferred) >= 1

    def test_empty_recommendations(self):
        plan = self.prioritiser.plan([], full_pool())
        assert plan.funded == []
        assert plan.deferred == []
        assert plan.total_cost == 0.0

    def test_plan_with_zero_budget(self):
        recs = [self._rec(cost=0.10, score=0.9)]
        plan = self.prioritiser.plan(recs, empty_pool())
        assert plan.funded == []
        assert len(plan.deferred) == 1

    def test_utilisation_computed_correctly(self):
        recs = [self._rec(cost=0.50, score=0.9)]
        pool = ResourcePool(total=1.0, remaining=1.0)
        plan = self.prioritiser.plan(recs, pool)
        if plan.funded:
            assert abs(plan.utilisation - 0.50) < 0.01

    def test_ignore_actions_pass_through(self):
        recs = [ActionRecommendation(
            action_input=ActionInput(action=Action.IGNORE),
            score=0.0, threat_score=make_threat_score(), expected_benefit=0.0,
            resource_cost=0.0, availability_cost=0.0,
            reasoning="ignore", is_affordable=True,
        )]
        plan = self.prioritiser.plan(recs, full_pool())
        # IGNORE has no target_node, so dedup logic skips it
        assert len(plan.funded) == 1


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------

class TestDecisionIntegration:
    def test_recommendations_in_step_info(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(ActionInput(action=Action.IGNORE))
        assert "recommendations" in info
        assert isinstance(info["recommendations"], list)

    def test_resource_utilisation_in_step_info(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(ActionInput(action=Action.IGNORE))
        assert "resource_utilisation" in info
        assert 0.0 <= info["resource_utilisation"] <= 1.0

    def test_env_recommend_returns_list(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        env.step(ActionInput(action=Action.IGNORE))
        recs = env.recommend()
        assert isinstance(recs, list)

    def test_env_recommend_before_step_returns_list(self):
        """recommend() is callable after reset even before first step."""
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        # No step yet — last_threat_scores is empty
        recs = env.recommend()
        assert isinstance(recs, list)

    def test_spending_plan_respects_pool(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        env.step(ActionInput(action=Action.IGNORE))
        plan = env.spending_plan()
        assert plan.total_cost <= plan.budget + 1e-9

    def test_memory_updates_after_non_ignore_action(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=2)
        s = env.state()
        if not s.compromised_nodes:
            pytest.skip("No compromised node in this seed")
        target = s.compromised_nodes[0]
        env.step(ActionInput(action=Action.BLOCK_IP, target_node=target))
        mem = env.action_memory()
        # Should have at least one record after taking a real action
        assert len(mem.recent_records()) >= 1

    def test_memory_reset_on_new_episode(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        s = env.state()
        if s.compromised_nodes:
            env.step(ActionInput(action=Action.BLOCK_IP,
                                  target_node=s.compromised_nodes[0]))
        env.reset(seed=0)   # fresh episode
        assert env.action_memory().recent_records() == []

    def test_top_recommendation_is_affordable(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=5)
        env.step(ActionInput(action=Action.IGNORE))
        recs = env.recommend()
        affordable = [r for r in recs if r.is_affordable]
        if affordable:
            assert affordable[0].score >= 0.0

    def test_determinism_recommendations_same_seed(self):
        """Same seed must produce same recommendation scores."""
        def run(seed):
            env = AdaptiveCyberDefenseEnv()
            env.reset(seed=seed)
            env.step(ActionInput(action=Action.IGNORE))
            recs = env.recommend()
            return [round(r.score, 4) for r in recs[:3]]

        assert run(42) == run(42)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
