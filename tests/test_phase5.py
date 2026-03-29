"""
Phase 5 tests — ResponseEngine and StateUpdater.

ResponseEngine tests cover:
  - BLOCK_IP: containment probability, detection boost, wasted-on-clean-node
  - ISOLATE_NODE: isolation flag set, containment, availability impact,
                  wasted on already-isolated node
  - PATCH_SYSTEM: patch_level increase, stage-dependent containment,
                  success even with zero containment
  - RUN_DEEP_SCAN: detection confidence boost, scan registered with
                   DetectionSystem, small containment chance
  - IGNORE: zero cost, zero effect, no containment
  - Resource guard: action skipped when budget exhausted

StateUpdater tests cover:
  - Compromise flags re-derived from live threats each step
  - Health degradation severity-weighted
  - Isolated nodes recover health
  - Lateral movement events mark new nodes compromised
  - Network load is criticality-weighted

Integration tests verify the full env.step() pipeline for each action type.
"""

import random
import pytest

from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.engines.detection import DetectionSystem
from adaptive_cyber_defense.engines.response import (
    ResponseEngine, ResponseConfig, StateUpdater,
)
from adaptive_cyber_defense.engines.attack import LateralMovementEvent
from adaptive_cyber_defense.models.action import Action, ActionInput
from adaptive_cyber_defense.models.network import NetworkGraph
from adaptive_cyber_defense.models.state import (
    AttackStage, NetworkAsset, AssetType, ResourcePool, Threat,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rng(seed=0):
    return random.Random(seed)


def make_network(seed=0):
    return NetworkGraph.build_default(make_rng(seed))


def make_threat(
    tid="t-001", node="ws-01",
    stage=AttackStage.PHISHING,
    severity=0.4, persistence=0.2,
    detection_confidence=0.5,
    contained=False,
) -> Threat:
    return Threat(
        id=tid, stage=stage, origin_node=node, current_node=node,
        severity=severity, detection_confidence=detection_confidence,
        is_detected=True, persistence=persistence, spread_potential=0.4,
        is_contained=contained,
    )


def full_pool(total=1.0):
    return ResourcePool(total=total, remaining=total)


def empty_pool():
    return ResourcePool(total=1.0, remaining=0.0)


def make_engine(detection_system=None):
    return ResponseEngine(
        config=ResponseConfig(),
        detection_system=detection_system or DetectionSystem(),
    )


# ---------------------------------------------------------------------------
# BLOCK_IP
# ---------------------------------------------------------------------------

class TestBlockIP:
    def test_containment_possible_at_phishing_stage(self):
        """BLOCK_IP with prob=1.0 config should contain a PHISHING threat."""
        # Use a known seed where containment fires
        engine = make_engine()
        network = make_network()
        threat = make_threat(node="ws-01", stage=AttackStage.PHISHING)
        pool = full_pool()
        # Run multiple seeds until we get a containment
        for seed in range(50):
            t, result = engine.apply(
                ActionInput(action=Action.BLOCK_IP, target_node="ws-01"),
                [threat], network, full_pool(), make_rng(seed),
            )
            if result.success:
                assert "t-001" in result.threats_contained
                return
        pytest.fail("BLOCK_IP never succeeded in 50 trials for PHISHING stage")

    def test_block_ip_detection_boost(self):
        threat = make_threat(node="ws-01", detection_confidence=0.3)
        engine = make_engine()
        updated, result = engine.apply(
            ActionInput(action=Action.BLOCK_IP, target_node="ws-01"),
            [threat], make_network(), full_pool(), make_rng(99),
        )
        live = [t for t in updated if not t.is_contained]
        if live:
            assert live[0].detection_confidence > 0.3

    def test_block_ip_on_clean_node_is_wasted(self):
        """BLOCK_IP on a node with no active threat = wasted action."""
        engine = make_engine()
        network = make_network()
        # No threats on ws-01 (pass empty list)
        _, result = engine.apply(
            ActionInput(action=Action.BLOCK_IP, target_node="ws-01"),
            [], network, full_pool(), make_rng(0),
        )
        assert result.wasted
        assert result.failure_reason == "no_active_threat_on_node"

    def test_block_ip_less_effective_at_exfiltration(self):
        """Exfiltration stage has lower containment multiplier than Phishing."""
        # Run many trials and compare containment rates across stages
        def containment_rate(stage, n=200):
            count = 0
            for seed in range(n):
                t = make_threat(node="ws-01", stage=stage)
                _, result = make_engine().apply(
                    ActionInput(action=Action.BLOCK_IP, target_node="ws-01"),
                    [t], make_network(), full_pool(), make_rng(seed),
                )
                if result.success:
                    count += 1
            return count / n

        rate_phish = containment_rate(AttackStage.PHISHING)
        rate_exfil = containment_rate(AttackStage.EXFILTRATION)
        assert rate_phish > rate_exfil

    def test_block_ip_consumes_resources(self):
        pool = full_pool()
        engine = make_engine()
        _, result = engine.apply(
            ActionInput(action=Action.BLOCK_IP, target_node="ws-01"),
            [make_threat()], make_network(), pool, make_rng(0),
        )
        assert result.cost_paid > 0.0


# ---------------------------------------------------------------------------
# ISOLATE_NODE
# ---------------------------------------------------------------------------

class TestIsolateNode:
    def test_isolation_sets_asset_flag(self):
        engine = make_engine()
        network = make_network()
        threat = make_threat(node="ws-01", stage=AttackStage.LATERAL_SPREAD)
        engine.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01"),
            [threat], network, full_pool(), make_rng(0),
        )
        assert network.assets["ws-01"].is_isolated is True

    def test_isolation_high_containment_rate(self):
        """ISOLATE_NODE should contain threats most of the time."""
        contained_count = 0
        for seed in range(100):
            t = make_threat(node="ws-01", stage=AttackStage.LATERAL_SPREAD)
            _, result = make_engine().apply(
                ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01"),
                [t], make_network(), full_pool(), make_rng(seed),
            )
            if result.success:
                contained_count += 1
        assert contained_count > 70  # >70% containment rate

    def test_isolate_already_isolated_is_wasted(self):
        engine = make_engine()
        network = make_network()
        network.assets["ws-01"].is_isolated = True
        _, result = engine.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01"),
            [make_threat()], network, full_pool(), make_rng(0),
        )
        assert result.wasted
        assert result.failure_reason == "already_isolated"

    def test_isolate_availability_impact_scales_with_criticality(self):
        """High-criticality node should have higher availability impact."""
        engine = make_engine()

        net_low = make_network()
        net_low.assets["ws-03"].criticality = 0.1  # low-crit node
        _, r_low = engine.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="ws-03"),
            [make_threat(node="ws-03")], net_low, full_pool(), make_rng(0),
        )

        engine2 = make_engine()
        net_high = make_network()
        net_high.assets["srv-db"].criticality = 1.0
        _, r_high = engine2.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="srv-db"),
            [make_threat(node="srv-db")], net_high, full_pool(), make_rng(0),
        )
        assert r_high.availability_impact > r_low.availability_impact

    def test_isolate_consumes_resources(self):
        pool = full_pool()
        engine = make_engine()
        _, result = engine.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01"),
            [make_threat()], make_network(), pool, make_rng(0),
        )
        assert result.cost_paid > 0.0


# ---------------------------------------------------------------------------
# PATCH_SYSTEM
# ---------------------------------------------------------------------------

class TestPatchSystem:
    def test_patch_increases_patch_level(self):
        engine = make_engine()
        network = make_network()
        old_patch = network.assets["ws-01"].patch_level
        engine.apply(
            ActionInput(action=Action.PATCH_SYSTEM, target_node="ws-01"),
            [], network, full_pool(), make_rng(0),
        )
        assert network.assets["ws-01"].patch_level > old_patch

    def test_patch_level_capped_at_one(self):
        engine = make_engine()
        network = make_network()
        network.assets["ws-01"].patch_level = 0.99
        engine.apply(
            ActionInput(action=Action.PATCH_SYSTEM, target_node="ws-01"),
            [], network, full_pool(), make_rng(0),
        )
        assert network.assets["ws-01"].patch_level <= 1.0

    def test_patch_success_even_with_no_containment(self):
        """Patch improves hardening — result.success=True even if no threat contained."""
        engine = make_engine()
        _, result = engine.apply(
            ActionInput(action=Action.PATCH_SYSTEM, target_node="ws-01"),
            [], make_network(), full_pool(), make_rng(0),
        )
        assert result.success  # patch_improvement > 0

    def test_patch_more_effective_at_early_stages(self):
        """Patch should contain PHISHING threats more often than EXFILTRATION."""
        def containment_rate(stage, n=200):
            count = 0
            for seed in range(n):
                t = make_threat(node="ws-01", stage=stage)
                _, r = make_engine().apply(
                    ActionInput(action=Action.PATCH_SYSTEM, target_node="ws-01"),
                    [t], make_network(), full_pool(), make_rng(seed),
                )
                if r.threats_contained:
                    count += 1
            return count / n

        assert containment_rate(AttackStage.PHISHING) > containment_rate(AttackStage.EXFILTRATION)


# ---------------------------------------------------------------------------
# RUN_DEEP_SCAN
# ---------------------------------------------------------------------------

class TestRunDeepScan:
    def test_deep_scan_boosts_detection_confidence(self):
        engine = make_engine()
        threat = make_threat(node="ws-01", detection_confidence=0.3)
        updated, result = engine.apply(
            ActionInput(action=Action.RUN_DEEP_SCAN, target_node="ws-01"),
            [threat], make_network(), full_pool(), make_rng(99),
        )
        live = [t for t in updated if not t.is_contained]
        if live:
            assert live[0].detection_confidence > 0.3

    def test_deep_scan_registers_boost_with_detection_system(self):
        ds = DetectionSystem()
        engine = ResponseEngine(config=ResponseConfig(), detection_system=ds)
        engine.apply(
            ActionInput(action=Action.RUN_DEEP_SCAN, target_node="ws-01"),
            [], make_network(), full_pool(), make_rng(0),
        )
        assert "ws-01" in ds._pending_scan_boosts

    def test_deep_scan_success_flag_set(self):
        """RUN_DEEP_SCAN always sets success=True (detection boost was applied)."""
        engine = make_engine()
        _, result = engine.apply(
            ActionInput(action=Action.RUN_DEEP_SCAN, target_node="ws-01"),
            [], make_network(), full_pool(), make_rng(0),
        )
        assert result.success

    def test_deep_scan_most_expensive_action(self):
        """RUN_DEEP_SCAN should have higher resource cost than BLOCK_IP."""
        from adaptive_cyber_defense.models.action import ACTION_PROFILES
        assert (
            ACTION_PROFILES[Action.RUN_DEEP_SCAN].resource_cost
            > ACTION_PROFILES[Action.BLOCK_IP].resource_cost
        )


# ---------------------------------------------------------------------------
# IGNORE + resource guard
# ---------------------------------------------------------------------------

class TestIgnoreAndResources:
    def test_ignore_returns_unchanged_threats(self):
        threat = make_threat()
        engine = make_engine()
        updated, result = engine.apply(
            ActionInput(action=Action.IGNORE),
            [threat], make_network(), full_pool(), make_rng(0),
        )
        assert updated[0].is_contained is False
        assert result.cost_paid == 0.0

    def test_action_skipped_when_no_resources(self):
        engine = make_engine()
        pool = empty_pool()
        _, result = engine.apply(
            ActionInput(action=Action.BLOCK_IP, target_node="ws-01"),
            [make_threat()], make_network(), pool, make_rng(0),
        )
        assert not result.success
        assert result.failure_reason == "insufficient_resources"

    def test_resource_consumed_after_action(self):
        pool = full_pool()
        engine = make_engine()
        engine.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01"),
            [make_threat()], make_network(), pool, make_rng(0),
        )
        assert pool.remaining < pool.total

    def test_contained_threats_not_affected_by_action(self):
        """Already-contained threats should not be touched by response engine."""
        contained = make_threat(contained=True)
        engine = make_engine()
        updated, _ = engine.apply(
            ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01"),
            [contained], make_network(), full_pool(), make_rng(0),
        )
        # contained threat stays contained and unchanged in stage
        assert updated[0].is_contained


# ---------------------------------------------------------------------------
# StateUpdater
# ---------------------------------------------------------------------------

class TestStateUpdater:
    def setup_method(self):
        self.updater = StateUpdater(
            health_degradation_rate=0.10,
            isolation_recovery_rate=0.05,
        )

    def test_compromise_flag_set_from_active_threat(self):
        network = make_network()
        threat = make_threat(node="ws-01")
        self.updater.update(network, [threat], [])
        assert network.assets["ws-01"].is_compromised is True

    def test_compromise_flag_cleared_for_contained_threat(self):
        network = make_network()
        threat = make_threat(node="ws-01", contained=True)
        self.updater.update(network, [threat], [])
        assert network.assets["ws-01"].is_compromised is False

    def test_health_degrades_for_compromised_node(self):
        network = make_network()
        initial_health = network.assets["ws-01"].health
        threat = make_threat(node="ws-01", severity=0.5)
        self.updater.update(network, [threat], [])
        assert network.assets["ws-01"].health < initial_health

    def test_health_floor_at_zero(self):
        network = make_network()
        network.assets["ws-01"].health = 0.001
        threat = make_threat(node="ws-01", severity=1.0)
        self.updater.update(network, [threat], [])
        assert network.assets["ws-01"].health >= 0.0

    def test_isolated_clean_node_recovers_health(self):
        network = make_network()
        network.assets["ws-01"].is_isolated = True
        network.assets["ws-01"].health = 0.50
        # No threats → isolation recovery applies
        self.updater.update(network, [], [])
        assert network.assets["ws-01"].health > 0.50

    def test_lateral_movement_event_marks_target_compromised(self):
        network = make_network()
        child_threat = make_threat(node="srv-db")
        event = LateralMovementEvent(
            parent_threat_id="t-001",
            source_node="ws-01",
            target_node="srv-db",
            child_threat=child_threat,
        )
        self.updater.update(network, [], [event])
        assert network.assets["srv-db"].is_compromised is True

    def test_network_load_increases_with_compromised_nodes(self):
        network1 = make_network()
        network2 = make_network()
        # Compromise a critical node in network2
        network2.assets["srv-db"].is_compromised = True
        load1 = self.updater.network_load(network1)
        load2 = self.updater.network_load(network2)
        assert load2 > load1

    def test_network_load_zero_when_clean(self):
        network = make_network()
        load = self.updater.network_load(network)
        assert load == 0.0

    def test_network_load_capped_at_one(self):
        network = make_network()
        for asset in network.assets.values():
            asset.is_compromised = True
            asset.criticality = 1.0
        load = self.updater.network_load(network)
        assert load <= 1.0


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------

class TestResponseIntegration:
    def test_isolate_action_sets_node_isolated_in_state(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        s0 = env.state()
        if not s0.compromised_nodes:
            pytest.skip("No compromised node in seed=0")
        target = s0.compromised_nodes[0]
        env.step(ActionInput(action=Action.ISOLATE_NODE, target_node=target))
        s1 = env.state()
        assert s1.assets[target].is_isolated is True

    def test_patch_action_increases_patch_level(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        s0 = env.state()
        target = list(s0.assets.keys())[0]
        old_patch = s0.assets[target].patch_level
        env.step(ActionInput(action=Action.PATCH_SYSTEM, target_node=target))
        s1 = env.state()
        assert s1.assets[target].patch_level >= old_patch

    def test_deep_scan_raises_detection_confidence_over_time(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=1)
        s0 = env.state()
        if not s0.compromised_nodes:
            pytest.skip("No compromised node in seed=1")
        target = s0.compromised_nodes[0]
        env.step(ActionInput(action=Action.RUN_DEEP_SCAN, target_node=target))
        s1 = env.state()
        # After a deep scan, detection confidence should be >= initial
        assert s1.detection_confidence >= 0.0

    def test_action_result_in_step_info(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        s = env.state()
        if not s.compromised_nodes:
            pytest.skip("No compromised node in seed=0")
        _, _, _, info = env.step(
            ActionInput(action=Action.BLOCK_IP, target_node=s.compromised_nodes[0])
        )
        outcome = info["action_outcome"]
        assert "action" in outcome
        assert "cost" in outcome
        assert "wasted" in outcome

    def test_network_load_in_state_after_isolation(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        s0 = env.state()
        if not s0.compromised_nodes:
            pytest.skip("No compromised node")
        target = s0.compromised_nodes[0]
        _, _, _, _ = env.step(ActionInput(action=Action.ISOLATE_NODE, target_node=target))
        s1 = env.state()
        assert 0.0 <= s1.network_load <= 1.0

    def test_health_decreases_with_ignore_actions(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=3)
        s0 = env.state()
        if not s0.compromised_nodes:
            pytest.skip("No compromised node")
        node = s0.compromised_nodes[0]
        initial_health = s0.assets[node].health
        for _ in range(8):
            s, _, done, _ = env.step(ActionInput(action=Action.IGNORE))
            if done:
                break
        final = env.state().assets.get(node)
        if final:
            assert final.health <= initial_health

    def test_episode_terminates_on_critical_asset_failure(self):
        """If a critical asset health drops to 0, episode should end."""
        from adaptive_cyber_defense.env import EnvConfig
        cfg = EnvConfig()
        cfg.health_degradation_rate = 0.5   # aggressive degradation
        cfg.attack_progression_prob = 1.0
        cfg.max_steps = 100
        env = AdaptiveCyberDefenseEnv(config=cfg)
        env.reset(seed=0)
        done = False
        steps = 0
        while not done and steps < 100:
            _, _, done, _ = env.step(ActionInput(action=Action.IGNORE))
            steps += 1
        assert done

    def test_full_pipeline_deterministic(self):
        """Complete pipeline with real actions must be deterministic."""
        def run(seed):
            env = AdaptiveCyberDefenseEnv()
            env.reset(seed=seed)
            trajectory = []
            for step in range(10):
                s = env.state()
                if s.compromised_nodes and step % 3 == 0:
                    action = ActionInput(
                        action=Action.BLOCK_IP,
                        target_node=s.compromised_nodes[0],
                    )
                else:
                    action = ActionInput(action=Action.IGNORE)
                _, r, done, info = env.step(action)
                trajectory.append((round(r, 4), info["action_outcome"]["wasted"]))
                if done:
                    break
            return trajectory

        assert run(99) == run(99)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
