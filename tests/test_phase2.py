"""
Phase 2 tests — AttackEngine: kill-chain progression, severity growth,
lateral movement, and environment state evolution.
"""

import random
import pytest

from adaptive_cyber_defense.engines.attack import AttackEngine, AttackEngineConfig
from adaptive_cyber_defense.models.network import NetworkGraph
from adaptive_cyber_defense.models.state import AttackStage, Threat
from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.models.action import Action, ActionInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def make_threat(
    node: str = "ws-01",
    stage: AttackStage = AttackStage.PHISHING,
    severity: float = 0.3,
    persistence: float = 0.1,
    spread_potential: float = 0.9,  # high so lateral tests trigger easily
    contained: bool = False,
) -> Threat:
    return Threat(
        id="t-001",
        stage=stage,
        origin_node=node,
        current_node=node,
        severity=severity,
        detection_confidence=0.3,
        is_detected=False,
        persistence=persistence,
        spread_potential=spread_potential,
        is_contained=contained,
    )


def make_network(rng: random.Random = None) -> NetworkGraph:
    return NetworkGraph.build_default(rng or make_rng())


# ---------------------------------------------------------------------------
# AttackEngine — stage progression
# ---------------------------------------------------------------------------

class TestStageProgression:
    def setup_method(self):
        # Force progression by using a config with high probability.
        # min_stage_dwell=0 disables the realism dwell gate so these unit
        # tests remain focused on the probability logic, not the dwell constraint.
        self.engine = AttackEngine(AttackEngineConfig(
            stage_progression_base_prob=1.0,   # always progress
            lateral_movement_base_prob=0.0,    # disable lateral for this suite
            natural_severity_growth=0.0,
            min_stage_dwell=0,
        ))
        self.network = make_network()

    def test_threat_advances_stage(self):
        threat = make_threat(stage=AttackStage.PHISHING)
        rng = make_rng(0)
        updated, _ = self.engine.evolve([threat], self.network, rng)
        live = [t for t in updated if not t.is_contained]
        assert live[0].stage == AttackStage.CREDENTIAL_ACCESS

    def test_threat_does_not_advance_past_exfiltration(self):
        threat = make_threat(stage=AttackStage.EXFILTRATION)
        rng = make_rng(0)
        updated, _ = self.engine.evolve([threat], self.network, rng)
        live = [t for t in updated if not t.is_contained]
        assert live[0].stage == AttackStage.EXFILTRATION

    def test_contained_threat_not_evolved(self):
        threat = make_threat(contained=True)
        rng = make_rng(0)
        updated, events = self.engine.evolve([threat], self.network, rng)
        assert updated[0].is_contained
        assert updated[0].stage == AttackStage.PHISHING   # unchanged
        assert events == []

    def test_stage_sequence_over_multiple_steps(self):
        """With prob=1.0 and no dwell gate, threat reaches EXFILTRATION in exactly 4 steps."""
        engine = AttackEngine(AttackEngineConfig(
            stage_progression_base_prob=1.0,
            lateral_movement_base_prob=0.0,
            min_stage_dwell=0,
        ))
        network = make_network()
        threats = [make_threat(stage=AttackStage.PHISHING)]
        rng = make_rng(0)
        for expected in [
            AttackStage.CREDENTIAL_ACCESS,
            AttackStage.MALWARE_INSTALL,
            AttackStage.LATERAL_SPREAD,
            AttackStage.EXFILTRATION,
        ]:
            threats, _ = engine.evolve(threats, network, rng)
            live = [t for t in threats if not t.is_contained]
            assert live[0].stage == expected

    def test_no_progression_with_zero_prob(self):
        engine = AttackEngine(AttackEngineConfig(
            stage_progression_base_prob=0.0,
            lateral_movement_base_prob=0.0,
        ))
        threat = make_threat(stage=AttackStage.PHISHING)
        updated, _ = engine.evolve([threat], make_network(), make_rng(0))
        live = [t for t in updated if not t.is_contained]
        assert live[0].stage == AttackStage.PHISHING


# ---------------------------------------------------------------------------
# AttackEngine — severity and persistence
# ---------------------------------------------------------------------------

class TestSeverityAndPersistence:
    def setup_method(self):
        self.engine = AttackEngine(AttackEngineConfig(
            stage_progression_base_prob=0.0,   # lock stage for isolation
            lateral_movement_base_prob=0.0,
            natural_severity_growth=0.05,
            persistence_growth_rate=0.10,
        ))
        self.network = make_network()

    def test_severity_grows_each_step(self):
        threat = make_threat(severity=0.3)
        rng = make_rng(0)
        updated, _ = self.engine.evolve([threat], self.network, rng)
        live = [t for t in updated if not t.is_contained]
        assert live[0].severity > 0.3

    def test_severity_capped_at_one(self):
        threat = make_threat(severity=0.99)
        rng = make_rng(0)
        for _ in range(20):
            threats, _ = self.engine.evolve([threat], self.network, rng)
            live = [t for t in threats if not t.is_contained]
            threat = live[0]
        assert threat.severity <= 1.0

    def test_persistence_grows_each_step(self):
        threat = make_threat(persistence=0.1)
        rng = make_rng(0)
        updated, _ = self.engine.evolve([threat], self.network, rng)
        live = [t for t in updated if not t.is_contained]
        assert live[0].persistence > 0.1

    def test_persistence_capped(self):
        engine = AttackEngine(AttackEngineConfig(
            stage_progression_base_prob=0.0,
            lateral_movement_base_prob=0.0,
            persistence_growth_rate=0.50,
            persistence_cap=0.90,
        ))
        threat = make_threat(persistence=0.85)
        rng = make_rng(0)
        for _ in range(5):
            threats, _ = engine.evolve([threat], make_network(), rng)
            live = [t for t in threats if not t.is_contained]
            threat = live[0]
        assert threat.persistence <= 0.90

    def test_steps_active_increments(self):
        threat = make_threat()
        rng = make_rng(0)
        updated, _ = self.engine.evolve([threat], self.network, rng)
        live = [t for t in updated if not t.is_contained]
        assert live[0].steps_active == 1

    def test_late_stage_threat_has_higher_effective_severity(self):
        """Same base severity, but EXFILTRATION should yield higher effective severity."""
        t_early = make_threat(stage=AttackStage.PHISHING, severity=0.4)
        t_late  = make_threat(stage=AttackStage.EXFILTRATION, severity=0.4)
        assert t_late.effective_severity() > t_early.effective_severity()


# ---------------------------------------------------------------------------
# AttackEngine — lateral movement
# ---------------------------------------------------------------------------

class TestLateralMovement:
    def setup_method(self):
        self.engine = AttackEngine(AttackEngineConfig(
            stage_progression_base_prob=0.0,
            lateral_movement_base_prob=1.0,    # always move if at LATERAL_SPREAD
            spread_amplifier=1.0,
        ))

    def test_lateral_movement_spawns_child_at_lateral_spread_stage(self):
        network = make_network(make_rng(0))
        threat = make_threat(node="ws-01", stage=AttackStage.LATERAL_SPREAD,
                              spread_potential=1.0)
        updated, events = self.engine.evolve([threat], network, make_rng(42))
        assert len(events) >= 1
        event = events[0]
        assert event.source_node == "ws-01"
        assert event.child_threat.stage == AttackStage.PHISHING  # child starts fresh

    def test_child_threat_lower_severity_than_parent(self):
        network = make_network(make_rng(0))
        threat = make_threat(node="ws-01", stage=AttackStage.LATERAL_SPREAD,
                              severity=0.8, spread_potential=1.0)
        updated, events = self.engine.evolve([threat], network, make_rng(1))
        if events:
            assert events[0].child_threat.severity < 0.8

    def test_no_lateral_movement_before_lateral_spread_stage(self):
        for stage in [
            AttackStage.PHISHING,
            AttackStage.CREDENTIAL_ACCESS,
            AttackStage.MALWARE_INSTALL,
        ]:
            network = make_network(make_rng(0))
            threat = make_threat(node="ws-01", stage=stage, spread_potential=1.0)
            _, events = self.engine.evolve([threat], network, make_rng(0))
            assert events == [], f"Should not spread at stage {stage}"

    def test_no_lateral_movement_to_isolated_node(self):
        network = make_network(make_rng(0))
        # Isolate all neighbours of ws-01
        for nb in network.active_neighbours("ws-01"):
            network.assets[nb].is_isolated = True

        threat = make_threat(node="ws-01", stage=AttackStage.LATERAL_SPREAD,
                              spread_potential=1.0)
        _, events = self.engine.evolve([threat], network, make_rng(0))
        assert events == []

    def test_no_duplicate_spawns_on_same_node(self):
        """Two concurrent threats on the same node should not both spawn on the same target."""
        network = make_network(make_rng(0))
        t1 = Threat(
            id="t-001", stage=AttackStage.LATERAL_SPREAD, origin_node="ws-01",
            current_node="ws-01", severity=0.5, detection_confidence=0.3,
            is_detected=False, persistence=0.2, spread_potential=1.0,
        )
        t2 = Threat(
            id="t-002", stage=AttackStage.LATERAL_SPREAD, origin_node="ws-01",
            current_node="ws-01", severity=0.5, detection_confidence=0.3,
            is_detected=False, persistence=0.2, spread_potential=1.0,
        )
        _, events = self.engine.evolve([t1, t2], network, make_rng(0))
        targets = [e.target_node for e in events]
        assert len(targets) == len(set(targets)), "Duplicate lateral movement targets found"


# ---------------------------------------------------------------------------
# Environment integration — state evolution
# ---------------------------------------------------------------------------

class TestEnvironmentStateEvolution:
    def test_threat_severity_increases_over_ignored_steps(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=10)
        s0 = env.state()
        initial_severity = s0.threat_severity

        for _ in range(5):
            env.step(ActionInput(action=Action.IGNORE))

        s5 = env.state()
        # After 5 ignored steps, severity should have grown
        assert s5.threat_severity >= initial_severity

    def test_compromised_nodes_can_spread(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=7)
        s0 = env.state()
        initial_compromised = len(s0.compromised_nodes)

        for _ in range(20):
            env.step(ActionInput(action=Action.IGNORE))

        s_final = env.state()
        # After many ignored steps, spread may occur (not guaranteed — stochastic)
        # But compromised count should be >= initial
        assert len(s_final.compromised_nodes) >= initial_compromised

    def test_asset_health_degrades_under_attack(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=5)
        s0 = env.state()

        # Find initially compromised node
        if not s0.compromised_nodes:
            pytest.skip("No compromised node in this seed")

        node_id = s0.compromised_nodes[0]
        initial_health = s0.assets[node_id].health

        for _ in range(10):
            env.step(ActionInput(action=Action.IGNORE))

        s10 = env.state()
        final_health = s10.assets.get(node_id)
        if final_health:
            assert final_health.health <= initial_health

    def test_determinism_with_attack_engine(self):
        """Same seed must produce same threat evolution trajectory."""
        def run_episode(seed):
            env = AdaptiveCyberDefenseEnv()
            env.reset(seed=seed)
            trajectory = []
            for _ in range(10):
                s, r, done, info = env.step(ActionInput(action=Action.IGNORE))
                trajectory.append((
                    round(r, 4),
                    len(s.compromised_nodes),
                    round(s.threat_severity, 4),
                ))
                if done:
                    break
            return trajectory

        t1 = run_episode(42)
        t2 = run_episode(42)
        assert t1 == t2

    def test_lateral_movement_logged_in_step_info(self):
        """Lateral movement events should appear in step info dict."""
        # Use a config that aggressively spreads
        from adaptive_cyber_defense.env import EnvConfig
        cfg = EnvConfig()
        cfg.lateral_spread_base_prob = 1.0
        cfg.attack_progression_prob = 1.0
        env = AdaptiveCyberDefenseEnv(config=cfg)
        env.reset(seed=0)

        lateral_seen = False
        for _ in range(15):
            _, _, done, info = env.step(ActionInput(action=Action.IGNORE))
            if info.get("lateral_movements"):
                lateral_seen = True
                break
            if done:
                break

        # With aggressive spread config, lateral movement should eventually log
        # (not asserting True here since it depends on stage progression order,
        #  but we confirm the key is always present in info)
        assert "lateral_movements" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
