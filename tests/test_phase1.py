"""
Phase 1 smoke tests — verify the OpenEnv API contract and data models.
"""

import pytest
from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.models import (
    Action,
    ActionInput,
    AttackStage,
    EnvironmentState,
    NetworkAsset,
    AssetType,
    ResourcePool,
    Threat,
)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------

class TestNetworkAsset:
    def test_vulnerability_score_isolated(self):
        asset = NetworkAsset(
            id="ws-01", asset_type=AssetType.WORKSTATION,
            health=1.0, is_compromised=False, is_isolated=True,
            patch_level=0.0, criticality=0.5,
        )
        assert asset.vulnerability_score() == 0.0

    def test_vulnerability_score_unpatched(self):
        asset = NetworkAsset(
            id="ws-01", asset_type=AssetType.WORKSTATION,
            health=1.0, is_compromised=False, is_isolated=False,
            patch_level=0.0, criticality=0.5,
        )
        assert asset.vulnerability_score() == 1.0

    def test_vulnerability_score_fully_patched(self):
        asset = NetworkAsset(
            id="ws-01", asset_type=AssetType.WORKSTATION,
            health=1.0, is_compromised=False, is_isolated=False,
            patch_level=1.0, criticality=0.5,
        )
        assert asset.vulnerability_score() == 0.0


class TestAttackStage:
    def test_progression(self):
        assert AttackStage.PHISHING.next_stage() == AttackStage.CREDENTIAL_ACCESS
        assert AttackStage.CREDENTIAL_ACCESS.next_stage() == AttackStage.MALWARE_INSTALL
        assert AttackStage.EXFILTRATION.next_stage() is None

    def test_ordering(self):
        stages = list(AttackStage)
        assert stages[0] == AttackStage.PHISHING
        assert stages[-1] == AttackStage.EXFILTRATION


class TestResourcePool:
    def test_consume_success(self):
        pool = ResourcePool(total=1.0, remaining=1.0)
        assert pool.consume(0.3) is True
        assert abs(pool.remaining - 0.7) < 1e-9

    def test_consume_insufficient(self):
        pool = ResourcePool(total=1.0, remaining=0.1)
        assert pool.consume(0.5) is False
        assert pool.remaining == 0.1

    def test_utilization(self):
        pool = ResourcePool(total=1.0, remaining=0.25)
        assert abs(pool.utilization - 0.75) < 1e-9

    def test_reset_step(self):
        pool = ResourcePool(total=1.0, remaining=0.2)
        pool.reset_step()
        assert pool.remaining == pool.total


class TestThreat:
    def _make_threat(self, stage=AttackStage.PHISHING, severity=0.5):
        return Threat(
            id="t-001", stage=stage, origin_node="ws-01", current_node="ws-01",
            severity=severity, detection_confidence=0.5, is_detected=False,
            persistence=0.3, spread_potential=0.3,
        )

    def test_effective_severity_grows_with_stage(self):
        t_early = self._make_threat(AttackStage.PHISHING, severity=0.5)
        t_late  = self._make_threat(AttackStage.EXFILTRATION, severity=0.5)
        assert t_late.effective_severity() > t_early.effective_severity()

    def test_effective_severity_capped(self):
        t = self._make_threat(AttackStage.EXFILTRATION, severity=1.0)
        assert t.effective_severity() <= 1.0


class TestActionInput:
    def test_invalid_without_target(self):
        a = ActionInput(action=Action.BLOCK_IP)
        valid, reason = a.validate()
        assert not valid

    def test_ignore_valid_without_target(self):
        a = ActionInput(action=Action.IGNORE)
        valid, _ = a.validate()
        assert valid

    def test_profile_lookup(self):
        a = ActionInput(action=Action.ISOLATE_NODE, target_node="ws-01")
        assert a.profile.resource_cost > 0


# ---------------------------------------------------------------------------
# OpenEnv API tests
# ---------------------------------------------------------------------------

class TestOpenEnvAPI:
    def setup_method(self):
        self.env = AdaptiveCyberDefenseEnv()

    def test_reset_returns_state(self):
        s = self.env.reset(seed=0)
        assert isinstance(s, EnvironmentState)

    def test_reset_deterministic(self):
        s1 = self.env.reset(seed=42)
        s2 = self.env.reset(seed=42)
        assert s1.time_step == s2.time_step
        assert set(s1.compromised_nodes) == set(s2.compromised_nodes)
        assert s1.threat_severity == s2.threat_severity

    def test_reset_different_seeds_differ(self):
        s1 = self.env.reset(seed=1)
        s2 = self.env.reset(seed=9999)
        # Patch levels are seeded — very unlikely to be identical for all 8 nodes
        pl1 = [a.patch_level for a in s1.assets.values()]
        pl2 = [a.patch_level for a in s2.assets.values()]
        assert pl1 != pl2

    def test_state_before_reset_raises(self):
        env2 = AdaptiveCyberDefenseEnv()
        with pytest.raises(RuntimeError):
            env2.state()

    def test_step_before_reset_raises(self):
        env2 = AdaptiveCyberDefenseEnv()
        with pytest.raises(RuntimeError):
            env2.step(ActionInput(action=Action.IGNORE))

    def test_step_returns_tuple(self):
        self.env.reset(seed=7)
        result = self.env.step(ActionInput(action=Action.IGNORE))
        assert len(result) == 4
        next_state, reward, done, info = result
        assert isinstance(next_state, EnvironmentState)
        assert 0.0 <= reward <= 1.0
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_time_step_increments(self):
        self.env.reset(seed=1)
        _, _, _, _ = self.env.step(ActionInput(action=Action.IGNORE))
        s = self.env.state()
        assert s.time_step == 1

    def test_state_is_copy(self):
        self.env.reset(seed=1)
        s1 = self.env.state()
        s2 = self.env.state()
        assert s1 is not s2

    def test_episode_ends_within_max_steps(self):
        self.env.reset(seed=3)
        done = False
        steps = 0
        while not done:
            _, _, done, _ = self.env.step(ActionInput(action=Action.IGNORE))
            steps += 1
            if steps > 200:
                break
        assert done
        assert steps <= self.env.config.max_steps + 1

    def test_seed_property(self):
        self.env.reset(seed=55)
        assert self.env.seed() == 55

    def test_network_has_eight_nodes(self):
        s = self.env.reset(seed=0)
        assert len(s.assets) == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
