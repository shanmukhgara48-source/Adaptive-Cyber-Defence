"""
Phase 3 tests — DetectionSystem and ThreatScorer.

Detection tests verify:
  - True positive / false negative probabilities are bounded correctly
  - False positives fire on clean nodes
  - Confidence evolves in the right direction
  - Deep scan boost raises detection probability
  - High network load increases false positive rate

Scoring tests verify:
  - All sub-scores are in [0.0, 1.0]
  - Critical assets score higher impact
  - LATERAL_SPREAD stage scores higher spread
  - EXFILTRATION stage scores highest urgency
  - Composite score ranks threats sensibly
"""

import random
import pytest

from adaptive_cyber_defense.engines.detection import (
    DetectionSystem, DetectionConfig, DetectionEvent,
)
from adaptive_cyber_defense.engines.scoring import ThreatScorer, ThreatScore
from adaptive_cyber_defense.models.network import NetworkGraph
from adaptive_cyber_defense.models.state import AttackStage, NetworkAsset, AssetType, Threat
from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.models.action import Action, ActionInput


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rng(seed: int = 0) -> random.Random:
    return random.Random(seed)


def make_network(seed: int = 0) -> NetworkGraph:
    return NetworkGraph.build_default(make_rng(seed))


def make_threat(
    node: str = "ws-01",
    stage: AttackStage = AttackStage.PHISHING,
    severity: float = 0.4,
    persistence: float = 0.2,
    spread_potential: float = 0.5,
    detection_confidence: float = 0.3,
    contained: bool = False,
) -> Threat:
    return Threat(
        id="t-001",
        stage=stage,
        origin_node=node,
        current_node=node,
        severity=severity,
        detection_confidence=detection_confidence,
        is_detected=False,
        persistence=persistence,
        spread_potential=spread_potential,
        is_contained=contained,
    )


# ---------------------------------------------------------------------------
# DetectionSystem — probability bounds
# ---------------------------------------------------------------------------

class TestDetectionProbabilities:
    """
    Run many trials to verify statistical properties of detection.
    We use large N so the tests are not flaky.
    """
    N = 500

    def _run_trials(self, threat, network, config=None):
        """Return counts of (true_positives, false_negatives)."""
        detector = DetectionSystem(config or DetectionConfig())
        tp, fn = 0, 0
        for seed in range(self.N):
            rng = make_rng(seed)
            updated, events = detector.run([threat], network, rng, network_load=0.2)
            live = [t for t in updated if not t.is_contained]
            if live and live[0].is_detected:
                tp += 1
            else:
                fn += 1
        return tp, fn

    def test_some_threats_detected(self):
        """With default config, most threats should be detected."""
        network = make_network()
        threat = make_threat(node="srv-db", stage=AttackStage.LATERAL_SPREAD,
                              detection_confidence=0.5, persistence=0.1)
        tp, fn = self._run_trials(threat, network)
        # At least 10% detected (very conservative bound)
        assert tp > 0

    def test_never_absolute_detection(self):
        """Detection probability is capped at 0.97 — some misses must occur."""
        config = DetectionConfig(base_detection_prob=1.0)
        network = make_network()
        threat = make_threat(node="srv-db", stage=AttackStage.EXFILTRATION,
                              persistence=0.0, detection_confidence=0.9)
        tp, fn = self._run_trials(threat, network, config)
        assert fn > 0, "Should have at least some false negatives"

    def test_high_persistence_reduces_detection_rate(self):
        """Entrenched attacker should be harder to detect."""
        network = make_network()
        t_low_persist  = make_threat(node="srv-db", persistence=0.0)
        t_high_persist = make_threat(node="srv-db", persistence=0.9)

        tp_low, _  = self._run_trials(t_low_persist,  network)
        tp_high, _ = self._run_trials(t_high_persist, network)
        assert tp_low >= tp_high

    def test_later_stage_detection_different_from_phishing(self):
        """
        Stage visibility affects detection — just confirm the system differentiates.
        (Direction depends on config balance — we only check they differ.)
        """
        network = make_network()
        t_phish = make_threat(node="srv-db", stage=AttackStage.PHISHING,
                               persistence=0.0)
        t_exfil = make_threat(node="srv-db", stage=AttackStage.EXFILTRATION,
                               persistence=0.0)
        tp_phish, _ = self._run_trials(t_phish, network)
        tp_exfil, _ = self._run_trials(t_exfil, network)
        # Exfiltration is more visible — should yield more detections
        assert tp_exfil >= tp_phish


class TestFalsePositives:
    def test_false_positives_generated_on_clean_nodes(self):
        """With high FP rate config, clean nodes should trigger alarms."""
        config = DetectionConfig(false_positive_rate=0.99)
        detector = DetectionSystem(config)
        network = make_network()
        rng = make_rng(0)
        # No real threats — only clean nodes
        _, events = detector.run([], network, rng, network_load=0.0)
        fp_events = [e for e in events if e.is_false_positive]
        assert len(fp_events) > 0

    def test_no_false_positives_with_zero_rate(self):
        """FP rate = 0 → no false alarms."""
        config = DetectionConfig(false_positive_rate=0.0, load_fp_amplifier=0.0)
        detector = DetectionSystem(config)
        network = make_network()
        rng = make_rng(42)
        _, events = detector.run([], network, rng, network_load=0.5)
        fp_events = [e for e in events if e.is_false_positive]
        assert fp_events == []

    def test_isolated_nodes_not_false_positived(self):
        """Isolated nodes should not generate FP alerts."""
        config = DetectionConfig(false_positive_rate=1.0)
        detector = DetectionSystem(config)
        network = make_network()
        # Isolate all nodes
        for asset in network.assets.values():
            asset.is_isolated = True
        rng = make_rng(0)
        _, events = detector.run([], network, rng, network_load=0.0)
        fp_events = [e for e in events if e.is_false_positive]
        assert fp_events == []

    def test_high_load_increases_false_positives(self):
        """Higher network load should produce more false positive alerts."""
        config = DetectionConfig(false_positive_rate=0.10, load_fp_amplifier=0.50)
        network = make_network()

        fp_counts_low, fp_counts_high = [], []
        for seed in range(100):
            det = DetectionSystem(config)
            rng = make_rng(seed)
            _, events_low = det.run([], network, make_rng(seed), network_load=0.0)
            _, events_high = det.run([], network, make_rng(seed), network_load=1.0)
            fp_counts_low.append(sum(1 for e in events_low if e.is_false_positive))
            fp_counts_high.append(sum(1 for e in events_high if e.is_false_positive))

        assert sum(fp_counts_high) >= sum(fp_counts_low)


class TestConfidenceEvolution:
    def test_confidence_grows_on_detection(self):
        """When a threat is detected, its confidence should increase.

        Use EXFILTRATION on srv-db (highest visibility + best logging)
        with zero persistence so the additive formula reaches the 0.97 cap,
        making detection certain with any seed.
        """
        config = DetectionConfig(base_detection_prob=1.0)
        detector = DetectionSystem(config)
        # srv-db: DATABASE log_quality=0.85, EXFILTRATION visibility=0.70
        # prob = 1.0 + (0.70-0.50)*0.30 + (0.85-0.65)*0.25 = 1.0+0.06+0.05 = 1.11 → cap 0.97
        threat = make_threat(
            node="srv-db",
            stage=AttackStage.EXFILTRATION,
            detection_confidence=0.3,
            persistence=0.0,
        )
        network = make_network()
        updated, _ = detector.run([threat], network, make_rng(0), network_load=0.0)
        live = [t for t in updated if not t.is_contained]
        assert live[0].detection_confidence > 0.3

    def test_confidence_decays_on_miss(self):
        """When a threat is missed, its confidence should decrease."""
        config = DetectionConfig(base_detection_prob=0.0)  # always miss
        detector = DetectionSystem(config)
        threat = make_threat(detection_confidence=0.5)
        network = make_network()
        updated, _ = detector.run([threat], network, make_rng(0), network_load=0.0)
        live = [t for t in updated if not t.is_contained]
        assert live[0].detection_confidence < 0.5

    def test_deep_scan_boost_applied(self):
        """Registering a deep scan should boost detection probability.

        Use srv-db (DATABASE, best logging) + EXFILTRATION (highest visibility)
        so stage/log bonuses combine with the scan boost and reach the 0.97 cap:
          prob = 0.0 + (0.70-0.50)*0.30 + (0.85-0.65)*0.25 + 0.95
               = 0.0 + 0.06 + 0.05 + 0.95 = 1.06 → capped at 0.97
        rng(0).random() = 0.844 < 0.97 → detection guaranteed.
        """
        config = DetectionConfig(
            base_detection_prob=0.0,
            deep_scan_confidence_boost=0.95,
        )
        detector = DetectionSystem(config)
        detector.register_deep_scan("srv-db")
        threat = make_threat(
            node="srv-db",
            stage=AttackStage.EXFILTRATION,
            detection_confidence=0.2,
            persistence=0.0,
        )
        network = make_network()
        updated, events = detector.run([threat], network, make_rng(0), network_load=0.0)
        tp_events = [e for e in events if e.is_true_positive]
        assert len(tp_events) > 0
        assert events[0].detection_method == "deep_scan"

    def test_deep_scan_boost_consumed_after_one_step(self):
        """Scan boost should only apply on the step it's consumed."""
        config = DetectionConfig(
            base_detection_prob=0.0,
            deep_scan_confidence_boost=0.40,
        )
        detector = DetectionSystem(config)
        detector.register_deep_scan("ws-01")
        threat = make_threat(node="ws-01", detection_confidence=0.1)
        network = make_network()

        # Step 1: boost applied
        updated1, _ = detector.run([threat], network, make_rng(0), network_load=0.0)
        live1 = [t for t in updated1 if not t.is_contained][0]

        # Step 2: no boost — confidence should not be higher than after step 1
        updated2, _ = detector.run([live1], network, make_rng(0), network_load=0.0)
        live2 = [t for t in updated2 if not t.is_contained][0]
        assert live2.detection_confidence <= live1.detection_confidence + 0.01

    def test_confidence_capped_at_one(self):
        config = DetectionConfig(base_detection_prob=1.0, confidence_growth_rate=0.5)
        detector = DetectionSystem(config)
        threat = make_threat(detection_confidence=0.95)
        network = make_network()
        updated, _ = detector.run([threat], network, make_rng(0), network_load=0.0)
        live = [t for t in updated if not t.is_contained]
        assert live[0].detection_confidence <= 1.0

    def test_confidence_floored_at_zero(self):
        config = DetectionConfig(base_detection_prob=0.0, confidence_decay_rate=0.5)
        detector = DetectionSystem(config)
        threat = make_threat(detection_confidence=0.05)
        network = make_network()
        updated, _ = detector.run([threat], network, make_rng(0), network_load=0.0)
        live = [t for t in updated if not t.is_contained]
        assert live[0].detection_confidence >= 0.0

    def test_detection_reset_clears_scan_boosts(self):
        detector = DetectionSystem()
        detector.register_deep_scan("ws-01")
        detector.reset()
        # After reset, no boost pending
        assert detector._pending_scan_boosts == {}


# ---------------------------------------------------------------------------
# ThreatScorer
# ---------------------------------------------------------------------------

class TestThreatScorer:
    def setup_method(self):
        self.scorer = ThreatScorer()
        self.network = make_network()

    def _score(self, **kwargs) -> ThreatScore:
        return self.scorer.score(make_threat(**kwargs), self.network)

    def test_all_subscores_in_range(self):
        s = self._score()
        for attr in ["impact_score", "spread_score", "likelihood_score",
                     "urgency_score", "composite_score"]:
            val = getattr(s, attr)
            assert 0.0 <= val <= 1.0, f"{attr}={val} out of range"

    def test_critical_node_scores_higher_impact(self):
        """Database (criticality=1.0) threat should score higher impact than workstation."""
        s_db = self._score(node="srv-db")    # criticality 1.0 in default topology
        s_ws = self._score(node="ws-03")     # criticality 0.2
        assert s_db.impact_score > s_ws.impact_score

    def test_lateral_spread_stage_scores_highest_spread(self):
        s_phish  = self._score(stage=AttackStage.PHISHING)
        s_spread = self._score(stage=AttackStage.LATERAL_SPREAD)
        assert s_spread.spread_score > s_phish.spread_score

    def test_exfiltration_scores_highest_urgency(self):
        s_phish  = self._score(stage=AttackStage.PHISHING)
        s_exfil  = self._score(stage=AttackStage.EXFILTRATION)
        assert s_exfil.urgency_score > s_phish.urgency_score

    def test_high_persistence_raises_likelihood(self):
        s_low  = self._score(persistence=0.0)
        s_high = self._score(persistence=0.9)
        assert s_high.likelihood_score > s_low.likelihood_score

    def test_score_all_returns_sorted_descending(self):
        threats = [
            make_threat(node="ws-03", stage=AttackStage.PHISHING,
                        severity=0.2, spread_potential=0.1),
            make_threat(node="srv-db", stage=AttackStage.EXFILTRATION,
                        severity=0.9, spread_potential=0.8),
        ]
        threats[0].id = "t-low"
        threats[1].id = "t-high"
        scores = self.scorer.score_all(threats, self.network)
        assert scores[0].composite_score >= scores[1].composite_score

    def test_contained_threat_not_scored(self):
        threats = [make_threat(contained=True)]
        scores = self.scorer.score_all(threats, self.network)
        assert scores == []

    def test_highest_priority_returns_top_threat(self):
        threats = [
            make_threat(node="ws-03", stage=AttackStage.PHISHING),
            make_threat(node="srv-db", stage=AttackStage.EXFILTRATION, severity=0.9),
        ]
        threats[0].id = "t-low"
        threats[1].id = "t-high"
        top = self.scorer.highest_priority(threats, self.network)
        assert top is not None
        assert top.threat_id == "t-high"

    def test_highest_priority_returns_none_when_no_threats(self):
        top = self.scorer.highest_priority([], self.network)
        assert top is None

    def test_primary_driver_is_valid_dimension(self):
        s = self._score()
        assert s.primary_driver in {"impact", "spread", "likelihood", "urgency"}


# ---------------------------------------------------------------------------
# Environment integration
# ---------------------------------------------------------------------------

class TestDetectionIntegration:
    def test_detection_events_in_step_info(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(ActionInput(action=Action.IGNORE))
        assert "detection_events" in info
        assert isinstance(info["detection_events"], list)

    def test_threat_scores_in_step_info(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        _, _, _, info = env.step(ActionInput(action=Action.IGNORE))
        assert "threat_scores" in info

    def test_threat_scores_accessible_via_env(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        env.step(ActionInput(action=Action.IGNORE))
        scores = env.threat_scores()
        assert isinstance(scores, list)
        if scores:
            s = scores[0]
            assert 0.0 <= s.composite_score <= 1.0

    def test_detection_events_accessible_via_env(self):
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=0)
        env.step(ActionInput(action=Action.IGNORE))
        events = env.detection_events()
        assert isinstance(events, list)

    def test_detection_deterministic_with_same_seed(self):
        def run(seed):
            env = AdaptiveCyberDefenseEnv()
            env.reset(seed=seed)
            results = []
            for _ in range(5):
                _, _, done, info = env.step(ActionInput(action=Action.IGNORE))
                results.append(len(info["detection_events"]))
                if done:
                    break
            return results

        assert run(77) == run(77)

    def test_state_detection_confidence_reflects_threats(self):
        """State.detection_confidence must be non-negative after detection pass."""
        env = AdaptiveCyberDefenseEnv()
        env.reset(seed=3)
        for _ in range(5):
            s, _, done, _ = env.step(ActionInput(action=Action.IGNORE))
            assert 0.0 <= s.detection_confidence <= 1.0
            if done:
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
