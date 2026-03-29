"""
Probabilistic Detection System for the Adaptive Cyber Defense Simulator.

Models the realistic uncertainty of a SOC's detection capability:

  - True positives:  real threats the system correctly flags
  - False negatives: real threats that slip past detection
  - False positives: clean nodes incorrectly flagged as suspicious
  - Confidence:      a per-threat score that evolves over time,
                     reflecting how sure the SOC is about each alert

Detection probability per threat is modulated by:
  - Stage visibility:   later kill-chain stages generate more telemetry
                        BUT also benefit from attacker evasion techniques
  - Threat persistence: deeply embedded attackers actively evade detection
  - Network load:       high-traffic periods mask malicious activity
  - Asset logging tier: servers/DBs have richer logs than workstations
  - Recent scan bonus:  RUN_DEEP_SCAN (Phase 5) leaves a temporary boost

Confidence evolution:
  - Grows when a threat is detected (evidence accumulates)
  - Decays slowly when missed (SOC loses track)
  - Boosts sharply when a deep scan runs on the node
  - Caps at [0.0, 1.0]
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..models.network import NetworkGraph
from ..models.state import AssetType, AttackStage, NetworkAsset, Threat


# ---------------------------------------------------------------------------
# Stage visibility profile
# ---------------------------------------------------------------------------

# How much each kill-chain stage raises the raw detection signal.
# Phishing is quiet; lateral movement and exfiltration generate network noise.
# But advanced stages also come with attacker evasion — net effect is non-linear.
_STAGE_VISIBILITY: Dict[AttackStage, float] = {
    AttackStage.PHISHING:          0.30,   # spear-phish hard to distinguish from legit mail
    AttackStage.CREDENTIAL_ACCESS: 0.45,   # credential stuffing leaves auth logs
    AttackStage.MALWARE_INSTALL:   0.55,   # AV / EDR may catch payload
    AttackStage.LATERAL_SPREAD:    0.65,   # lateral traffic anomalies visible in NDR
    AttackStage.EXFILTRATION:      0.70,   # large outbound transfers trigger DLP
}

# Logging quality per asset type — servers have SIEM integration, workstations don't
_ASSET_LOG_QUALITY: Dict[AssetType, float] = {
    AssetType.FIREWALL:    0.90,
    AssetType.ROUTER:      0.75,
    AssetType.SERVER:      0.80,
    AssetType.DATABASE:    0.85,
    AssetType.WORKSTATION: 0.45,
}


# ---------------------------------------------------------------------------
# Detection event
# ---------------------------------------------------------------------------

@dataclass
class DetectionEvent:
    """
    A single detection outcome for one node in one simulation step.

    Attributes:
        threat_id:         The threat that triggered this event (None for FP).
        node_id:           The asset being examined.
        is_true_positive:  Real threat correctly identified.
        is_false_positive: Clean node incorrectly flagged.
        is_false_negative: Real threat that was missed (no event generated,
                           but recorded for logging when threat is later found).
        updated_confidence: The threat's detection_confidence after this step.
        detection_method:  Which detection layer fired.
    """
    threat_id: Optional[str]
    node_id: str
    is_true_positive: bool
    is_false_positive: bool
    is_false_negative: bool
    updated_confidence: float
    detection_method: str

    @property
    def confidence(self) -> float:
        """Alias for updated_confidence (test-compatibility)."""
        return self.updated_confidence


# ---------------------------------------------------------------------------
# Detection configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectionConfig:
    """Tunable detection parameters."""
    # Base probability of detecting a threat given neutral conditions
    base_detection_prob: float = 0.55

    # False positive rate: probability per step that a CLEAN node raises an alert
    false_positive_rate: float = 0.08

    # Additional FP rate scaling with network load
    load_fp_amplifier: float = 0.10

    # Maximum penalty from high persistence (attacker evasion)
    persistence_evasion_cap: float = 0.35

    # How much network load suppresses detection (0 = no effect, 1 = total suppression)
    load_suppression_factor: float = 0.25

    # Per-step confidence growth when a threat IS detected
    confidence_growth_rate: float = 0.12

    # Per-step confidence decay when a threat is MISSED
    confidence_decay_rate: float = 0.06

    # Maximum confidence boost from a deep scan (consumed once, set externally)
    deep_scan_confidence_boost: float = 0.40

    # Passive evidence accumulation: confidence gained per step-active on a MISS.
    # Represents SIEM log traces, anomaly patterns, and forensic breadcrumbs that
    # accumulate even when no single detection event fires.  Prevents an active
    # threat from going permanently dark through repeated false negatives.
    time_active_evidence_rate: float = 0.007

    # Cap on the passive evidence gain per step (avoids runaway growth)
    time_active_evidence_cap: float = 0.05

    # Per-step detection probability bonus per step-active.
    # Counteracts the growing persistence-evasion penalty so a long-lived
    # threat doesn't become progressively invisible.
    time_active_detection_bonus_rate: float = 0.012

    # Maximum total time-active detection bonus (reached at ~15 steps)
    time_active_detection_bonus_cap: float = 0.18


# ---------------------------------------------------------------------------
# Detection system
# ---------------------------------------------------------------------------

class DetectionSystem:
    """
    Stateful detection engine.

    Maintains per-threat scan-boost state (set by the response engine when
    RUN_DEEP_SCAN is applied) so detection improves on the next step after
    a scan is performed.

    Usage::

        detector = DetectionSystem(config)
        threats, events = detector.run(threats, network, rng, network_load)
    """

    def __init__(self, config: Optional[DetectionConfig] = None) -> None:
        self.config = config or DetectionConfig()
        # node_id → remaining deep_scan boost (set by response engine)
        self._pending_scan_boosts: Dict[str, float] = {}

    def register_deep_scan(self, node_id: str) -> None:
        """
        Called by the response engine when RUN_DEEP_SCAN is executed.
        The boost is consumed on the next call to run().
        """
        self._pending_scan_boosts[node_id] = self.config.deep_scan_confidence_boost

    def reset(self) -> None:
        """Clear all scan boosts — call from env.reset()."""
        self._pending_scan_boosts.clear()

    # -----------------------------------------------------------------------
    # Main detection pass
    # -----------------------------------------------------------------------

    def run(
        self,
        threats: List[Threat],
        network: NetworkGraph,
        rng: random.Random,
        network_load: float,
    ) -> Tuple[List[Threat], List[DetectionEvent]]:
        """
        Run one detection pass over all active threats and clean nodes.

        For each active threat:
            - Compute detection probability
            - Roll for true positive / false negative
            - Update detection_confidence

        For each non-compromised node:
            - Roll for false positive alert

        Args:
            threats:      Current threat list (may include contained ones).
            network:      Read-only network topology.
            rng:          Seeded random instance from the environment.
            network_load: Current load fraction [0.0, 1.0].

        Returns:
            (updated_threats, detection_events)
        """
        updated: List[Threat] = []
        events: List[DetectionEvent] = []

        # Set of nodes currently under real threat
        compromised_nodes = {t.current_node for t in threats if not t.is_contained}

        # -- True positive / false negative pass ------------------------------
        for threat in threats:
            if threat.is_contained:
                updated.append(threat.clone())
                continue

            clone = threat.clone()
            asset = network.assets.get(clone.current_node)
            scan_boost = self._pending_scan_boosts.pop(clone.current_node, 0.0)

            detection_prob = self._compute_detection_prob(
                threat=clone,
                asset=asset,
                network_load=network_load,
                scan_boost=scan_boost,
            )

            roll = rng.random()
            if roll < detection_prob:
                # True positive
                clone.is_detected = True
                clone.detection_confidence = min(
                    1.0,
                    clone.detection_confidence
                    + self.config.confidence_growth_rate
                    + scan_boost,
                )
                events.append(DetectionEvent(
                    threat_id=clone.id,
                    node_id=clone.current_node,
                    is_true_positive=True,
                    is_false_positive=False,
                    is_false_negative=False,
                    updated_confidence=clone.detection_confidence,
                    detection_method=self._detection_method(clone.stage, scan_boost),
                ))
            else:
                # False negative — threat goes undetected this step.
                # Passive evidence accumulation: even on a miss, a long-lived
                # threat has left SIEM traces that partially offset the decay.
                # Net change = -(decay_rate - passive_evidence), so:
                #   young threats (steps_active≈0): full -0.06 decay
                #   old threats  (steps_active≥10): decay shrinks toward zero,
                #     preventing confidence from collapsing to 0 permanently.
                passive = min(
                    self.config.time_active_evidence_cap,
                    clone.steps_active * self.config.time_active_evidence_rate,
                )
                clone.is_detected = False
                clone.detection_confidence = max(
                    0.0,
                    clone.detection_confidence
                    - self.config.confidence_decay_rate
                    + passive,
                )
                events.append(DetectionEvent(
                    threat_id=clone.id,
                    node_id=clone.current_node,
                    is_true_positive=False,
                    is_false_positive=False,
                    is_false_negative=True,
                    updated_confidence=clone.detection_confidence,
                    detection_method="missed",
                ))

            updated.append(clone)

        # -- False positive pass on clean nodes --------------------------------
        for node_id, asset in network.assets.items():
            if node_id in compromised_nodes or asset.is_isolated:
                continue
            fp_prob = self._compute_fp_prob(network_load)
            if rng.random() < fp_prob:
                events.append(DetectionEvent(
                    threat_id=None,
                    node_id=node_id,
                    is_true_positive=False,
                    is_false_positive=True,
                    is_false_negative=False,
                    updated_confidence=0.0,
                    detection_method="false_alarm",
                ))

        return updated, events

    # -----------------------------------------------------------------------
    # Probability helpers
    # -----------------------------------------------------------------------

    def _compute_detection_prob(
        self,
        threat: Threat,
        asset: Optional[NetworkAsset],
        network_load: float,
        scan_boost: float,
    ) -> float:
        """
        Final detection probability for one threat.

        Uses an additive model so that base_detection_prob is the literal
        probability under neutral conditions (average stage, average log
        quality) and all other factors apply signed adjustments:

          base_detection_prob          baseline SOC capability
          + stage_bonus                later stages generate more telemetry
          + log_bonus                  richer logging infrastructure
          + scan_boost                 direct lift from RUN_DEEP_SCAN action
          - persistence_evasion        entrenched attackers evade detection
          - load_penalty               high traffic masks malicious signals

        Neutral values (zero adjustment):
          stage_vis centre  ≈ 0.50  (between PHISHING=0.30 and EXFIL=0.70)
          log_quality centre≈ 0.65  (between WORKSTATION=0.45 and DB=0.85)
        """
        stage_vis   = _STAGE_VISIBILITY.get(threat.stage, 0.5)
        log_quality = _ASSET_LOG_QUALITY.get(
            asset.asset_type if asset else AssetType.WORKSTATION, 0.5
        )

        # Additive bonuses relative to neutral centres
        stage_bonus = (stage_vis   - 0.50) * 0.30
        log_bonus   = (log_quality - 0.65) * 0.25

        # Persistence reduces detection (attacker living-off-the-land)
        evasion = min(
            self.config.persistence_evasion_cap,
            threat.persistence * self.config.persistence_evasion_cap,
        )

        # High network load drowns out the signal
        load_penalty = network_load * self.config.load_suppression_factor

        # Time-active bonus: accumulated behavioural evidence (SIEM logs,
        # baseline deviations, repeated network anomalies) makes a long-lived
        # threat progressively easier to detect, partially counteracting the
        # growing evasion penalty from persistence.
        time_bonus = min(
            self.config.time_active_detection_bonus_cap,
            threat.steps_active * self.config.time_active_detection_bonus_rate,
        )

        # Adaptive attacker strategy evasion bonus (APT/zero-day/insider)
        strategy_evasion = getattr(threat, "detection_evasion", 0.0)

        prob = (
            self.config.base_detection_prob
            + stage_bonus
            + log_bonus
            + scan_boost
            + time_bonus
            - evasion
            - load_penalty
            - strategy_evasion
        )
        return max(0.02, min(0.97, prob))   # never absolute certainty either way

    def _compute_fp_prob(self, network_load: float) -> float:
        """
        False positive probability for a clean node.
        Scales up with network load — busy networks produce more noise.
        """
        return min(
            0.30,
            self.config.false_positive_rate
            + network_load * self.config.load_fp_amplifier,
        )

    @staticmethod
    def _detection_method(stage: AttackStage, scan_boost: float) -> str:
        """Human-readable label for what triggered the detection."""
        if scan_boost > 0:
            return "deep_scan"
        mapping = {
            AttackStage.PHISHING:          "email_filter",
            AttackStage.CREDENTIAL_ACCESS: "auth_anomaly",
            AttackStage.MALWARE_INSTALL:   "edr_signature",
            AttackStage.LATERAL_SPREAD:    "ndr_anomaly",
            AttackStage.EXFILTRATION:      "dlp_alert",
        }
        return mapping.get(stage, "unknown")
