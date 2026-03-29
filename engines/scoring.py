"""
Threat Scoring System for the Adaptive Cyber Defense Simulator.

Produces a structured, multi-factor score for each active threat.
Used by:
  - The reward function (Phase 6): containment quality depends on what was contained
  - The baseline agent (Phase 7): rule-based prioritisation uses composite score
  - The step info dict: provides interpretable threat intelligence to the agent

Score dimensions
----------------
impact_score
    How much damage is happening RIGHT NOW on the current node.
    Factors: asset criticality, health already lost, attack stage.

spread_score
    How likely is this threat to infect NEW nodes in coming steps.
    Factors: spread_potential, number of reachable active neighbours,
    current stage (LATERAL_SPREAD stage is highest risk).

likelihood_score
    How probable is the threat to succeed in its next objective.
    Factors: asset vulnerability, persistence, detection_confidence (high
    confidence means SOC has a handle on it — slightly reduces urgency).

urgency_score
    Time pressure — how fast does this threat need a response.
    Factors: steps_active (older = more entrenched), severity growth rate,
    proximity to EXFILTRATION stage.

composite_score  [0.0, 1.0]
    Weighted sum of the four dimensions:
        0.35 × impact + 0.25 × spread + 0.20 × likelihood + 0.20 × urgency
    Weights tuned so that critical-node threats always surface to the top
    even if spread is low (e.g. targeted database attack).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..models.network import NetworkGraph
from ..models.state import AttackStage, NetworkAsset, Threat


# ---------------------------------------------------------------------------
# Score object
# ---------------------------------------------------------------------------

@dataclass
class ThreatScore:
    """
    Full decomposed score for a single Threat.

    All sub-scores are in [0.0, 1.0].  composite_score is the primary
    signal used for prioritisation.
    """
    threat_id: str
    node_id: str

    impact_score:      float   # damage on current node
    spread_score:      float   # lateral expansion risk
    likelihood_score:  float   # probability of next-step success
    urgency_score:     float   # time pressure

    composite_score:   float   # weighted aggregate

    # Human-readable explanation for the highest contributor
    primary_driver: str

    def __lt__(self, other: "ThreatScore") -> bool:
        return self.composite_score < other.composite_score

    def __repr__(self) -> str:
        return (
            f"ThreatScore({self.threat_id}@{self.node_id} "
            f"composite={self.composite_score:.3f} "
            f"[impact={self.impact_score:.2f} spread={self.spread_score:.2f} "
            f"likelihood={self.likelihood_score:.2f} urgency={self.urgency_score:.2f}] "
            f"driver={self.primary_driver})"
        )


# ---------------------------------------------------------------------------
# Stage weights for urgency
# ---------------------------------------------------------------------------

_STAGE_URGENCY: dict[AttackStage, float] = {
    AttackStage.PHISHING:          0.20,
    AttackStage.CREDENTIAL_ACCESS: 0.40,
    AttackStage.MALWARE_INSTALL:   0.60,
    AttackStage.LATERAL_SPREAD:    0.80,
    AttackStage.EXFILTRATION:      1.00,
}

_SPREAD_STAGE_MULTIPLIER: dict[AttackStage, float] = {
    AttackStage.PHISHING:          0.10,
    AttackStage.CREDENTIAL_ACCESS: 0.20,
    AttackStage.MALWARE_INSTALL:   0.40,
    AttackStage.LATERAL_SPREAD:    1.00,   # peak spread risk
    AttackStage.EXFILTRATION:      0.50,   # already exfiltrating — spread less important
}

# Composite weight vector (must sum to 1.0)
_WEIGHTS = {
    "impact":      0.35,
    "spread":      0.25,
    "likelihood":  0.20,
    "urgency":     0.20,
}


# ---------------------------------------------------------------------------
# Threat scorer
# ---------------------------------------------------------------------------

class ThreatScorer:
    """
    Stateless scorer — call score() for each threat each step.

    Usage::

        scorer = ThreatScorer()
        scores = scorer.score_all(threats, network)
        prioritised = sorted(scores, reverse=True)
    """

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def score(
        self,
        threat: Threat,
        network: NetworkGraph,
    ) -> ThreatScore:
        """
        Compute the full ThreatScore for a single active threat.

        Args:
            threat:  The Threat to score.
            network: Current network graph (for reachability queries).

        Returns:
            ThreatScore with all sub-scores and composite.
        """
        asset = network.assets.get(threat.current_node)

        impact      = self._impact_score(threat, asset)
        spread      = self._spread_score(threat, network)
        likelihood  = self._likelihood_score(threat, asset)
        urgency     = self._urgency_score(threat)

        composite = (
            _WEIGHTS["impact"]     * impact
            + _WEIGHTS["spread"]     * spread
            + _WEIGHTS["likelihood"] * likelihood
            + _WEIGHTS["urgency"]    * urgency
        )
        composite = min(1.0, max(0.0, composite))

        # Identify primary driver for explainability
        scores_map = {
            "impact": impact, "spread": spread,
            "likelihood": likelihood, "urgency": urgency,
        }
        primary_driver = max(scores_map, key=lambda k: scores_map[k])

        return ThreatScore(
            threat_id=threat.id,
            node_id=threat.current_node,
            impact_score=round(impact, 4),
            spread_score=round(spread, 4),
            likelihood_score=round(likelihood, 4),
            urgency_score=round(urgency, 4),
            composite_score=round(composite, 4),
            primary_driver=primary_driver,
        )

    def score_all(
        self,
        threats: List[Threat],
        network: NetworkGraph,
    ) -> List[ThreatScore]:
        """
        Score all active (non-contained) threats and return sorted list
        (highest composite first).
        """
        scores = [
            self.score(t, network)
            for t in threats
            if not t.is_contained
        ]
        return sorted(scores, reverse=True)

    def highest_priority(
        self,
        threats: List[Threat],
        network: NetworkGraph,
    ) -> Optional[ThreatScore]:
        """Return the ThreatScore of the most dangerous active threat, or None."""
        scored = self.score_all(threats, network)
        return scored[0] if scored else None

    # -----------------------------------------------------------------------
    # Sub-score computations
    # -----------------------------------------------------------------------

    def _impact_score(
        self,
        threat: Threat,
        asset: Optional[NetworkAsset],
    ) -> float:
        """
        Impact = how much damage is occurring on the current node.

        Components:
          - Asset criticality (database breach >> workstation breach)
          - Health already degraded (lost health = sunk cost of damage)
          - Current severity of the threat
          - Stage weight (exfiltration on a DB is catastrophic)
        """
        criticality   = asset.criticality if asset else 0.5
        health_lost   = 1.0 - (asset.health if asset else 1.0)
        stage_weight  = _STAGE_URGENCY[threat.stage]

        impact = (
            0.40 * criticality
            + 0.25 * health_lost
            + 0.20 * threat.severity
            + 0.15 * stage_weight
        )
        return min(1.0, impact)

    def _spread_score(
        self,
        threat: Threat,
        network: NetworkGraph,
    ) -> float:
        """
        Spread = how likely and how damaging is lateral movement.

        Components:
          - spread_potential of the threat itself
          - Stage multiplier (peak at LATERAL_SPREAD)
          - Number of reachable active neighbours weighted by their criticality
        """
        stage_mult = _SPREAD_STAGE_MULTIPLIER[threat.stage]
        neighbours = network.active_neighbours(threat.current_node)

        # Neighbour criticality mass: high-value neighbours amplify spread risk
        if neighbours:
            avg_crit = sum(
                network.assets[nb].criticality for nb in neighbours
            ) / len(neighbours)
            neighbour_factor = min(1.0, len(neighbours) / 4.0) * avg_crit
        else:
            neighbour_factor = 0.0

        spread = (
            0.50 * threat.spread_potential * stage_mult
            + 0.30 * neighbour_factor
            + 0.20 * (1.0 - threat.detection_confidence)  # undetected = higher risk
        )
        return min(1.0, spread)

    def _likelihood_score(
        self,
        threat: Threat,
        asset: Optional[NetworkAsset],
    ) -> float:
        """
        Likelihood = probability the attack succeeds in its next objective.

        Components:
          - Asset vulnerability (unpatched target → easy win for attacker)
          - Threat persistence (entrenched attacker has inside knowledge)
          - Inverse of detection_confidence (if SOC has high confidence, they
            can respond; low confidence means attacker operates freely)
        """
        vulnerability = asset.vulnerability_score() if asset else 0.5
        detect_gap    = 1.0 - threat.detection_confidence   # undetected = free rein

        likelihood = (
            0.45 * vulnerability
            + 0.35 * threat.persistence
            + 0.20 * detect_gap
        )
        return min(1.0, likelihood)

    def _urgency_score(self, threat: Threat) -> float:
        """
        Urgency = time pressure to respond.

        Components:
          - Stage proximity to EXFILTRATION (higher stage = less time)
          - Steps active: older threats are harder to remove (more entrenched)
          - Severity already accumulated
        """
        stage_urgency   = _STAGE_URGENCY[threat.stage]
        age_factor      = min(1.0, threat.steps_active / 20.0)  # saturates at 20 steps

        urgency = (
            0.50 * stage_urgency
            + 0.30 * age_factor
            + 0.20 * threat.severity
        )
        return min(1.0, urgency)
