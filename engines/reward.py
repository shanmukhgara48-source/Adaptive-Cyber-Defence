"""
Reward Function for the Adaptive Cyber Defense Simulator.

Produces a continuous reward in [0.0, 1.0] per simulation step.

Design goals
------------
1. Partial progress signals — the agent gets useful feedback every step,
   not just at episode end.  Even without containment, reducing severity
   or preserving resources gives a small positive signal.

2. Multi-component decomposition — each component is inspectable so a
   developer can diagnose exactly why the reward was high or low.

3. Stage-aware containment bonus — containing a threat at PHISHING is
   worth more than containing it at EXFILTRATION (less damage done).

4. Availability-aware — isolating critical nodes is penalised relative
   to how much availability was lost.

Reward components
-----------------
containment_bonus   [0.0 – ∞ before clip]
    + score for each threat contained this step, weighted by its
      composite threat_score and early-stage speed bonus.

severity_reduction  [0.0 – ∞ before clip]
    + score proportional to the drop in aggregate threat severity
      between the previous step and this one.

resource_efficiency [0.0 – 1.0]
    + small bonus for completing the step with resources left over.
      Encourages the agent not to over-spend on low-value actions.

survival_bonus      [0.0 – 1.0]
    + per-step reward for keeping critical assets healthy.
      Ensures the agent is rewarded for defensive success even when
      no threats are being actively contained.

spread_penalty      [0.0 – ∞ before clip]
    - deduction for each new lateral movement event.
      Scaled by the criticality of the newly infected node.

waste_penalty       [0.0 – 1.0]
    - deduction when the action had zero effect (wasted action).

false_positive_penalty [0.0 – 1.0]
    - deduction per false-positive detection event.
      Penalises the agent for chasing ghost alerts.

availability_penalty [0.0 – 1.0]
    - deduction proportional to the availability impact of the action.
      Isolating a critical node should cost the agent something.

Final formula
-------------
    raw = (
          w.containment  × containment_bonus
        + w.severity     × severity_reduction
        + w.efficiency   × resource_efficiency
        + w.survival     × survival_bonus
        - w.spread       × spread_penalty
        - w.waste        × waste_penalty
        - w.fp           × false_positive_penalty
        - w.availability × availability_penalty
    )
    reward = clamp(raw, 0.0, 1.0)

Weight defaults are calibrated so:
  - An agent that contains all threats immediately scores ~0.8–0.9
  - An agent that only ignores scores ~0.3–0.5 (survival keeps it positive)
  - An agent that wastes all actions and lets threats spread scores ~0.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..engines.attack import LateralMovementEvent
from ..engines.detection import DetectionEvent
from ..engines.response import ActionResult
from ..engines.scoring import ThreatScore
from ..models.network import NetworkGraph
from ..models.state import AttackStage, EnvironmentState, ResourcePool, Threat


# ---------------------------------------------------------------------------
# Reward weights
# ---------------------------------------------------------------------------

@dataclass
class RewardWeights:
    """
    Weight vector for the reward function.
    All weights ≥ 0.  Penalty weights are subtracted.

    Defaults calibrated for medium-difficulty balanced play.
    Override for task-specific tuning (EASY tasks use lighter penalties).
    """
    containment:  float = 0.40   # reward per threat contained
    severity:     float = 0.20   # reward for severity reduction
    efficiency:   float = 0.10   # reward for resource efficiency
    survival:     float = 0.15   # per-step critical-asset health reward
    spread:       float = 0.25   # penalty per lateral movement
    waste:        float = 0.07   # penalty for wasted actions (reduced from 0.15 —
                                 # the old value fully cancelled the survival bonus
                                 # on any wasted step, discouraging exploration)
    false_pos:    float = 0.05   # penalty per false positive
    availability: float = 0.10   # penalty for availability impact


# ---------------------------------------------------------------------------
# Reward breakdown (returned alongside the scalar)
# ---------------------------------------------------------------------------

@dataclass
class RewardBreakdown:
    """
    Decomposed reward for logging and diagnostics.
    All sub-values are the weighted contributions (not raw component values).
    """
    containment_bonus:      float = 0.0
    severity_reduction:     float = 0.0
    resource_efficiency:    float = 0.0
    survival_bonus:         float = 0.0
    spread_penalty:         float = 0.0
    waste_penalty:          float = 0.0
    false_positive_penalty: float = 0.0
    availability_penalty:   float = 0.0
    total:                  float = 0.0

    def to_dict(self) -> dict:
        return {
            "containment":    round(self.containment_bonus, 4),
            "severity":       round(self.severity_reduction, 4),
            "efficiency":     round(self.resource_efficiency, 4),
            "survival":       round(self.survival_bonus, 4),
            "spread_penalty": round(-self.spread_penalty, 4),
            "waste_penalty":  round(-self.waste_penalty, 4),
            "fp_penalty":     round(-self.false_positive_penalty, 4),
            "avail_penalty":  round(-self.availability_penalty, 4),
            "total":          round(self.total, 4),
        }


# ---------------------------------------------------------------------------
# Stage speed bonus lookup
# ---------------------------------------------------------------------------

# How much early containment is worth relative to containing at EXFILTRATION.
# PHISHING containment = full bonus; EXFILTRATION = minimal (damage done).
_SPEED_BONUS: dict[AttackStage, float] = {
    AttackStage.PHISHING:          1.00,
    AttackStage.CREDENTIAL_ACCESS: 0.80,
    AttackStage.MALWARE_INSTALL:   0.55,
    AttackStage.LATERAL_SPREAD:    0.30,
    AttackStage.EXFILTRATION:      0.10,
}


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

class RewardFunction:
    """
    Computes the per-step reward from environment transition data.

    Usage::

        rf = RewardFunction(weights)
        reward, breakdown = rf.compute(
            state_before, state_after,
            action_result, threat_scores_before,
            lateral_events, detection_events,
            resource_pool,
        )
    """

    def __init__(self, weights: Optional[RewardWeights] = None) -> None:
        self.weights = weights or RewardWeights()

    def compute(
        self,
        state_before: EnvironmentState,
        state_after: EnvironmentState,
        action_result: ActionResult,
        threat_scores_before: List[ThreatScore],
        lateral_events: List[LateralMovementEvent],
        detection_events: List[DetectionEvent],
        resource_pool: ResourcePool,
        network: NetworkGraph,
    ) -> tuple[float, RewardBreakdown]:
        """
        Compute the reward for one environment step.

        Args:
            state_before:         State snapshot before evolution + action.
            state_after:          State snapshot after all updates.
            action_result:        The ActionResult from the ResponseEngine.
            threat_scores_before: ThreatScores from Phase 3 (pre-action).
            lateral_events:       LateralMovementEvents from AttackEngine.
            detection_events:     DetectionEvents from DetectionSystem.
            resource_pool:        End-of-step resource pool (after consumption).
            network:              Current network (for criticality queries).

        Returns:
            (reward: float, breakdown: RewardBreakdown)
            reward is clamped to [0.0, 1.0].
        """
        w = self.weights
        bd = RewardBreakdown()

        # ---- 1. Containment bonus ----------------------------------------
        # Find threats that were active before but contained after
        active_before_ids = {t.id for t in state_before.active_threats}
        contained_after_ids = {
            t.id for t in state_after.active_threats if t.is_contained
        }
        newly_contained_ids = active_before_ids & contained_after_ids

        containment_raw = 0.0
        for tid in newly_contained_ids:
            # Look up the threat score (pre-action value)
            ts = next((s for s in threat_scores_before if s.threat_id == tid), None)
            score_weight = ts.composite_score if ts else 0.5

            # Look up the stage for the speed bonus
            threat = next(
                (t for t in state_before.active_threats if t.id == tid), None
            )
            speed = _SPEED_BONUS.get(threat.stage, 0.5) if threat else 0.5
            containment_raw += score_weight * speed

        # Normalise: 1 fully-scored threat contained = 1.0 raw (clip after)
        bd.containment_bonus = w.containment * min(1.0, containment_raw)

        # ---- 2. Severity reduction ----------------------------------------
        sev_before = state_before.threat_severity
        sev_after  = state_after.threat_severity
        sev_reduction = max(0.0, sev_before - sev_after)
        bd.severity_reduction = w.severity * sev_reduction

        # ---- 3. Resource efficiency ----------------------------------------
        if resource_pool.total > 0:
            efficiency = resource_pool.remaining / resource_pool.total
        else:
            efficiency = 1.0
        bd.resource_efficiency = w.efficiency * efficiency

        # ---- 4. Survival bonus (critical assets still healthy) -------------
        critical_assets = [
            a for a in network.assets.values() if a.criticality >= 0.7
        ]
        if critical_assets:
            avg_health = sum(
                a.health * a.criticality for a in critical_assets
            ) / sum(a.criticality for a in critical_assets)
        else:
            avg_health = 1.0
        bd.survival_bonus = w.survival * avg_health

        # ---- 5. Spread penalty --------------------------------------------
        spread_raw = 0.0
        for event in lateral_events:
            target_asset = network.assets.get(event.target_node)
            crit = target_asset.criticality if target_asset else 0.5
            spread_raw += crit
        bd.spread_penalty = w.spread * min(1.0, spread_raw)

        # ---- 6. Waste penalty ---------------------------------------------
        if action_result.wasted:
            bd.waste_penalty = w.waste
        else:
            bd.waste_penalty = 0.0

        # ---- 7. False positive penalty ------------------------------------
        fp_count = sum(1 for e in detection_events if e.is_false_positive)
        # Normalise by number of clean nodes (4 FPs on 8 clean nodes = 0.5 rate)
        clean_node_count = max(1, len(network.assets) - len(state_after.compromised_nodes))
        fp_rate = min(1.0, fp_count / clean_node_count)
        bd.false_positive_penalty = w.false_pos * fp_rate

        # ---- 8. Availability penalty -------------------------------------
        bd.availability_penalty = w.availability * action_result.availability_impact

        # ---- Final reward ------------------------------------------------
        raw = (
            bd.containment_bonus
            + bd.severity_reduction
            + bd.resource_efficiency
            + bd.survival_bonus
            - bd.spread_penalty
            - bd.waste_penalty
            - bd.false_positive_penalty
            - bd.availability_penalty
        )
        bd.total = max(0.0, min(1.0, raw))
        return bd.total, bd
