"""
Decision Logic + Resource Management for the Adaptive Cyber Defense Simulator.

Two concerns live here:

1. ActionMemory
   A rolling record of past action outcomes.  Over the course of an episode
   the memory accumulates evidence about which actions work well on which
   node/stage combinations, and adjusts expected-value estimates accordingly.
   This is the "adaptive" part of Adaptive Decision Support — no ML, just
   weighted running averages.

2. DecisionEngine
   Takes the current environment state, threat scores, resource pool, and
   action memory, and produces a ranked list of ActionRecommendation objects.
   It does NOT make decisions — it is a decision-support oracle.  The agent
   (or baseline rule engine in Phase 7) consumes these recommendations.

   Scoring formula per candidate action:
       score = (
           threat_weight       × threat.composite_score
         + effectiveness_weight × adjusted_effectiveness
         + memory_weight        × memory_ev
         - availability_weight  × availability_impact
         - cost_weight          × normalised_cost
       )

   Each weight is documented so tuning is explicit, not magic.

3. ResourcePrioritiser
   Given multiple threats and a limited resource budget, decides which
   (threat, action) pairs to fund and in what order.  Returns a spending
   plan that respects the per-step budget cap.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..models.action import Action, ActionInput, ACTION_PROFILES
from ..models.state import AttackStage, EnvironmentState, ResourcePool, Threat
from ..engines.scoring import ThreatScore


# ---------------------------------------------------------------------------
# Action memory
# ---------------------------------------------------------------------------

@dataclass
class OutcomeRecord:
    """Single historical data point from one step."""
    action: Action
    node_id: str
    threat_stage: AttackStage
    success: bool          # was the threat contained / meaningfully reduced?
    resource_cost: float
    threat_score_before: float   # composite score before action
    step: int


class ActionMemory:
    """
    Rolling episode memory of action outcomes.

    Internally maintains two structures:
    - A deque of raw OutcomeRecord objects (capped at `capacity`)
    - A two-level dict: action → node_id → running weighted average of success

    The weighted average decays older records so recent outcomes have more
    influence — captures the idea that a patching strategy effective early
    in an episode may become less effective once attackers are deeply embedded.

    Attributes:
        capacity:  Maximum records retained.  Older records are evicted (FIFO).
        decay:     Exponential weight applied to each prior record when a new
                   one arrives.  0.9 = recent records count ~10× more than
                   records from 10 steps ago.
    """

    def __init__(self, capacity: int = 200, decay: float = 0.92) -> None:
        self.capacity = capacity
        self.decay = decay
        self._records: deque[OutcomeRecord] = deque(maxlen=capacity)
        # action → node → (weighted_successes, weighted_total)
        self._stats: Dict[Action, Dict[str, List[float]]] = defaultdict(
            lambda: defaultdict(lambda: [0.0, 0.0])
        )

    # -----------------------------------------------------------------------
    # Recording
    # -----------------------------------------------------------------------

    def record(
        self,
        action: Action,
        node_id: str,
        threat_stage: AttackStage,
        success: bool,
        resource_cost: float,
        threat_score_before: float,
        step: int,
    ) -> None:
        """Add one outcome to memory and update running statistics."""
        rec = OutcomeRecord(
            action=action,
            node_id=node_id,
            threat_stage=threat_stage,
            success=success,
            resource_cost=resource_cost,
            threat_score_before=threat_score_before,
            step=step,
        )
        self._records.append(rec)
        self._update_stats(action, node_id, success)

    def _update_stats(self, action: Action, node_id: str, success: bool) -> None:
        """Apply exponential decay to existing stats, then add new observation."""
        for a_key in self._stats:
            for n_key in self._stats[a_key]:
                self._stats[a_key][n_key][0] *= self.decay  # weighted successes
                self._stats[a_key][n_key][1] *= self.decay  # weighted total

        entry = self._stats[action][node_id]
        entry[0] += 1.0 if success else 0.0
        entry[1] += 1.0

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def success_rate(
        self,
        action: Action,
        node_id: Optional[str] = None,
    ) -> float:
        """
        Estimated success rate for an action, optionally filtered by node.

        If node_id is given: return node-specific rate (or global fallback).
        If node_id is None: return aggregate rate across all nodes.

        Returns 0.5 (neutral prior) when no data exists.
        """
        NEUTRAL_PRIOR = 0.5

        if node_id is not None:
            entry = self._stats[action].get(node_id)
            if entry and entry[1] > 0:
                return entry[0] / entry[1]
            # Fall through to global estimate

        # Aggregate across all nodes
        total_w = total_s = 0.0
        for n_entry in self._stats[action].values():
            total_s += n_entry[0]
            total_w += n_entry[1]

        if total_w < 1e-9:
            return NEUTRAL_PRIOR
        return total_s / total_w

    def expected_value(
        self,
        action: Action,
        node_id: str,
        fallback_effectiveness: float,
    ) -> float:
        """
        Blended expected value for (action, node):
          50% prior (ActionProfile base_effectiveness)
          50% empirical success rate from memory

        Blending shrinks toward the prior when memory is sparse.
        """
        empirical = self.success_rate(action, node_id)
        # Blend weight: how much to trust empirical vs. prior
        node_records = sum(
            1 for r in self._records
            if r.action == action and r.node_id == node_id
        )
        blend = min(0.8, node_records / 10.0)   # saturates at 10 samples
        return blend * empirical + (1.0 - blend) * fallback_effectiveness

    def wasted_action_rate(self, action: Action) -> float:
        """
        Fraction of times this action was applied but produced no containment.
        Used by the reward function to penalise repeated ineffective choices.
        """
        relevant = [r for r in self._records if r.action == action]
        if not relevant:
            return 0.0
        wasted = sum(1 for r in relevant if not r.success)
        return wasted / len(relevant)

    def recent_records(self, n: int = 10) -> List[OutcomeRecord]:
        """Return the n most recent records."""
        return list(self._records)[-n:]

    def reset(self) -> None:
        """Clear all memory — call from env.reset()."""
        self._records.clear()
        self._stats.clear()

    def summary(self) -> str:
        lines = ["ActionMemory summary:"]
        for action in Action:
            rate = self.success_rate(action)
            waste = self.wasted_action_rate(action)
            lines.append(
                f"  {action.name:16s}  success={rate:.2f}  waste={waste:.2f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Recommendation object
# ---------------------------------------------------------------------------

@dataclass
class ActionRecommendation:
    """
    A ranked action suggestion produced by the DecisionEngine.

    Attributes:
        action_input:       The concrete action to take (action + target).
        score:              Decision score in [0.0, 1.0]. Higher = more urgent.
        threat_score:       The ThreatScore that motivated this recommendation.
        expected_benefit:   Estimated threat-reduction if action succeeds.
        resource_cost:      Resource units this action will consume.
        availability_cost:  Estimated service degradation (for trade-off awareness).
        reasoning:          One-line human-readable explanation.
        is_affordable:      Whether the current resource pool can fund this.
    """
    action_input: ActionInput
    score: float
    threat_score: ThreatScore
    expected_benefit: float
    resource_cost: float
    availability_cost: float
    reasoning: str
    is_affordable: bool
    confidence: float = 0.5        # recommendation confidence [0-1]
    expected_impact: str = ""      # one-line expected impact description

    def __lt__(self, other: "ActionRecommendation") -> bool:
        return self.score < other.score

    def __repr__(self) -> str:
        affordable = "✓" if self.is_affordable else "✗"
        return (
            f"Rec({self.action_input.action.name}@{self.action_input.target_node} "
            f"score={self.score:.3f} conf={self.confidence:.2f} "
            f"benefit={self.expected_benefit:.2f} "
            f"cost={self.resource_cost:.2f} afford={affordable})"
        )


# ---------------------------------------------------------------------------
# Decision engine configuration
# ---------------------------------------------------------------------------

@dataclass
class DecisionConfig:
    """
    Scoring weight vector for the DecisionEngine.
    All weights should sum to ~1.0 for interpretability.
    """
    # How much the threat's composite score drives the recommendation
    threat_weight: float = 0.35

    # How much expected action effectiveness matters
    effectiveness_weight: float = 0.25

    # How much historical memory (success rate) adjusts the score
    memory_weight: float = 0.15

    # Penalty for actions that degrade service availability
    availability_weight: float = 0.15

    # Penalty for expensive actions when resources are low
    cost_weight: float = 0.10

    # Minimum composite threat score below which IGNORE is recommended
    ignore_threshold: float = 0.15

    # Minimum detection confidence below which RUN_DEEP_SCAN is preferred
    scan_confidence_threshold: float = 0.30

    # Resource fraction below which only cheap actions are considered
    resource_scarcity_threshold: float = 0.25


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Rule-based decision support oracle with adaptive memory weighting.

    Does NOT select actions — produces a ranked list of recommendations
    that an agent or the baseline rule engine can consume.

    Key rules (applied in order before scoring):
      1. If threat score < ignore_threshold → recommend IGNORE
      2. If detection_confidence < scan_confidence_threshold AND resources allow
         → recommend RUN_DEEP_SCAN first (gather intelligence before acting)
      3. If threat is at LATERAL_SPREAD/EXFILTRATION AND resources allow
         → recommend ISOLATE_NODE (stop spread immediately)
      4. Otherwise score all viable actions and return ranked list

    Resource scarcity rule:
      When resource_availability < scarcity_threshold, filter out actions
      costing more than 50% of remaining budget.
    """

    # Action candidates to evaluate per threat (excludes IGNORE — always available)
    _CANDIDATE_ACTIONS = [
        Action.BLOCK_IP,
        Action.ISOLATE_NODE,
        Action.PATCH_SYSTEM,
        Action.RUN_DEEP_SCAN,
    ]

    def __init__(self, config: Optional[DecisionConfig] = None) -> None:
        self.config = config or DecisionConfig()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def recommend(
        self,
        state: EnvironmentState,
        threat_scores: List[ThreatScore],
        resource_pool: ResourcePool,
        memory: ActionMemory,
    ) -> List[ActionRecommendation]:
        """
        Produce a ranked list of ActionRecommendations for the current step.

        Args:
            state:         Current EnvironmentState.
            threat_scores: Scored, sorted threat list from ThreatScorer.
            resource_pool: Current resource budget.
            memory:        Episode action memory.

        Returns:
            List of recommendations sorted by score descending.
            Always includes at least one IGNORE recommendation as a fallback.
        """
        recommendations: List[ActionRecommendation] = []

        for ts in threat_scores:
            threat = self._find_threat(ts.threat_id, state.active_threats)
            if threat is None:
                continue

            asset = state.assets.get(ts.node_id)

            # Rule 1: threat too low-scoring → recommend IGNORE
            if ts.composite_score < self.config.ignore_threshold:
                recommendations.append(self._make_ignore(ts, "threat below action threshold"))
                continue

            # Rule 2: low detection confidence → scan first
            if (
                threat.detection_confidence < self.config.scan_confidence_threshold
                and resource_pool.can_afford(ACTION_PROFILES[Action.RUN_DEEP_SCAN].resource_cost)
            ):
                rec = self._score_action(
                    Action.RUN_DEEP_SCAN, ts, threat, asset, resource_pool, memory,
                    reasoning_override="low detection confidence — gather intel first",
                )
                recommendations.append(rec)
                continue

            # Rule 3: critical spread stage → prioritise isolation
            if (
                threat.stage in (AttackStage.LATERAL_SPREAD, AttackStage.EXFILTRATION)
                and resource_pool.can_afford(ACTION_PROFILES[Action.ISOLATE_NODE].resource_cost)
                and asset and not asset.is_isolated
            ):
                rec = self._score_action(
                    Action.ISOLATE_NODE, ts, threat, asset, resource_pool, memory,
                    reasoning_override="spread/exfil stage — isolate to contain immediately",
                )
                recommendations.append(rec)
                # Also evaluate other actions in case isolation is unaffordable later
                for action in [Action.BLOCK_IP, Action.RUN_DEEP_SCAN]:
                    recommendations.append(
                        self._score_action(action, ts, threat, asset, resource_pool, memory)
                    )
                continue

            # Rule 4: general scoring of all viable actions
            viable = self._filter_by_resources(resource_pool)
            for action in viable:
                rec = self._score_action(action, ts, threat, asset, resource_pool, memory)
                recommendations.append(rec)

        # Always add an IGNORE fallback
        if threat_scores:
            top_ts = threat_scores[0]
            recommendations.append(self._make_ignore(top_ts, "fallback — preserve resources"))

        # Sort by score descending, affordable actions first
        recommendations.sort(key=lambda r: (r.is_affordable, r.score), reverse=True)
        return recommendations

    def top_recommendation(
        self,
        state: EnvironmentState,
        threat_scores: List[ThreatScore],
        resource_pool: ResourcePool,
        memory: ActionMemory,
    ) -> ActionRecommendation:
        """
        Return the single highest-scored affordable recommendation,
        or IGNORE if nothing is affordable.
        """
        recs = self.recommend(state, threat_scores, resource_pool, memory)
        for rec in recs:
            if rec.is_affordable:
                return rec
        return self._make_ignore(
            threat_scores[0] if threat_scores else _null_threat_score(),
            "no affordable actions",
        )

    # -----------------------------------------------------------------------
    # Scoring helpers
    # -----------------------------------------------------------------------

    def _score_action(
        self,
        action: Action,
        ts: ThreatScore,
        threat: Threat,
        asset,
        resource_pool: ResourcePool,
        memory: ActionMemory,
        reasoning_override: Optional[str] = None,
    ) -> ActionRecommendation:
        """Compute a full ActionRecommendation for one (action, threat) pair."""
        profile = ACTION_PROFILES[action]
        cfg = self.config

        # Expected value blends prior effectiveness with memory
        ev = memory.expected_value(action, ts.node_id, profile.base_effectiveness)

        # Scarcity-adjusted cost penalty: expensive actions penalised more
        # when resources are tight
        remaining_fraction = (
            resource_pool.remaining / resource_pool.total
            if resource_pool.total > 0 else 0.0
        )
        cost_pressure = profile.resource_cost * (2.0 - remaining_fraction)
        norm_cost = min(1.0, cost_pressure)

        # Availability cost: high for isolating critical nodes
        criticality = asset.criticality if asset else 0.5
        avail_cost = profile.availability_impact * criticality

        score = (
            cfg.threat_weight        * ts.composite_score
            + cfg.effectiveness_weight * ev
            + cfg.memory_weight        * memory.success_rate(action, ts.node_id)
            - cfg.availability_weight  * avail_cost
            - cfg.cost_weight          * norm_cost
        )
        score = max(0.0, min(1.0, score))

        expected_benefit = ts.composite_score * ev
        reasoning = reasoning_override or self._default_reasoning(action, ts)

        # Confidence: blend of scoring strength and empirical expected value
        confidence = round(min(1.0, score * 0.6 + ev * 0.4), 3)

        # Expected impact: action-specific plain-language description
        _impact_map = {
            Action.BLOCK_IP:      f"Stop inbound malicious traffic on {ts.node_id}; reduce spread risk",
            Action.ISOLATE_NODE:  f"Quarantine {ts.node_id}; halt lateral movement immediately",
            Action.PATCH_SYSTEM:  f"Close exploitable vulnerabilities on {ts.node_id}; lower re-infection risk",
            Action.RUN_DEEP_SCAN: f"Raise detection confidence on {ts.node_id} by +15–25%",
        }
        expected_impact = _impact_map.get(action, "No direct impact")

        return ActionRecommendation(
            action_input=ActionInput(action=action, target_node=ts.node_id),
            score=round(score, 4),
            threat_score=ts,
            expected_benefit=round(expected_benefit, 4),
            resource_cost=profile.resource_cost,
            availability_cost=round(avail_cost, 4),
            reasoning=reasoning,
            is_affordable=resource_pool.can_afford(profile.resource_cost),
            confidence=confidence,
            expected_impact=expected_impact,
        )

    def _filter_by_resources(self, resource_pool: ResourcePool) -> List[Action]:
        """
        Return candidate actions affordable under current budget.
        In scarcity mode, further limit to cheap actions only.
        """
        remaining_fraction = (
            resource_pool.remaining / resource_pool.total
            if resource_pool.total > 0 else 0.0
        )
        scarce = remaining_fraction < self.config.resource_scarcity_threshold
        result = []
        for action in self._CANDIDATE_ACTIONS:
            cost = ACTION_PROFILES[action].resource_cost
            if not resource_pool.can_afford(cost):
                continue
            # In scarcity mode, skip actions costing > 50% of remaining budget
            if scarce and cost > resource_pool.remaining * 0.5:
                continue
            result.append(action)
        return result if result else [Action.IGNORE]

    @staticmethod
    def _make_ignore(ts: ThreatScore, reason: str) -> ActionRecommendation:
        return ActionRecommendation(
            action_input=ActionInput(action=Action.IGNORE),
            score=0.0,
            threat_score=ts,
            expected_benefit=0.0,
            resource_cost=0.0,
            availability_cost=0.0,
            reasoning=reason,
            is_affordable=True,
            confidence=0.1,
            expected_impact="No action taken; threat may continue to escalate",
        )

    @staticmethod
    def _find_threat(
        threat_id: str, threats: List[Threat]
    ) -> Optional[Threat]:
        for t in threats:
            if t.id == threat_id:
                return t
        return None

    @staticmethod
    def _default_reasoning(action: Action, ts: ThreatScore) -> str:
        msgs = {
            Action.BLOCK_IP:      f"block malicious traffic to {ts.node_id}",
            Action.ISOLATE_NODE:  f"isolate {ts.node_id} (spread={ts.spread_score:.2f})",
            Action.PATCH_SYSTEM:  f"patch {ts.node_id} (likelihood={ts.likelihood_score:.2f})",
            Action.RUN_DEEP_SCAN: f"scan {ts.node_id} (impact={ts.impact_score:.2f})",
            Action.IGNORE:        "no action warranted",
        }
        return msgs.get(action, "")


# ---------------------------------------------------------------------------
# Resource prioritiser
# ---------------------------------------------------------------------------

@dataclass
class SpendingPlan:
    """
    Output of ResourcePrioritiser — an ordered list of (action, threat) pairs
    that fit within the step's resource budget.

    Attributes:
        funded:    Ordered list of ActionRecommendations to execute.
        deferred:  Recommendations that could not be funded.
        total_cost: Sum of resource costs in `funded`.
        budget:    Available budget this step.
    """
    funded: List[ActionRecommendation]
    deferred: List[ActionRecommendation]
    total_cost: float
    budget: float

    @property
    def utilisation(self) -> float:
        return self.total_cost / self.budget if self.budget > 0 else 0.0


class ResourcePrioritiser:
    """
    Given a ranked list of ActionRecommendations and a resource budget,
    greedily selects the highest-scoring affordable subset.

    Uses a greedy knapsack approach (sorted by score, take while affordable).
    This is optimal for this use case because:
    - All items have the same "weight" semantics (resource cost)
    - The agent makes one decision per step, not a batch
    - Simplicity is appropriate for the hackathon-ready SOC simulation context

    Duplicate node targets are deduplicated: only the highest-scoring action
    per node is funded (two actions on the same node in one step is wasteful).
    """

    def plan(
        self,
        recommendations: List[ActionRecommendation],
        resource_pool: ResourcePool,
    ) -> SpendingPlan:
        """
        Build a funding plan from a ranked recommendation list.

        Args:
            recommendations: Sorted list (score desc) from DecisionEngine.
            resource_pool:   Current per-step resource pool.

        Returns:
            SpendingPlan with funded and deferred lists.
        """
        budget = resource_pool.remaining
        remaining = budget
        funded: List[ActionRecommendation] = []
        deferred: List[ActionRecommendation] = []
        funded_nodes: set[str] = set()

        for rec in recommendations:
            node = rec.action_input.target_node

            # Deduplicate: skip if we already have an action on this node
            if node is not None and node in funded_nodes:
                deferred.append(rec)
                continue

            if rec.resource_cost <= remaining:
                funded.append(rec)
                remaining -= rec.resource_cost
                if node:
                    funded_nodes.add(node)
            else:
                deferred.append(rec)

        return SpendingPlan(
            funded=funded,
            deferred=deferred,
            total_cost=round(budget - remaining, 4),
            budget=round(budget, 4),
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _null_threat_score() -> ThreatScore:
    """Placeholder ThreatScore for edge cases (no threats active)."""
    from ..engines.scoring import ThreatScore
    return ThreatScore(
        threat_id="none", node_id="none",
        impact_score=0.0, spread_score=0.0,
        likelihood_score=0.0, urgency_score=0.0,
        composite_score=0.0, primary_driver="impact",
    )
