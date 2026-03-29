"""
Response Engine for the Adaptive Cyber Defense Simulator.

Implements the full effect model for each defender action.
Each action has precisely specified effects on threats, assets, and subsystems.

Action effect model
-------------------

BLOCK_IP
    Mechanism: Firewall/ACL rule blocks all traffic from suspected source.
    Threat effects:
      - Effective against early-stage threats (PHISHING, CREDENTIAL_ACCESS)
      - Reduced effectiveness at MALWARE_INSTALL+ (attacker already inside)
      - Detection_confidence bonus: confirms the IOC was real
      - False-positive risk: if applied to undetected/clean node it wastes
        resources and leaves a note in ActionResult
    Asset effects: none directly; minor network disruption (1% load)

ISOLATE_NODE
    Mechanism: Network quarantine — asset removed from production network.
    Threat effects:
      - Very high containment probability for active threats on this node
      - Prevents ALL future lateral movement from this node
    Asset effects:
      - asset.is_isolated = True (permanent until patched + re-enabled)
      - availability_impact = criticality × profile.availability_impact
      - health starts recovering (+recovery_rate/step) once isolated
    Cost: highest availability cost; should not be used on critical nodes
          without careful consideration.

PATCH_SYSTEM
    Mechanism: Deploy security patches and configuration hardening.
    Threat effects:
      - Directly effective only against PHISHING and CREDENTIAL_ACCESS stages
        (patches remove initial-access vulnerability before malware lands)
      - Minimal effect on MALWARE_INSTALL+ (attacker already established)
      - Increases asset.patch_level permanently
    Asset effects:
      - asset.patch_level += patch_improvement (capped at 1.0)
      - vulnerability_score decreases → future attacks harder
    Trade-off: no immediate containment for late-stage threats; invest in
               hardening while using other actions to contain active threats

RUN_DEEP_SCAN
    Mechanism: Full forensic investigation (memory analysis, log correlation).
    Threat effects:
      - Registers deep_scan boost with DetectionSystem (applied next step)
      - High detection_confidence increase immediately
      - Small containment probability (investigation may reveal and remove IOCs)
    Asset effects: temporary performance degradation (scan load)
    Cost: most expensive; justified when confidence is low and spread risk high

IGNORE
    No effect. Zero cost. Threat continues to evolve.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from ..models.action import Action, ActionInput, ACTION_PROFILES
from ..models.network import NetworkGraph
from ..models.state import AttackStage, NetworkAsset, ResourcePool, Threat

if TYPE_CHECKING:
    from .detection import DetectionSystem


# ---------------------------------------------------------------------------
# Action result
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """
    Full outcome record from applying one defender action.

    Attributes:
        action:                  The action taken.
        target_node:             Node the action was applied to.
        success:                 True if at least one threat was contained.
        threats_contained:       IDs of threats successfully contained.
        threats_affected:        Count of threats whose state changed.
        cost_paid:               Resource units actually consumed.
        availability_impact:     Estimated service degradation [0.0, 1.0].
        detection_boost_applied: How much detection confidence was boosted.
        patch_improvement:       How much patch_level was improved on asset.
        message:                 Human-readable summary.
        failure_reason:          Populated when success=False. None otherwise.
        wasted:                  True if action had zero effect (e.g. no threat
                                 on target, already isolated, etc.)
    """
    action: Action
    target_node: Optional[str]
    success: bool = False
    threats_contained: List[str] = field(default_factory=list)
    threats_affected: int = 0
    cost_paid: float = 0.0
    availability_impact: float = 0.0
    detection_boost_applied: float = 0.0
    patch_improvement: float = 0.0
    message: str = ""
    failure_reason: Optional[str] = None
    wasted: bool = False

    def to_dict(self) -> Dict:
        return {
            "action": self.action.name,
            "target": self.target_node,
            "success": self.success,
            "contained": self.threats_contained,
            "cost": round(self.cost_paid, 3),
            "availability_impact": round(self.availability_impact, 3),
            "detection_boost": round(self.detection_boost_applied, 3),
            "patch_improvement": round(self.patch_improvement, 3),
            "message": self.message,
            "wasted": self.wasted,
        }


# ---------------------------------------------------------------------------
# Stage-based effectiveness modifiers
# ---------------------------------------------------------------------------

# Multiplier applied to base_effectiveness per action × stage combination.
# Values < 1.0 mean the action is less effective at this stage.
_EFFECTIVENESS_BY_STAGE: Dict[Action, Dict[AttackStage, float]] = {
    Action.BLOCK_IP: {
        AttackStage.PHISHING:          1.20,   # block the delivery vector early
        AttackStage.CREDENTIAL_ACCESS: 1.00,
        AttackStage.MALWARE_INSTALL:   0.60,   # malware already running
        AttackStage.LATERAL_SPREAD:    0.50,   # attacker using internal channels
        AttackStage.EXFILTRATION:      0.40,   # exfil may use encrypted channels
    },
    Action.ISOLATE_NODE: {
        AttackStage.PHISHING:          0.90,
        AttackStage.CREDENTIAL_ACCESS: 0.90,
        AttackStage.MALWARE_INSTALL:   1.00,
        AttackStage.LATERAL_SPREAD:    1.10,   # best time to isolate
        AttackStage.EXFILTRATION:      1.10,
    },
    Action.PATCH_SYSTEM: {
        AttackStage.PHISHING:          1.30,   # patch before exploit lands
        AttackStage.CREDENTIAL_ACCESS: 1.10,
        AttackStage.MALWARE_INSTALL:   0.30,   # too late for patching to help
        AttackStage.LATERAL_SPREAD:    0.20,
        AttackStage.EXFILTRATION:      0.10,
    },
    Action.RUN_DEEP_SCAN: {
        # Scan effectiveness for containment is low across all stages
        # (scanning reveals, but doesn't contain)
        AttackStage.PHISHING:          0.80,
        AttackStage.CREDENTIAL_ACCESS: 0.80,
        AttackStage.MALWARE_INSTALL:   0.60,
        AttackStage.LATERAL_SPREAD:    0.50,
        AttackStage.EXFILTRATION:      0.40,
    },
}


# ---------------------------------------------------------------------------
# Response engine configuration
# ---------------------------------------------------------------------------

@dataclass
class ResponseConfig:
    # Patch improvement per PATCH_SYSTEM action [0.0, 1.0]
    patch_improvement_per_action: float = 0.20

    # Health recovery per step for isolated nodes
    isolation_health_recovery: float = 0.03

    # Penalty multiplier when applying action to already-isolated node
    redundant_action_cost_fraction: float = 0.50

    # How much detection_confidence is boosted immediately by RUN_DEEP_SCAN
    # (in addition to the DetectionSystem scan boost registered for next step)
    immediate_scan_confidence_boost: float = 0.15

    # Probability of false-positive containment waste (BLOCK_IP on clean node)
    false_positive_waste_prob: float = 0.05


# ---------------------------------------------------------------------------
# Response engine
# ---------------------------------------------------------------------------

class ResponseEngine:
    """
    Applies a defender action to the current threat list and network state.

    Usage::

        engine = ResponseEngine(config, detection_system)
        updated_threats, result = engine.apply(action, threats, network, resource_pool, rng)

    The engine:
      1. Validates resource availability
      2. Selects all threats on the target node
      3. Applies action-specific containment logic
      4. Updates asset state (patch_level, isolation flag)
      5. Registers deep-scan boost with DetectionSystem when applicable
      6. Returns updated threat list + ActionResult

    Does NOT mutate arguments — works on clones.
    """

    def __init__(
        self,
        config: Optional[ResponseConfig] = None,
        detection_system: Optional["DetectionSystem"] = None,
    ) -> None:
        self.config = config or ResponseConfig()
        self._detection_system = detection_system   # injected from env

    def set_detection_system(self, ds: "DetectionSystem") -> None:
        self._detection_system = ds

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def apply(
        self,
        action: ActionInput,
        threats: List[Threat],
        network: NetworkGraph,
        resource_pool: ResourcePool,
        rng: random.Random,
    ) -> Tuple[List[Threat], ActionResult]:
        """
        Apply a defender action and return (updated_threats, result).

        Args:
            action:        The chosen ActionInput.
            threats:       Current list of Threat objects.
            network:       Current network graph (assets mutated in place for
                           isolation/patch — callers own this).
            resource_pool: Current step's resource pool (mutated in place).
            rng:           Seeded random from the environment.

        Returns:
            (updated_threats, ActionResult)
        """
        profile = action.profile

        # IGNORE — fast path
        if action.action == Action.IGNORE:
            return threats, ActionResult(
                action=Action.IGNORE, target_node=None,
                success=False, message="no action taken",
            )

        # Resource check
        if not resource_pool.consume(profile.resource_cost):
            return threats, ActionResult(
                action=action.action,
                target_node=action.target_node,
                success=False,
                failure_reason="insufficient_resources",
                message="action skipped: resource budget exhausted",
            )

        asset = network.assets.get(action.target_node)

        # Dispatch to action-specific handler
        if action.action == Action.BLOCK_IP:
            return self._apply_block_ip(action, threats, asset, resource_pool, rng, profile.resource_cost)
        elif action.action == Action.ISOLATE_NODE:
            return self._apply_isolate(action, threats, asset, rng, profile.resource_cost)
        elif action.action == Action.PATCH_SYSTEM:
            return self._apply_patch(action, threats, asset, rng, profile.resource_cost)
        elif action.action == Action.RUN_DEEP_SCAN:
            return self._apply_deep_scan(action, threats, asset, rng, profile.resource_cost)
        else:
            return threats, ActionResult(
                action=action.action, target_node=action.target_node,
                success=False, failure_reason="unknown_action",
                message=f"unhandled action: {action.action}",
            )

    # -----------------------------------------------------------------------
    # Action handlers
    # -----------------------------------------------------------------------

    def _apply_block_ip(
        self, action, threats, asset, resource_pool, rng, cost_paid
    ) -> Tuple[List[Threat], ActionResult]:
        """
        Block malicious IP traffic to/from target node.

        Most effective against early-stage threats before malware establishes
        a persistent foothold that doesn't depend on the original IP.
        """
        target_threats = self._threats_on_node(threats, action.target_node)
        result = ActionResult(
            action=Action.BLOCK_IP, target_node=action.target_node,
            cost_paid=cost_paid,
        )

        if not target_threats:
            # Potential false positive — wasted action
            result.wasted = True
            result.failure_reason = "no_active_threat_on_node"
            result.message = f"BLOCK_IP on {action.target_node}: no active threat found (wasted)"
            return threats, result

        updated = []
        for t in threats:
            clone = t.clone()
            if t.current_node != action.target_node or t.is_contained:
                updated.append(clone)
                continue

            stage_mult = _EFFECTIVENESS_BY_STAGE[Action.BLOCK_IP].get(t.stage, 1.0)
            effective_prob = min(
                0.92,
                ACTION_PROFILES[Action.BLOCK_IP].base_effectiveness * stage_mult,
            )
            roll = rng.random()
            if roll < effective_prob:
                clone.is_contained = True
                result.threats_contained.append(t.id)
                result.success = True

            # Detection confidence boost — we just confirmed the IOC
            clone.detection_confidence = min(
                1.0,
                clone.detection_confidence + ACTION_PROFILES[Action.BLOCK_IP].detection_boost,
            )
            result.threats_affected += 1
            updated.append(clone)

        result.detection_boost_applied = ACTION_PROFILES[Action.BLOCK_IP].detection_boost
        result.availability_impact = ACTION_PROFILES[Action.BLOCK_IP].availability_impact
        result.message = (
            f"BLOCK_IP on {action.target_node}: "
            f"contained {len(result.threats_contained)}/{len(target_threats)} threats"
        )
        return updated, result

    def _apply_isolate(
        self, action, threats, asset, rng, cost_paid
    ) -> Tuple[List[Threat], ActionResult]:
        """
        Network-quarantine the target node.

        Stops lateral movement immediately. High containment probability.
        Significant availability impact, especially for critical nodes.
        """
        result = ActionResult(
            action=Action.ISOLATE_NODE, target_node=action.target_node,
            cost_paid=cost_paid,
        )

        if asset and asset.is_isolated:
            # Already isolated — charge reduced cost (already done)
            result.wasted = True
            result.failure_reason = "already_isolated"
            result.message = f"ISOLATE {action.target_node}: node already isolated (wasted)"
            return threats, result

        # Isolate the asset
        if asset:
            asset.is_isolated = True

        # Containment pass — threats on this node are likely contained
        updated = []
        for t in threats:
            clone = t.clone()
            if t.current_node != action.target_node or t.is_contained:
                updated.append(clone)
                continue

            stage_mult = _EFFECTIVENESS_BY_STAGE[Action.ISOLATE_NODE].get(t.stage, 1.0)
            effective_prob = min(
                0.95,
                ACTION_PROFILES[Action.ISOLATE_NODE].base_effectiveness * stage_mult,
            )
            roll = rng.random()
            if roll < effective_prob:
                clone.is_contained = True
                result.threats_contained.append(t.id)
                result.success = True
            result.threats_affected += 1
            updated.append(clone)

        criticality = asset.criticality if asset else 0.5
        avail_impact = ACTION_PROFILES[Action.ISOLATE_NODE].availability_impact * criticality
        result.availability_impact = round(avail_impact, 4)
        result.detection_boost_applied = ACTION_PROFILES[Action.ISOLATE_NODE].detection_boost
        result.message = (
            f"ISOLATE {action.target_node}: "
            f"quarantined node, contained {len(result.threats_contained)}/{result.threats_affected} threats "
            f"(availability impact: {result.availability_impact:.2f})"
        )
        return updated, result

    def _apply_patch(
        self, action, threats, asset, rng, cost_paid
    ) -> Tuple[List[Threat], ActionResult]:
        """
        Deploy security patches to the target node.

        Increases patch_level (reduces future vulnerability).
        Only effective for early-stage threats — patching after malware
        installation does not remove the implant.
        """
        result = ActionResult(
            action=Action.PATCH_SYSTEM, target_node=action.target_node,
            cost_paid=cost_paid,
        )

        # Improve patch level
        patch_boost = self.config.patch_improvement_per_action
        if asset:
            old_patch = asset.patch_level
            asset.patch_level = min(1.0, asset.patch_level + patch_boost)
            result.patch_improvement = round(asset.patch_level - old_patch, 4)

        # Containment pass — only works well against early stages
        target_threats = self._threats_on_node(threats, action.target_node)
        updated = []
        for t in threats:
            clone = t.clone()
            if t.current_node != action.target_node or t.is_contained:
                updated.append(clone)
                continue

            stage_mult = _EFFECTIVENESS_BY_STAGE[Action.PATCH_SYSTEM].get(t.stage, 0.3)
            effective_prob = min(
                0.80,
                ACTION_PROFILES[Action.PATCH_SYSTEM].base_effectiveness * stage_mult,
            )
            roll = rng.random()
            if roll < effective_prob:
                clone.is_contained = True
                result.threats_contained.append(t.id)
                result.success = True
            result.threats_affected += 1
            updated.append(clone)

        result.availability_impact = ACTION_PROFILES[Action.PATCH_SYSTEM].availability_impact
        result.message = (
            f"PATCH {action.target_node}: "
            f"patch_level +{result.patch_improvement:.2f}, "
            f"contained {len(result.threats_contained)}/{len(target_threats)} threats"
        )
        # Patching is useful even with 0 containments (hardening value)
        result.success = result.success or (result.patch_improvement > 0)
        result.wasted = len(target_threats) == 0 and result.patch_improvement == 0
        return updated, result

    def _apply_deep_scan(
        self, action, threats, asset, rng, cost_paid
    ) -> Tuple[List[Threat], ActionResult]:
        """
        Full forensic scan of the target node.

        Primary effect: registers a large detection boost with the DetectionSystem
        for the next simulation step (attacker TTPs are now mapped).
        Secondary effect: small immediate containment chance (IOCs removed during
        investigation).
        """
        result = ActionResult(
            action=Action.RUN_DEEP_SCAN, target_node=action.target_node,
            cost_paid=cost_paid,
        )

        # Register deep scan boost with detection system for NEXT step
        if self._detection_system and action.target_node:
            self._detection_system.register_deep_scan(action.target_node)

        # Scan boost is always applied (even with no active threats on the node)
        result.detection_boost_applied = self.config.immediate_scan_confidence_boost

        # Immediate detection_confidence boost on threat objects
        updated = []
        for t in threats:
            clone = t.clone()
            if t.current_node != action.target_node or t.is_contained:
                updated.append(clone)
                continue

            # Immediate confidence lift
            clone.detection_confidence = min(
                1.0,
                clone.detection_confidence + self.config.immediate_scan_confidence_boost,
            )

            # Small containment probability — scan may uncover and neutralise IOCs
            stage_mult = _EFFECTIVENESS_BY_STAGE[Action.RUN_DEEP_SCAN].get(t.stage, 0.6)
            effective_prob = min(
                0.50,
                ACTION_PROFILES[Action.RUN_DEEP_SCAN].base_effectiveness * stage_mult,
            )
            roll = rng.random()
            if roll < effective_prob:
                clone.is_contained = True
                result.threats_contained.append(t.id)
                result.success = True
            result.threats_affected += 1
            updated.append(clone)

        result.availability_impact = ACTION_PROFILES[Action.RUN_DEEP_SCAN].availability_impact
        result.success = result.success or (result.detection_boost_applied > 0)
        result.message = (
            f"DEEP_SCAN {action.target_node}: "
            f"detection boost registered, "
            f"contained {len(result.threats_contained)}/{result.threats_affected} threats"
        )
        return updated, result

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _threats_on_node(threats: List[Threat], node_id: str) -> List[Threat]:
        return [t for t in threats if t.current_node == node_id and not t.is_contained]


# ---------------------------------------------------------------------------
# State updater
# ---------------------------------------------------------------------------

class StateUpdater:
    """
    Applies per-step state transitions that are not owned by any single engine.

    Responsibilities:
      - Apply health degradation from active threats (severity-weighted)
      - Apply health recovery for isolated nodes
      - Derive compromise flags from current threat list
      - Recalculate network_load
      - Enforce asset health floor (0.0)

    This is separate from the ResponseEngine so each concern is testable
    independently.
    """

    def __init__(
        self,
        health_degradation_rate: float = 0.05,
        isolation_recovery_rate: float = 0.03,
    ) -> None:
        self.health_degradation_rate = health_degradation_rate
        self.isolation_recovery_rate = isolation_recovery_rate

    def update(
        self,
        network: NetworkGraph,
        threats: List[Threat],
        lateral_events: list,
    ) -> None:
        """
        Update network asset state in place.

        Args:
            network:        Current NetworkGraph (assets mutated).
            threats:        Post-response threat list.
            lateral_events: LateralMovementEvents from AttackEngine this step.
        """
        # Step 1: Reset compromise flags — re-derive from live threats
        for asset in network.assets.values():
            asset.is_compromised = False

        # Step 2: Mark compromised nodes + apply health degradation
        for threat in threats:
            if threat.is_contained:
                continue
            asset = network.assets.get(threat.current_node)
            if not asset:
                continue
            asset.is_compromised = True
            # Severity-weighted degradation: bad threats hurt faster
            degradation = self.health_degradation_rate * threat.effective_severity()
            asset.health = max(0.0, asset.health - degradation)

        # Step 3: Mark lateral movement targets as compromised
        for event in lateral_events:
            target = network.assets.get(event.target_node)
            if target:
                target.is_compromised = True

        # Step 4: Isolated nodes slowly recover health (SOC can remediate offline)
        for asset in network.assets.values():
            if asset.is_isolated and asset.is_compromised is False:
                asset.health = min(1.0, asset.health + self.isolation_recovery_rate)

    def network_load(self, network: NetworkGraph) -> float:
        """
        Compute network load as weighted fraction of affected nodes.
        Compromised nodes contribute more to load than isolated-but-clean ones.
        """
        total = len(network.assets)
        if total == 0:
            return 0.0
        load = 0.0
        for asset in network.assets.values():
            if asset.is_compromised:
                load += 1.0 * asset.criticality  # critical compromises hurt more
            elif asset.is_isolated:
                load += 0.4                       # isolated nodes degrade routing
        return round(min(1.0, load / total), 4)
