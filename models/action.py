"""
Action space definition for the Adaptive Cyber Defense Simulator.

Each action has explicit cost, benefit, and risk metadata so the reward
function and resource system can reason about trade-offs consistently.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Action(Enum):
    """
    Discrete action set available to the defending agent.

    BLOCK_IP           — Block all traffic from a suspected malicious IP.
    ISOLATE_NODE       — Cut a compromised asset from the network.
    PATCH_SYSTEM       — Deploy security patches to a vulnerable asset.
    RUN_DEEP_SCAN      — Trigger a full forensic scan of a node.
    IGNORE             — Take no action against a specific threat.
    DECRYPT            — Decrypt/restore a ransomware-encrypted node.
    REVOKE_CREDENTIALS — Revoke credentials for an insider threat / compromised account.
    QUARANTINE_SERVICE — Quarantine a compromised service dependency (supply chain).
    RESTORE_NODE       — Restore an isolated node back to normal operation.
    SCAN               — Lightweight scan (alias for RUN_DEEP_SCAN with lower cost).
    PATCH_VULNERABILITY — Alias for PATCH_SYSTEM used in response engine.
    """
    BLOCK_IP            = 0
    ISOLATE_NODE        = 1
    PATCH_SYSTEM        = 2
    RUN_DEEP_SCAN       = 3
    IGNORE              = 4
    DECRYPT             = 5
    REVOKE_CREDENTIALS  = 6
    QUARANTINE_SERVICE  = 7
    RESTORE_NODE        = 8
    SCAN                = 9
    PATCH_VULNERABILITY = 10


@dataclass(frozen=True)
class ActionProfile:
    """
    Static metadata for a given action type.

    Attributes:
        action:           The action enum value.
        resource_cost:    Resource units consumed [0.0, 1.0].
        base_effectiveness: Probability of success against a median threat [0.0, 1.0].
        availability_impact: Reduction in network availability when applied [0.0, 1.0].
                            Positive = degrades availability (e.g. isolating a server).
        detection_boost:  Increase in detection_confidence after action [0.0, 1.0].
        description:      Human-readable summary.
    """
    action: Action
    resource_cost: float
    base_effectiveness: float
    availability_impact: float
    detection_boost: float
    description: str


# ---------------------------------------------------------------------------
# Action profile registry — single source of truth for costs/benefits
# ---------------------------------------------------------------------------

ACTION_PROFILES: dict[Action, ActionProfile] = {
    Action.BLOCK_IP: ActionProfile(
        action=Action.BLOCK_IP,
        resource_cost=0.10,
        base_effectiveness=0.55,
        availability_impact=0.05,
        detection_boost=0.05,
        description=(
            "Block all inbound/outbound traffic for a suspected IP. "
            "Low cost, moderate effectiveness. Risk of false-positive blocking."
        ),
    ),
    Action.ISOLATE_NODE: ActionProfile(
        action=Action.ISOLATE_NODE,
        resource_cost=0.25,
        base_effectiveness=0.85,
        availability_impact=0.30,   # significant availability hit
        detection_boost=0.10,
        description=(
            "Quarantine a node from the network. Stops lateral spread "
            "immediately but removes the asset from service."
        ),
    ),
    Action.PATCH_SYSTEM: ActionProfile(
        action=Action.PATCH_SYSTEM,
        resource_cost=0.20,
        base_effectiveness=0.40,    # doesn't remove existing threats
        availability_impact=0.10,   # brief downtime for patching
        detection_boost=0.00,
        description=(
            "Apply security patches. Reduces future vulnerability; "
            "does not directly contain active threats."
        ),
    ),
    Action.RUN_DEEP_SCAN: ActionProfile(
        action=Action.RUN_DEEP_SCAN,
        resource_cost=0.35,
        base_effectiveness=0.30,    # reveals threats, doesn't remove them
        availability_impact=0.15,   # scan load
        detection_boost=0.40,       # major confidence improvement
        description=(
            "Full forensic scan of a node. Expensive but dramatically "
            "improves detection confidence for all threats on that node."
        ),
    ),
    Action.IGNORE: ActionProfile(
        action=Action.IGNORE,
        resource_cost=0.00,
        base_effectiveness=0.00,
        availability_impact=0.00,
        detection_boost=0.00,
        description=(
            "Take no defensive action. Zero cost; threat continues to evolve."
        ),
    ),
    Action.DECRYPT: ActionProfile(
        action=Action.DECRYPT,
        resource_cost=0.40,
        base_effectiveness=0.75,
        availability_impact=0.20,
        detection_boost=0.10,
        description=(
            "Decrypt and restore a ransomware-encrypted node. "
            "Expensive but reverses encryption damage."
        ),
    ),
    Action.REVOKE_CREDENTIALS: ActionProfile(
        action=Action.REVOKE_CREDENTIALS,
        resource_cost=0.15,
        base_effectiveness=0.90,
        availability_impact=0.05,
        detection_boost=0.15,
        description=(
            "Revoke credentials of a compromised account. "
            "Effective against insider threats and stolen creds."
        ),
    ),
    Action.QUARANTINE_SERVICE: ActionProfile(
        action=Action.QUARANTINE_SERVICE,
        resource_cost=0.30,
        base_effectiveness=0.80,
        availability_impact=0.25,
        detection_boost=0.10,
        description=(
            "Quarantine a compromised service dependency. "
            "Stops supply-chain propagation."
        ),
    ),
    Action.RESTORE_NODE: ActionProfile(
        action=Action.RESTORE_NODE,
        resource_cost=0.20,
        base_effectiveness=1.00,
        availability_impact=-0.30,  # negative = restores availability
        detection_boost=0.00,
        description=(
            "Restore an isolated node back to normal operation. "
            "Recovers availability; node must be verified clean first."
        ),
    ),
    Action.SCAN: ActionProfile(
        action=Action.SCAN,
        resource_cost=0.15,
        base_effectiveness=0.20,
        availability_impact=0.05,
        detection_boost=0.20,
        description=(
            "Lightweight scan of a node. Lower cost than RUN_DEEP_SCAN; "
            "moderate detection boost."
        ),
    ),
    Action.PATCH_VULNERABILITY: ActionProfile(
        action=Action.PATCH_VULNERABILITY,
        resource_cost=0.20,
        base_effectiveness=0.40,
        availability_impact=0.10,
        detection_boost=0.00,
        description=(
            "Apply vulnerability patch. Alias for PATCH_SYSTEM. "
            "Reduces future attack surface; does not contain active threats."
        ),
    ),
}


@dataclass
class ActionInput:
    """
    A concrete action chosen by the agent for one environment step.

    Attributes:
        action:          Which action to take.
        target_node:     Asset ID to act upon (required for all except IGNORE).
        target_threat_id: Specific threat to address (optional; if None the
                          environment picks the highest-severity threat on the node).
    """
    action: Action
    target_node: Optional[str] = None
    target_threat_id: Optional[str] = None

    @property
    def profile(self) -> ActionProfile:
        return ACTION_PROFILES[self.action]

    def validate(self) -> tuple[bool, str]:
        """
        Returns (is_valid, reason).
        Agents should call this before submitting; env also validates on step().
        """
        if self.action != Action.IGNORE and self.target_node is None:
            return False, f"Action {self.action.name} requires a target_node."
        return True, "ok"
