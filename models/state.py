"""
Core state data models for the Adaptive Cyber Defense Simulator.

All state objects are immutable-by-convention dataclasses.
The environment owns mutation; components receive copies.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AssetType(Enum):
    WORKSTATION = "workstation"
    SERVER      = "server"
    ROUTER      = "router"
    DATABASE    = "database"
    FIREWALL    = "firewall"


class ThreatSeverity(Enum):
    """Severity levels for threat classification."""
    LOW      = 0
    MEDIUM   = 1
    HIGH     = 2
    CRITICAL = 3

    @classmethod
    def from_score(cls, score: float) -> "ThreatSeverity":
        """Map a 0-1 severity score to an enum level."""
        if score < 0.25:
            return cls.LOW
        elif score < 0.50:
            return cls.MEDIUM
        elif score < 0.75:
            return cls.HIGH
        else:
            return cls.CRITICAL


class AttackStage(Enum):
    """Ordered stages of a kill-chain attack."""
    PHISHING          = 0   # Initial contact / delivery
    CREDENTIAL_ACCESS = 1   # Credential harvesting
    MALWARE_INSTALL   = 2   # Payload execution
    LATERAL_SPREAD    = 3   # Moving through the network
    EXFILTRATION      = 4   # Data theft / impact

    def next_stage(self) -> Optional["AttackStage"]:
        """Return the next kill-chain stage, or None if at final stage."""
        members = list(AttackStage)
        idx = members.index(self)
        return members[idx + 1] if idx + 1 < len(members) else None

    @property
    def technique_id(self) -> str:
        """MITRE ATT&CK technique ID for this kill-chain stage."""
        _map = {0: "T1566", 1: "T1078", 2: "T1204", 3: "T1021", 4: "T1041"}
        return _map.get(self.value, "T0000")

    @property
    def technique_name(self) -> str:
        """MITRE ATT&CK technique name for this kill-chain stage."""
        _map = {
            0: "Phishing",
            1: "Valid Accounts",
            2: "User Execution",
            3: "Remote Services",
            4: "Exfiltration Over C2 Channel",
        }
        return _map.get(self.value, "Unknown Technique")

    @property
    def tactic(self) -> str:
        """MITRE ATT&CK tactic name for this kill-chain stage."""
        _map = {
            0: "Initial Access",
            1: "Credential Access",
            2: "Execution",
            3: "Lateral Movement",
            4: "Exfiltration",
        }
        return _map.get(self.value, "Unknown")

    @property
    def tactic_id(self) -> str:
        """MITRE ATT&CK tactic ID for this kill-chain stage."""
        _map = {0: "TA0001", 1: "TA0006", 2: "TA0002", 3: "TA0008", 4: "TA0010"}
        return _map.get(self.value, "TA0000")


# ---------------------------------------------------------------------------
# Network asset
# ---------------------------------------------------------------------------

@dataclass
class NetworkAsset:
    """
    Represents a single device or server in the simulated network.

    Attributes:
        id:           Unique identifier (e.g. "ws-01", "srv-db").
        asset_type:   Category of asset (workstation, server, …).
        health:       Current health score [0.0, 1.0]. 0.0 = fully down.
        is_compromised: True if adversary has foothold.
        is_isolated:  True if defender has network-isolated this asset.
        patch_level:  Software patch currency [0.0, 1.0]. Low = more vulnerable.
        criticality:  Business importance [0.0, 1.0]. High = bigger impact if lost.
        connected_to: IDs of directly connected assets (adjacency list).
    """
    id: str
    asset_type: AssetType
    health: float               # [0.0, 1.0]
    is_compromised: bool
    is_isolated: bool
    patch_level: float          # [0.0, 1.0]
    criticality: float          # [0.0, 1.0]
    connected_to: List[str] = field(default_factory=list)

    def vulnerability_score(self) -> float:
        """
        Derived score: how easy is this asset to attack?
        Higher patch_level → lower vulnerability.
        """
        return max(0.0, 1.0 - self.patch_level) * (1.0 if not self.is_isolated else 0.0)

    def clone(self) -> "NetworkAsset":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# NetworkNode (extended node with 0-100 health scale and clamping)
# ---------------------------------------------------------------------------

@dataclass
class NetworkNode:
    """
    Network node with 0-100 health scale and automatic clamping.
    Used for TC001-TC015 style tests; maps to NetworkAsset semantics.
    """
    id: str
    asset_type: AssetType = AssetType.WORKSTATION
    health: float = 100.0            # [0.0, 100.0]
    is_compromised: bool = False
    is_isolated: bool = False
    patch_level: float = 1.0         # [0.0, 1.0]
    criticality: float = 0.5         # [0.0, 1.0]
    is_critical: bool = False
    connected_to: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.health = max(0.0, min(100.0, self.health))

    def connect(self, other: "NetworkNode") -> None:
        """Bidirectional connection between two nodes."""
        if other.id not in self.connected_to:
            self.connected_to.append(other.id)
        if self.id not in other.connected_to:
            other.connected_to.append(self.id)

    def set_health(self, value: float) -> None:
        self.health = max(0.0, min(100.0, value))

    def vulnerability_score(self) -> float:
        return max(0.0, 1.0 - self.patch_level) * (1.0 if not self.is_isolated else 0.0)

    def clone(self) -> "NetworkNode":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# User, Service, Event (extended models for TC010, TC012, TC013)
# ---------------------------------------------------------------------------

@dataclass
class User:
    """Represents a network user account."""
    id: str
    username: str
    role: str = "user"          # "user", "admin", "service"
    is_admin: bool = False
    node_id: Optional[str] = None
    is_active: bool = True

    def __post_init__(self):
        if self.role == "admin":
            self.is_admin = True


@dataclass
class Service:
    """Represents a running network service."""
    id: str
    name: str
    node_id: str
    is_critical: bool = False
    is_running: bool = True
    version: str = "1.0.0"
    dependencies: List[str] = field(default_factory=list)
    is_compromised: bool = False


@dataclass
class Event:
    """An event emitted by the simulation (detection, attack, response, etc.)."""
    type: str
    payload: Any = None
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    priority: int = 0   # 0=low, 1=medium, 2=high, 3=critical

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "source": self.source,
            "priority": self.priority,
        }


# ---------------------------------------------------------------------------
# Threat
# ---------------------------------------------------------------------------

@dataclass
class Threat:
    """
    An active attack campaign tracked by the environment.

    Attributes:
        id:                  Unique threat identifier.
        stage:               Current kill-chain stage.
        origin_node:         Asset where the attack started.
        current_node:        Asset currently under active attack.
        severity:            Combined impact score [0.0, 1.0].
        detection_confidence: Probability the detection system has flagged this
                             threat [0.0, 1.0].  NOT a boolean — reflects
                             uncertainty.
        is_detected:         Whether a detection event was raised this step
                             (can be false-positive or miss).
        persistence:         How deeply embedded the attacker is [0.0, 1.0].
                             High persistence → harder to remove.
        spread_potential:    Likelihood of lateral movement each step [0.0, 1.0].
        steps_active:        Number of environment steps this threat has existed.
        is_contained:        True once the defender has successfully neutralised it.
    """
    id: str
    stage: AttackStage
    origin_node: str
    current_node: str
    severity: float             # [0.0, 1.0]
    detection_confidence: float # [0.0, 1.0]
    is_detected: bool
    persistence: float          # [0.0, 1.0]
    spread_potential: float     # [0.0, 1.0]
    steps_active: int = 0
    is_contained: bool = False
    steps_at_current_stage: int = 0  # steps spent at the current kill-chain stage
    timestamp: float = field(default_factory=time.time)
    attack_type: str = "generic"   # e.g. "apt", "ransomware", "ddos", etc.
    mitre_technique_id: str = ""
    mitre_technique_name: str = ""
    mitre_tactic: str = ""
    mitre_tactic_id: str = ""

    @property
    def target_node(self) -> str:
        """Alias for current_node (test-compatibility)."""
        return self.current_node

    @property
    def severity_level(self) -> ThreatSeverity:
        """Return ThreatSeverity enum level for current severity score."""
        return ThreatSeverity.from_score(self.severity)

    def effective_severity(self) -> float:
        """
        Severity weighted by stage progression.
        Later stages are exponentially more dangerous.
        """
        stage_multiplier = 1.0 + (self.stage.value * 0.25)
        return min(1.0, self.severity * stage_multiplier)

    def clone(self) -> "Threat":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Resource pool
# ---------------------------------------------------------------------------

@dataclass
class ResourcePool:
    """
    Tracks the SOC's available operational resources for the current step.

    Resources represent analyst time, compute budget, and tooling capacity
    combined into a single abstract unit.  Each action consumes resources;
    running out forces the agent to IGNORE remaining threats.

    Attributes:
        total:     Maximum resources per episode step.
        remaining: Resources not yet spent this step.
    """
    total: float
    remaining: float

    @property
    def utilization(self) -> float:
        """Fraction of resources consumed this step."""
        if self.total == 0:
            return 1.0
        return 1.0 - (self.remaining / self.total)

    def can_afford(self, cost: float) -> bool:
        return self.remaining >= cost

    def consume(self, cost: float) -> bool:
        """Deduct cost. Returns False if insufficient resources."""
        if not self.can_afford(cost):
            return False
        self.remaining = max(0.0, self.remaining - cost)
        return True

    def reset_step(self) -> None:
        self.remaining = self.total

    def clone(self) -> "ResourcePool":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# Environment state (top-level observation)
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentState:
    """
    The complete, structured observation returned by env.state().

    This is the canonical object an agent receives.  All fields are typed
    and documented so an agent (or human analyst) can reason about them.

    Attributes:
        assets:               Map of asset_id → NetworkAsset.
        compromised_nodes:    IDs of assets with active adversary foothold.
        active_threats:       List of live Threat objects.
        threat_severity:      Aggregate severity across all active threats [0.0, 1.0].
        network_load:         Current network utilisation [0.0, 1.0].
                              High load can mask attack traffic.
        resource_availability: Fraction of SOC resources still available [0.0, 1.0].
        detection_confidence: Mean detection confidence across active threats [0.0, 1.0].
        time_step:            Current step index within the episode.
        episode_score:        Running score accumulated so far [0.0, 1.0].
        is_terminal:          True once episode end condition is met.
    """
    assets: Dict[str, NetworkAsset]
    compromised_nodes: List[str]
    active_threats: List[Threat]
    threat_severity: float          # [0.0, 1.0]
    network_load: float             # [0.0, 1.0]
    resource_availability: float    # [0.0, 1.0]
    detection_confidence: float     # [0.0, 1.0]
    time_step: int
    episode_score: float = 0.0
    is_terminal: bool = False

    # ---- derived helpers ------------------------------------------------

    def compromised_count(self) -> int:
        return len(self.compromised_nodes)

    def threat_count(self) -> int:
        return len(self.active_threats)

    def critical_assets_compromised(self) -> List[str]:
        """Assets with criticality > 0.7 that are currently compromised."""
        return [
            nid for nid in self.compromised_nodes
            if self.assets[nid].criticality > 0.7
        ]

    def to_vector(self) -> List[float]:
        """
        Flatten state to a numeric vector for ML agents.
        Order is stable across episodes with the same asset set.
        """
        vec: List[float] = []
        for asset_id in sorted(self.assets.keys()):
            a = self.assets[asset_id]
            vec.extend([
                float(a.is_compromised),
                float(a.is_isolated),
                a.health,
                a.patch_level,
                a.criticality,
                a.vulnerability_score(),
            ])
        vec.extend([
            self.threat_severity,
            self.network_load,
            self.resource_availability,
            self.detection_confidence,
            self.time_step / 100.0,   # normalised
            self.episode_score,
        ])
        return vec

    def clone(self) -> "EnvironmentState":
        return copy.deepcopy(self)
