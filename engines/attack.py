"""
Multi-Stage Attack Engine for the Adaptive Cyber Defense Simulator.

Implements the full cyber kill-chain:
    PHISHING → CREDENTIAL_ACCESS → MALWARE_INSTALL → LATERAL_SPREAD → EXFILTRATION

Each step the engine:
    1. Tries to advance each threat's kill-chain stage
    2. Degrades the compromised asset's health
    3. Increases threat persistence (entrenchment)
    4. Attempts lateral movement when stage == LATERAL_SPREAD
    5. Spawns child threats on newly infected nodes
    6. Recomputes severity based on stage + asset criticality

Design principles:
    - All randomness routed through the caller's rng instance (determinism)
    - Network graph is read-only here; mutations go back through EnvironmentState
    - New threats spawned by lateral movement start at PHISHING on the new node
      (attacker re-establishes foothold before advancing)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..models.network import NetworkGraph
from ..models.state import AttackStage, NetworkAsset, Threat


# ---------------------------------------------------------------------------
# Engine configuration (separate from EnvConfig for testability)
# ---------------------------------------------------------------------------

@dataclass
class AttackEngineConfig:
    """
    Tunable knobs for attack behaviour.  Defaults match medium difficulty.
    """
    # Probability per step that a threat advances one kill-chain stage
    stage_progression_base_prob: float = 0.25

    # Vulnerability amplifier: fully unpatched node increases progression prob.
    # Reduced from 2.0 — the old value let vuln=0.7 nodes multiply base prob
    # by 1.7×, making PHISHING→EXFILTRATION possible in ~5 steps on hard.
    vulnerability_stage_multiplier: float = 1.5

    # Minimum steps a threat must spend in a stage before it can advance.
    # Enforces a realistic "dwell time" — an attacker can't skip from PHISHING
    # to CREDENTIAL_ACCESS in one step even on a fully unpatched node.
    min_stage_dwell: int = 2

    # Persistence growth per step (how quickly attacker digs in)
    persistence_growth_rate: float = 0.04

    # Max persistence cap — prevents instant containment being impossible
    persistence_cap: float = 0.90

    # Base probability per step of lateral movement (only at LATERAL_SPREAD)
    lateral_movement_base_prob: float = 0.20

    # Spread-potential amplifier: high spread_potential raises movement chance
    spread_amplifier: float = 1.5

    # Severity growth per step if no defender action taken
    natural_severity_growth: float = 0.03

    # Stage-based severity multiplier (exfiltration is 2× phishing baseline)
    stage_severity_weights: dict = None   # set in __post_init__

    def __post_init__(self):
        if self.stage_severity_weights is None:
            self.stage_severity_weights = {
                AttackStage.PHISHING:          1.00,
                AttackStage.CREDENTIAL_ACCESS: 1.25,
                AttackStage.MALWARE_INSTALL:   1.50,
                AttackStage.LATERAL_SPREAD:    1.75,
                AttackStage.EXFILTRATION:      2.00,
            }


# ---------------------------------------------------------------------------
# Lateral movement event (returned alongside updated threats)
# ---------------------------------------------------------------------------

@dataclass
class LateralMovementEvent:
    """
    Records a successful lateral movement for logging and state updates.

    Attributes:
        parent_threat_id: The threat that performed the movement.
        source_node:      Node the attacker moved FROM.
        target_node:      Node the attacker moved TO.
        child_threat:     The new Threat spawned on the target node.
    """
    parent_threat_id: str
    source_node: str
    target_node: str
    child_threat: Threat


# ---------------------------------------------------------------------------
# Attack engine
# ---------------------------------------------------------------------------

class AttackEngine:
    """
    Stateless engine — all state is passed in and returned.

    Usage::

        engine = AttackEngine(config)
        updated_threats, events = engine.evolve(threats, network, rng)

    The engine never modifies its arguments in-place; it always returns
    new/cloned objects.
    """

    def __init__(self, config: Optional[AttackEngineConfig] = None) -> None:
        self.config = config or AttackEngineConfig()
        self._threat_counter: int = 0   # used to generate unique child IDs

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def evolve(
        self,
        threats: List[Threat],
        network: NetworkGraph,
        rng: random.Random,
    ) -> Tuple[List[Threat], List[LateralMovementEvent]]:
        """
        Advance all active threats by one simulation step.

        Args:
            threats: Current list of Threat objects.
            network: Read-only network graph.
            rng:     Seeded random instance from the environment.

        Returns:
            (updated_threats, lateral_events)
            - updated_threats includes both evolved originals AND any new
              child threats spawned by lateral movement.
            - lateral_events is a log of movements for the environment to
              consume (marks new nodes as compromised, etc.)
        """
        updated: List[Threat] = []
        lateral_events: List[LateralMovementEvent] = []
        # Track nodes already targeted this step to avoid duplicate spawns
        spawned_on: set[str] = {t.current_node for t in threats if not t.is_contained}

        for threat in threats:
            if threat.is_contained:
                updated.append(threat.clone())
                continue

            clone = threat.clone()
            asset = network.assets.get(clone.current_node)

            # 1. Increment age and stage dwell counter
            clone.steps_active += 1
            clone.steps_at_current_stage += 1

            # 2. Grow persistence (attacker digs deeper over time)
            clone.persistence = min(
                self.config.persistence_cap,
                clone.persistence + self.config.persistence_growth_rate,
            )

            # 3. Try to advance kill-chain stage
            clone = self._try_stage_progression(clone, asset, rng)

            # 4. Grow severity based on current stage + asset criticality
            clone.severity = self._recompute_severity(clone, asset)

            # 5. Attempt lateral movement (only at LATERAL_SPREAD stage)
            if clone.stage == AttackStage.LATERAL_SPREAD:
                event = self._try_lateral_movement(clone, network, rng, spawned_on)
                if event is not None:
                    lateral_events.append(event)
                    updated.append(event.child_threat)
                    spawned_on.add(event.target_node)

            updated.append(clone)

        return updated, lateral_events

    # -----------------------------------------------------------------------
    # Stage progression
    # -----------------------------------------------------------------------

    def _try_stage_progression(
        self,
        threat: Threat,
        asset: Optional[NetworkAsset],
        rng: random.Random,
    ) -> Threat:
        """
        Probabilistically advance the threat one kill-chain stage.

        Probability is modulated by:
        - Asset vulnerability (unpatched nodes speed up progression)
        - Threat persistence (deeply embedded attackers move faster)
        - Current stage (exfiltration cannot advance further)
        """
        next_stage = threat.stage.next_stage()
        if next_stage is None:
            return threat   # already at final stage

        # Minimum dwell check: threat must spend at least min_stage_dwell steps
        # at the current stage before it can attempt to advance.  This prevents
        # a single-step PHISHING→EXFILTRATION run even at high base probability.
        if threat.steps_at_current_stage < self.config.min_stage_dwell:
            return threat

        vulnerability = asset.vulnerability_score() if asset else 0.5
        persistence_factor = 1.0 + (threat.persistence * 0.5)

        prob = (
            self.config.stage_progression_base_prob
            * (1.0 + vulnerability * (self.config.vulnerability_stage_multiplier - 1.0))
            * persistence_factor
        )
        prob = min(0.95, prob)   # cap at 95% — advancement is never guaranteed

        if rng.random() < prob:
            clone = threat.clone()
            clone.stage = next_stage
            clone.steps_at_current_stage = 0   # reset dwell counter on transition
            return clone

        return threat

    # -----------------------------------------------------------------------
    # Severity computation
    # -----------------------------------------------------------------------

    def _recompute_severity(
        self,
        threat: Threat,
        asset: Optional[NetworkAsset],
    ) -> float:
        """
        Severity = base growth × stage weight × asset criticality modifier.

        Formula:
            new_severity = min(1.0,
                (old_severity + natural_growth)
                × stage_weight
                × (1 + criticality_bonus)
            )

        Critical assets (criticality > 0.8) attract disproportionate severity.
        """
        criticality = asset.criticality if asset else 0.5
        stage_weight = self.config.stage_severity_weights[threat.stage]

        # Natural growth
        grown = threat.severity + self.config.natural_severity_growth

        # Stage and criticality amplification
        amplified = grown * stage_weight * (1.0 + criticality * 0.3)

        return min(1.0, amplified)

    # -----------------------------------------------------------------------
    # Lateral movement
    # -----------------------------------------------------------------------

    def _try_lateral_movement(
        self,
        threat: Threat,
        network: NetworkGraph,
        rng: random.Random,
        already_spawned: set[str],
    ) -> Optional[LateralMovementEvent]:
        """
        Attempt to spread the threat to an adjacent, uncompromised node.

        Movement probability:
            p = lateral_movement_base_prob × spread_potential × spread_amplifier

        Target selection: most vulnerable uncompromised neighbour.

        If movement succeeds, a child Threat is created at PHISHING stage
        on the target node (the attacker re-establishes a foothold before
        advancing — mirrors real-world lateral movement behaviour).
        """
        move_prob = min(
            0.90,
            self.config.lateral_movement_base_prob
            * threat.spread_potential
            * self.config.spread_amplifier,
        )

        if rng.random() >= move_prob:
            return None

        # Find eligible targets: active, not already compromised/spawned
        candidates = [
            nb for nb in network.active_neighbours(threat.current_node)
            if nb not in already_spawned
            and not network.assets[nb].is_compromised
        ]

        if not candidates:
            return None

        # Pick most vulnerable candidate (with stochastic tie-break)
        candidates.sort(
            key=lambda nid: (
                -network.assets[nid].vulnerability_score(),
                nid,
            )
        )
        if rng.random() < 0.80:
            target_node = candidates[0]
        else:
            target_node = rng.choice(candidates)

        # Spawn child threat at PHISHING stage on target
        self._threat_counter += 1
        child = Threat(
            id=f"threat-{self._threat_counter:03d}",
            stage=AttackStage.PHISHING,       # attacker re-establishes foothold
            origin_node=threat.current_node,   # tracks where it came from
            current_node=target_node,
            severity=threat.severity * 0.6,    # starts weaker on new node
            detection_confidence=0.05,         # initially very stealthy
            is_detected=False,
            persistence=0.05,                  # not yet entrenched
            spread_potential=threat.spread_potential * rng.uniform(0.7, 1.0),
            steps_active=0,
        )

        return LateralMovementEvent(
            parent_threat_id=threat.id,
            source_node=threat.current_node,
            target_node=target_node,
            child_threat=child,
        )

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def reset_counter(self, base: int = 100) -> None:
        """
        Reset internal threat ID counter.
        Call from env.reset() to keep IDs predictable across episodes.
        """
        self._threat_counter = base
