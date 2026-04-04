"""
AdaptiveCyberDefenseEnv — OpenEnv-compliant environment.

API contract:
    env.reset(seed=None)  → EnvironmentState
    env.step(action)      → (EnvironmentState, float, bool, dict)
    env.state()           → EnvironmentState

All randomness is routed through self._rng (seeded in reset) to guarantee
that identical seeds produce identical episode trajectories.
"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

from .engines.attack import AttackEngine, AttackEngineConfig, LateralMovementEvent
from .engines.detection import DetectionSystem, DetectionConfig, DetectionEvent
from .models.threat import get_mitre_info
from .engines.scoring import ThreatScorer, ThreatScore
from .engines.decision import (
    ActionMemory, DecisionEngine, DecisionConfig,
    ResourcePrioritiser, SpendingPlan, ActionRecommendation,
)
from .engines.response import ResponseEngine, ResponseConfig, ActionResult, StateUpdater
from .engines.reward import RewardFunction, RewardWeights, RewardBreakdown
from .models.action import Action, ActionInput, ACTION_PROFILES
from .models.network import NetworkGraph
from .models.state import (
    AttackStage,
    EnvironmentState,
    NetworkAsset,
    ResourcePool,
    Threat,
)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

class EnvConfig:
    """
    Tunable parameters that control episode difficulty.
    Override via subclassing or pass a dict to the constructor.
    """
    max_steps: int = 50
    resource_per_step: float = 1.0      # total SOC resource budget per step
    initial_threat_count: int = 1
    attack_progression_prob: float = 0.25   # chance threat advances one stage per step
    lateral_spread_base_prob: float = 0.20  # base chance of lateral movement per step
    natural_severity_growth: float = 0.03   # severity increase per step if ignored
    health_degradation_rate: float = 0.05   # health lost per step on compromised node
    false_positive_rate: float = 0.10       # fraction of clean nodes falsely flagged
    false_negative_rate: float = 0.15       # fraction of real threats missed by detection


# ---------------------------------------------------------------------------
# Main environment
# ---------------------------------------------------------------------------

class AdaptiveCyberDefenseEnv:
    """
    Adaptive Cyber Defense Simulator.

    Simulates a Security Operations Center defending an 8-node corporate
    network against evolving multi-stage cyber attacks.

    Usage::

        env = AdaptiveCyberDefenseEnv()
        state = env.reset(seed=42)

        done = False
        while not done:
            action = agent.choose(state)
            state, reward, done, info = env.step(action)

        print(f"Episode score: {env.state().episode_score:.3f}")
    """

    # OpenEnv metadata
    metadata: Dict[str, Any] = {
        "name": "AdaptiveCyberDefense-v1",
        "version": "1.0.0",
        "action_space_size": len(Action),
        "max_episode_steps": EnvConfig.max_steps,
    }

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()

        # Attack engine (Phase 2)
        self._attack_config = AttackEngineConfig(
            stage_progression_base_prob=self.config.attack_progression_prob,
            lateral_movement_base_prob=self.config.lateral_spread_base_prob,
            natural_severity_growth=self.config.natural_severity_growth,
        )
        self._attack_overrides: dict = {}
        self._attack_engine = AttackEngine(self._attack_config)

        # Detection system (Phase 3)
        self._detection_system = DetectionSystem(DetectionConfig(
            false_positive_rate=self.config.false_positive_rate,
        ))

        # Threat scorer (Phase 3)
        self._threat_scorer = ThreatScorer()

        # Decision engine + action memory + resource prioritiser (Phase 4)
        self._action_memory = ActionMemory()
        self._decision_engine = DecisionEngine()
        self._resource_prioritiser = ResourcePrioritiser()
        self._last_recommendations: List[ActionRecommendation] = []

        # Response engine + state updater (Phase 5)
        self._response_engine = ResponseEngine(
            config=ResponseConfig(),
            detection_system=self._detection_system,
        )
        self._state_updater = StateUpdater(
            health_degradation_rate=self.config.health_degradation_rate,
        )

        # Reward function (Phase 6)
        self._reward_function = RewardFunction()
        self._last_reward_breakdown: Optional[RewardBreakdown] = None

        # Last step's detection events and threat scores (exposed in step info)
        self._last_detection_events: List[DetectionEvent] = []
        self._last_threat_scores: List[ThreatScore] = []

        # Runtime state — populated by reset()
        self._rng: random.Random = random.Random()
        self._network: Optional[NetworkGraph] = None
        self._resource_pool: Optional[ResourcePool] = None
        self._current_state: Optional[EnvironmentState] = None
        self._step_count: int = 0
        self._episode_score: float = 0.0
        self._seed: Optional[int] = None

        # Episode history for adaptive decision support (Phase 6)
        self._action_history: List[Dict[str, Any]] = []
        self._lateral_events: List[LateralMovementEvent] = []

    # -----------------------------------------------------------------------
    # Adaptive attacker hook
    # -----------------------------------------------------------------------

    def set_attack_overrides(self, overrides: dict) -> None:
        """
        Apply adaptive attacker config overrides to the attack engine.
        Call before reset() to take effect for the next episode.
        """
        self._attack_overrides = overrides or {}
        self._attack_engine = AttackEngine(self._attack_config, overrides=self._attack_overrides)

    # -----------------------------------------------------------------------
    # OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> EnvironmentState:
        """
        Reset the environment to an initial state.

        Args:
            seed: Integer seed for deterministic episodes.
                  If None, a random seed is chosen.

        Returns:
            The initial EnvironmentState observation.
        """
        self._seed = seed if seed is not None else random.randint(0, 2**32 - 1)
        self._rng = random.Random(self._seed)

        # Build network topology with seeded randomness
        self._network = NetworkGraph.build_default(self._rng)

        # Initialise resource pool
        self._resource_pool = ResourcePool(
            total=self.config.resource_per_step,
            remaining=self.config.resource_per_step,
        )

        # Seed initial threats
        initial_threats = self._spawn_initial_threats()

        # Mark initially compromised nodes
        compromised = list({t.current_node for t in initial_threats})
        for node_id in compromised:
            self._network.assets[node_id].is_compromised = True

        # Build the first state snapshot
        self._step_count = 0
        self._episode_score = 0.0
        self._action_history = []
        self._lateral_events = []
        self._last_detection_events = []
        self._last_threat_scores = []
        self._last_recommendations = []
        self._attack_engine.reset_counter(base=len(initial_threats))
        self._detection_system.reset()
        self._action_memory.reset()

        self._current_state = self._build_state(initial_threats)
        return self._current_state.clone()

    def step(
        self,
        action: ActionInput,
    ) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Simulation loop per step:
            1. Reset resource pool for this step
            2. Validate action input
            3. Evolve active threats (attack engine placeholder — Phase 2)
            4. Run probabilistic detection (placeholder — Phase 3)
            5. Apply defender action (response engine placeholder — Phase 5)
            6. Update asset health and compromise status
            7. Compute reward (placeholder — Phase 6)
            8. Build new state snapshot
            9. Log step

        Args:
            action: ActionInput describing what the agent wants to do.

        Returns:
            (next_state, reward, done, info)
            - next_state: Updated EnvironmentState.
            - reward:     Float in [0.0, 1.0].
            - done:       True if episode has ended.
            - info:       Diagnostic dict (not used for learning).
        """
        if self._current_state is None:
            raise RuntimeError("Call reset() before step().")

        # -- 1. Reset per-step resource budget --------------------------------
        self._resource_pool.reset_step()

        # -- 2. Validate action -----------------------------------------------
        is_valid, reason = action.validate()
        if not is_valid:
            # Invalid actions are treated as IGNORE + small penalty
            action = ActionInput(action=Action.IGNORE)
            info_flag = f"Invalid action: {reason}"
        else:
            info_flag = "ok"

        if action.target_node and action.target_node not in self._network.assets:
            action = ActionInput(action=Action.IGNORE)
            info_flag = f"Unknown target node: {action.target_node}"

        # -- 3. Evolve threats via AttackEngine (Phase 2) ---------------------
        threats_after_evolution, lateral_events = self._evolve_threats(
            self._current_state.active_threats
        )
        self._lateral_events = lateral_events

        # -- 4. Detection pass (Phase 3) --------------------------------------
        threats_after_detection, detection_events = self._run_detection(
            threats_after_evolution
        )
        self._last_detection_events = detection_events

        # -- 5. Apply action via ResponseEngine (Phase 5) --------------------
        threats_after_response, action_result = self._apply_action(
            action, threats_after_detection
        )
        action_outcome = action_result.to_dict()

        # -- 6. Update asset health and compromise flags ----------------------
        self._update_assets(threats_after_response)

        # -- 6b. Score threats (Phase 3) ----------------------------------------
        self._last_threat_scores = self._threat_scorer.score_all(
            threats_after_response, self._network
        )

        # -- 6c. Record action outcome in memory (Phase 4) --------------------
        if action.action != Action.IGNORE and action.target_node:
            threat_score_before = next(
                (s.composite_score for s in self._last_threat_scores
                 if s.node_id == action.target_node),
                0.0,
            )
            target_threat = next(
                (t for t in threats_after_detection
                 if t.current_node == action.target_node and not t.is_contained),
                None,
            )
            self._action_memory.record(
                action=action.action,
                node_id=action.target_node,
                threat_stage=target_threat.stage if target_threat else AttackStage.PHISHING,
                success=action_result.success and not action_result.wasted,
                resource_cost=action_result.cost_paid,
                threat_score_before=threat_score_before,
                step=self._step_count,
            )

        # -- 6d. Generate decision recommendations (Phase 4) ------------------
        self._last_recommendations = self._decision_engine.recommend(
            state=self._build_state(threats_after_response),
            threat_scores=self._last_threat_scores,
            resource_pool=self._resource_pool,
            memory=self._action_memory,
        )

        # -- 7. Compute reward via RewardFunction (Phase 6) ------------------
        reward, reward_breakdown = self._reward_function.compute(
            state_before=self._current_state,
            state_after=self._build_state(threats_after_response),
            action_result=action_result,
            threat_scores_before=self._last_threat_scores,
            lateral_events=self._lateral_events,
            detection_events=self._last_detection_events,
            resource_pool=self._resource_pool,
            network=self._network,
        )
        self._last_reward_breakdown = reward_breakdown

        self._episode_score = min(1.0, self._episode_score + reward / self.config.max_steps)
        self._step_count += 1

        # -- 8. Check terminal conditions ------------------------------------
        done = self._is_terminal(threats_after_response)

        # -- 9. Build new state snapshot -------------------------------------
        self._current_state = self._build_state(threats_after_response)
        self._current_state.episode_score = self._episode_score
        self._current_state.is_terminal = done

        # -- 10. Log ---------------------------------------------------------
        step_log: Dict[str, Any] = {
            "step": self._step_count,
            "action": action.action.name,
            "target": action.target_node,
            "reward": round(reward, 4),
            "done": done,
            "threats_active": len([t for t in threats_after_response if not t.is_contained]),
            "compromised_nodes": list(self._current_state.compromised_nodes),
            "info": info_flag,
            "action_outcome": action_outcome,
            "lateral_movements": [
                {"from": e.source_node, "to": e.target_node, "parent": e.parent_threat_id}
                for e in lateral_events
            ],
            "detection_events": [
                {
                    "threat_id": e.threat_id,
                    "node": e.node_id,
                    "type": (
                        "true_positive" if e.is_true_positive
                        else "false_positive" if e.is_false_positive
                        else "false_negative"
                    ),
                    "method": e.detection_method,
                    "confidence": round(e.updated_confidence, 3),
                }
                for e in detection_events
            ],
            "threat_scores": [
                {
                    "id": s.threat_id,
                    "node": s.node_id,
                    "composite": s.composite_score,
                    "driver": s.primary_driver,
                }
                for s in self._last_threat_scores
            ],
            "recommendations": [
                {
                    "action": r.action_input.action.name,
                    "target": r.action_input.target_node,
                    "score": r.score,
                    "reasoning": r.reasoning,
                    "affordable": r.is_affordable,
                }
                for r in self._last_recommendations[:3]   # top 3 only in log
            ],
            "resource_utilisation": round(self._resource_pool.utilization, 4),
            "reward_breakdown": reward_breakdown.to_dict(),
            "timestamp": time.time(),
        }
        self._action_history.append(step_log)

        return self._current_state.clone(), reward, done, step_log

    def state(self) -> EnvironmentState:
        """
        Return the current environment state without advancing the simulation.

        Returns:
            A deep copy of the current EnvironmentState.

        Raises:
            RuntimeError: If reset() has not been called.
        """
        if self._current_state is None:
            raise RuntimeError("Call reset() before state().")
        return self._current_state.clone()

    # -----------------------------------------------------------------------
    # Initialisation helpers
    # -----------------------------------------------------------------------

    def _spawn_initial_threats(self) -> List[Threat]:
        """
        Create the initial set of Threat objects for the episode.
        Entry points are chosen from workstation-class nodes (phishing targets).
        """
        workstations = [
            nid for nid, asset in self._network.assets.items()
            if asset.asset_type.value == "workstation"
        ]
        entry_nodes = self._rng.sample(
            workstations,
            k=min(self.config.initial_threat_count, len(workstations)),
        )

        mitre = get_mitre_info("PHISHING")
        threats = []
        for i, node_id in enumerate(entry_nodes):
            threat = Threat(
                id=f"threat-{i:03d}",
                stage=AttackStage.PHISHING,
                origin_node=node_id,
                current_node=node_id,
                severity=self._rng.uniform(0.2, 0.5),
                detection_confidence=self._rng.uniform(0.1, 0.4),
                is_detected=False,
                persistence=self._rng.uniform(0.1, 0.4),
                spread_potential=self._rng.uniform(0.2, 0.5),
                steps_active=0,
                mitre_technique_id=mitre["technique_id"],
                mitre_technique_name=mitre["technique_name"],
                mitre_tactic=mitre["tactic"],
                mitre_tactic_id=mitre["tactic_id"],
            )
            threats.append(threat)
        return threats

    # -----------------------------------------------------------------------
    # Simulation step helpers
    # -----------------------------------------------------------------------

    def _evolve_threats(
        self, threats: List[Threat]
    ) -> Tuple[List[Threat], List[LateralMovementEvent]]:
        """
        Delegate to AttackEngine for full kill-chain progression and
        lateral movement.  Returns (evolved_threats, lateral_events).
        """
        return self._attack_engine.evolve(threats, self._network, self._rng)

    def _run_detection(
        self, threats: List[Threat]
    ) -> Tuple[List[Threat], List[DetectionEvent]]:
        """Delegate to DetectionSystem for probabilistic detection (Phase 3)."""
        return self._detection_system.run(
            threats,
            self._network,
            self._rng,
            self._compute_network_load(),
        )

    def _apply_action(
        self,
        action: ActionInput,
        threats: List[Threat],
    ) -> Tuple[List[Threat], "ActionResult"]:
        """Delegate to ResponseEngine for full action application (Phase 5)."""
        return self._response_engine.apply(
            action, threats, self._network, self._resource_pool, self._rng
        )

    def _update_assets(self, threats: List[Threat]) -> None:
        """Delegate to StateUpdater for full asset state update (Phase 5)."""
        self._state_updater.update(self._network, threats, self._lateral_events)

    # -----------------------------------------------------------------------
    # State builder
    # -----------------------------------------------------------------------

    def _build_state(self, threats: List[Threat]) -> EnvironmentState:
        active = [t for t in threats if not t.is_contained]
        compromised = [
            nid for nid, asset in self._network.assets.items()
            if asset.is_compromised
        ]

        threat_severity = (
            sum(t.effective_severity() for t in active) / len(active)
            if active else 0.0
        )
        detection_confidence = (
            sum(t.detection_confidence for t in active) / len(active)
            if active else 0.0
        )

        return EnvironmentState(
            assets=dict(self._network.assets),   # shared ref — clone on return
            compromised_nodes=compromised,
            active_threats=active,
            threat_severity=round(min(1.0, threat_severity), 4),
            network_load=self._compute_network_load(),
            resource_availability=round(
                self._resource_pool.remaining / self._resource_pool.total, 4
            ),
            detection_confidence=round(min(1.0, detection_confidence), 4),
            time_step=self._step_count,
            episode_score=self._episode_score,
            is_terminal=False,
        )

    # -----------------------------------------------------------------------
    # Terminal condition
    # -----------------------------------------------------------------------

    def _is_terminal(self, threats: List[Threat]) -> bool:
        """
        Episode ends when:
        - All threats are contained, OR
        - A critical asset's health drops to 0, OR
        - Max steps reached.
        """
        if self._step_count >= self.config.max_steps:
            return True

        active = [t for t in threats if not t.is_contained]
        if not active:
            return True   # clean sweep

        # Critical asset failure
        for asset in self._network.assets.values():
            if asset.criticality >= 0.9 and asset.health <= 0.0:
                return True

        return False

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def _compute_network_load(self) -> float:
        """Delegate to StateUpdater for criticality-weighted network load."""
        return self._state_updater.network_load(self._network)

    def threat_scores(self) -> List[ThreatScore]:
        """Return the threat scores computed during the last step."""
        return list(self._last_threat_scores)

    def detection_events(self) -> List[DetectionEvent]:
        """Return the detection events from the last step."""
        return list(self._last_detection_events)

    def recommend(self) -> List[ActionRecommendation]:
        """
        Return decision recommendations for the current state.
        Regenerates fresh recommendations on each call (resource pool may
        have changed since the last step).
        """
        if self._current_state is None:
            raise RuntimeError("Call reset() before recommend().")
        return self._decision_engine.recommend(
            state=self._current_state,
            threat_scores=self._last_threat_scores,
            resource_pool=self._resource_pool,
            memory=self._action_memory,
        )

    def spending_plan(
        self, recommendations: Optional[List[ActionRecommendation]] = None
    ) -> SpendingPlan:
        """
        Build a resource spending plan from recommendations.
        Uses last step's recommendations if none provided.
        """
        recs = recommendations or self._last_recommendations
        return self._resource_prioritiser.plan(recs, self._resource_pool)

    def action_memory(self) -> ActionMemory:
        """Return the episode action memory (read-only reference)."""
        return self._action_memory

    def reward_breakdown(self) -> Optional[RewardBreakdown]:
        """Return the RewardBreakdown from the last step, or None."""
        return self._last_reward_breakdown

    def set_reward_weights(self, weights: RewardWeights) -> None:
        """Override reward weights (used by task builders for difficulty tuning)."""
        self._reward_function = RewardFunction(weights)

    def seed(self) -> Optional[int]:
        """Return the seed used for the current episode."""
        return self._seed

    def action_history(self) -> List[Dict[str, Any]]:
        """Return a copy of the step log for the current episode."""
        return list(self._action_history)

    def __repr__(self) -> str:
        step = self._step_count if self._current_state else "not started"
        return (
            f"AdaptiveCyberDefenseEnv("
            f"seed={self._seed}, step={step}, "
            f"max_steps={self.config.max_steps})"
        )
