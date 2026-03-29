"""
Baseline greedy agent for the Adaptive Cyber Defense Simulator.

Strategy
--------
Each step the agent uses the environment's built-in DecisionEngine to get
action recommendations, then applies the following priority order:

1. If the highest-priority recommendation is affordable, take it.
2. Otherwise fall back to the cheapest affordable action that targets a
   compromised node.
3. If no affordable action exists, IGNORE.

The agent is stateless — it makes decisions purely from the current
EnvironmentState plus the recommendation list injected via the env
public API.  It does NOT need to hold a reference to the environment;
instead it inspects `state.active_threats` and `state.compromised_nodes`
and accepts an optional `recommendations` list from the caller.

Usage (standalone)::

    from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
    from adaptive_cyber_defense.agents.baseline import BaselineAgent

    env = AdaptiveCyberDefenseEnv()
    agent = BaselineAgent()

    state = env.reset(seed=42)
    done = False
    while not done:
        recs = env.recommend()
        action = agent.choose(state, recommendations=recs)
        state, reward, done, info = env.step(action)
"""

from __future__ import annotations

from typing import List, Optional

from ..models.action import Action, ActionInput
from ..models.state import AttackStage, EnvironmentState


class BaselineAgent:
    """
    Rule-based greedy agent.

    Accepts an optional list of ActionRecommendations produced by the
    environment's DecisionEngine.  Falls back to a hard-coded heuristic
    when no recommendations are available.

    Parameters
    ----------
    prefer_isolation_threshold : float
        If a threat's effective severity exceeds this value AND the threat
        is at LATERAL_SPREAD or beyond, the agent biases toward ISOLATE_NODE
        even if a cheaper BLOCK_IP is also recommended.  Default 0.70.
    """

    def __init__(self, prefer_isolation_threshold: float = 0.70) -> None:
        self.prefer_isolation_threshold = prefer_isolation_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose(
        self,
        state: EnvironmentState,
        recommendations: Optional[List] = None,
    ) -> ActionInput:
        """
        Choose an action for the given state.

        Args:
            state:           Current environment state snapshot.
            recommendations: Optional list of ActionRecommendation objects
                             from env.recommend().  When provided the agent
                             delegates primary decision logic to them.

        Returns:
            ActionInput ready to pass to env.step().
        """
        if recommendations:
            action = self._choose_from_recommendations(state, recommendations)
            if action is not None:
                return action

        # Fall back to heuristic when no usable recommendation
        return self._heuristic(state)

    # ------------------------------------------------------------------
    # Recommendation-driven path
    # ------------------------------------------------------------------

    def _choose_from_recommendations(
        self,
        state: EnvironmentState,
        recommendations: list,
    ) -> Optional[ActionInput]:
        """
        Pick the best affordable recommendation.

        Applies the isolation-bias override: if the top recommended threat
        is late-stage and high severity, prefer ISOLATE_NODE even if the
        top recommendation is something cheaper.
        """
        affordable = [r for r in recommendations if r.is_affordable]
        if not affordable:
            return None

        # Sort by recommendation score descending (already sorted, but be safe)
        affordable.sort(key=lambda r: r.score, reverse=True)
        top = affordable[0]

        # Isolation bias override
        if (
            top.threat_score is not None
            and top.threat_score.composite_score >= self.prefer_isolation_threshold
            and top.action_input.action != Action.ISOLATE_NODE
        ):
            # Check if a threat is at lateral-spread or beyond on that node
            node = top.action_input.target_node
            for threat in state.active_threats:
                if (
                    threat.current_node == node
                    and threat.stage.value >= AttackStage.LATERAL_SPREAD.value
                    and threat.effective_severity() >= self.prefer_isolation_threshold
                    and not threat.is_contained
                ):
                    # Bias toward isolation if the node is not already isolated
                    asset = state.assets.get(node)
                    if asset and not asset.is_isolated:
                        return ActionInput(
                            action=Action.ISOLATE_NODE,
                            target_node=node,
                        )

        return top.action_input

    # ------------------------------------------------------------------
    # Fallback heuristic (no DecisionEngine output)
    # ------------------------------------------------------------------

    def _heuristic(self, state: EnvironmentState) -> ActionInput:
        """
        Simple rule-based fallback.

        Priority:
        1. ISOLATE the most critical compromised node at late stage.
        2. BLOCK_IP on any compromised node.
        3. RUN_DEEP_SCAN on a compromised node that has undetected threats.
        4. IGNORE.
        """
        if not state.active_threats:
            return ActionInput(action=Action.IGNORE)

        # resource_availability is the remaining budget fraction [0, 1].
        # Use it to gate expensive actions; ResponseEngine enforces exact costs.
        resources = state.resource_availability

        # Identify late-stage high-severity threats
        late_threats = sorted(
            [
                t for t in state.active_threats
                if t.stage.value >= AttackStage.LATERAL_SPREAD.value
                and not t.is_contained
            ],
            key=lambda t: t.effective_severity(),
            reverse=True,
        )

        if late_threats:
            target = late_threats[0].current_node
            asset = state.assets.get(target)
            # Prefer isolation for late-stage threats if not already isolated
            if asset and not asset.is_isolated and resources > 0.30:
                return ActionInput(action=Action.ISOLATE_NODE, target_node=target)
            # Otherwise block
            if resources > 0.10:
                return ActionInput(action=Action.BLOCK_IP, target_node=target)

        # Early-stage: block first detected threat
        early_threats = sorted(
            [t for t in state.active_threats if not t.is_contained],
            key=lambda t: t.effective_severity(),
            reverse=True,
        )
        if early_threats:
            target = early_threats[0].current_node
            if resources > 0.10:
                return ActionInput(action=Action.BLOCK_IP, target_node=target)
            if resources > 0.15:
                return ActionInput(action=Action.RUN_DEEP_SCAN, target_node=target)

        return ActionInput(action=Action.IGNORE)
