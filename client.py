"""Adaptive Cyber Defense Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import Action, Observation


class AdaptiveCyberDefenseEnv(EnvClient[Action, Observation, State]):
    """
    Client for the Adaptive Cyber Defense Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with AdaptiveCyberDefenseEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     result = client.step(Action(action="scan_node_1"))
        ...     print(result.observation.visible_threats)

    Example with Docker:
        >>> client = AdaptiveCyberDefenseEnv.from_docker_image("cyber-defense:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(Action(action="block_ip"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: Action) -> Dict:
        """Convert Action to JSON payload for step message."""
        return {
            "action": action.action,
            "target_node": action.target_node,
        }

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        """Parse server response into StepResult[Observation]."""
        obs_data = payload.get("observation", payload)

        visible_threats = obs_data.get("visible_threats", [])
        network_state   = obs_data.get("network_state", {})

        observation = Observation(
            visible_threats=visible_threats,
            hidden_threat_count=obs_data.get("hidden_threat_count", 0),
            scan_coverage=obs_data.get("scan_coverage", 0.0),
            system_health=obs_data.get("system_health", 100),
            step=obs_data.get("step", 0),
            score=obs_data.get("score", 0.0),
            done=payload.get("done", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step", 0),
        )
