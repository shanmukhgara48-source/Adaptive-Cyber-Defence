"""
IgnoreAgent — always takes the IGNORE action.
Used as a baseline lower bound in comparisons.
"""

from ..models.action import Action, ActionInput
from ..models.state import EnvironmentState


class IgnoreAgent:
    """Agent that always ignores all threats."""

    def choose(self, state: EnvironmentState, recommendations=None) -> ActionInput:
        return ActionInput(action=Action.IGNORE)
