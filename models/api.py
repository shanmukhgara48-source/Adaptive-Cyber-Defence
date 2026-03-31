"""
Pydantic v2 API models for FastAPI request/response typing.

These are the wire-format schemas exposed by the HTTP API.
They are separate from the internal simulation dataclasses in
models/state.py and models/action.py.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, field_validator

MAX_ACTION_LEN = 64


class Observation(BaseModel):
    """Response schema for /reset and /state."""
    visible_threats:   list[dict[str, Any]] = []
    hidden_node_count: int   = 0
    scan_coverage:     float = 0.0
    system_health:     int   = 100
    score:             float = 0.0
    step:              int   = 0
    done:              bool  = False


class Action(BaseModel):
    """Request body for /step."""
    action: str

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v: Any) -> str:
        if not isinstance(v, str):
            v = str(v)
        if len(v) > MAX_ACTION_LEN:
            v = v[:MAX_ACTION_LEN]
        return v


class Reward(BaseModel):
    """Reward-related fields returned by /step."""
    action:     str
    reward:     float
    reason:     str   = ""
    confidence: float = 0.0
    done:       bool  = False
    error:      str | None = None
