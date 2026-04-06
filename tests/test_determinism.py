"""
Determinism test suite for the Adaptive Cyber Defense HTTP API.

Verifies that identical (seed, action_sequence) pairs always produce
identical per-step rewards, regardless of session.

Tests specifically cover:
  - Basic scan sequence (exercises sess.rng in detection path)
  - Defense actions (exercises sess.rng in the resource-exhaustion branch)
  - Mixed long sequences across all task difficulties
  - Multiple independent seeds
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import app as _app

CLIENT = TestClient(_app.app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_episode(task: str, seed: int, actions: list[str]) -> list[float]:
    """Reset a fresh session and execute `actions`, returning per-step rewards."""
    data = CLIENT.post("/reset", json={"task": task, "seed": seed}).json()
    sid  = data["session_id"]
    rewards: list[float] = []
    for action in actions:
        resp = CLIENT.post("/step", json={"action": action, "session_id": sid}).json()
        rewards.append(resp["reward"])
        if resp.get("done"):
            break
    return rewards


def _assert_deterministic(task: str, seed: int, actions: list[str]) -> None:
    """Run two sessions with same seed+actions and assert reward sequences match."""
    r1 = _run_episode(task, seed, actions)
    r2 = _run_episode(task, seed, actions)
    assert r1 == r2, (
        f"Non-deterministic rewards for task={task!r} seed={seed}:\n"
        + "\n".join(
            f"  step {i+1:2d} {actions[i]:<22} r1={v1:+.4f} r2={v2:+.4f}  ← DIFFERS"
            for i, (v1, v2) in enumerate(zip(r1, r2)) if v1 != v2
        )
    )


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

SCAN_ONLY = ["scan_node_1", "scan_node_2", "scan_node_3", "scan_node_4", "scan_node_5"] * 2
MIXED = [
    "scan_node_1", "block_ip", "isolate_machine", "patch", "ignore",
    "scan_node_2", "block_ip", "isolate_machine", "patch", "ignore",
    "scan_node_3", "scan_node_4", "scan_node_5", "block_ip", "ignore",
]
# Exhausts budget quickly then hits the conditional rng.random() resource-exhaustion path
EXHAUST_BUDGET = ["isolate_machine"] * 8 + ["isolate_machine"] * 5 + ["block_ip", "patch", "ignore"]


class TestDeterminismScans:
    """Scan-only sequences exercise sess.rng in visibility / detection paths."""

    def test_scan_sequence_easy_seed42(self):
        _assert_deterministic("easy", 42, SCAN_ONLY)

    def test_scan_sequence_hard_seed42(self):
        _assert_deterministic("hard", 42, SCAN_ONLY)

    def test_scan_sequence_nightmare_seed42(self):
        _assert_deterministic("nightmare", 42, SCAN_ONLY)

    def test_scan_sequence_multiple_seeds(self):
        for seed in [0, 1, 99, 1337, 2**20, 2**31 - 1]:
            _assert_deterministic("hard", seed, SCAN_ONLY)


class TestDeterminismMixed:
    """Mixed sequences exercise both scan and defense rng paths."""

    def test_mixed_easy(self):
        _assert_deterministic("easy", 42, MIXED)

    def test_mixed_medium(self):
        _assert_deterministic("medium", 42, MIXED)

    def test_mixed_hard(self):
        _assert_deterministic("hard", 42, MIXED)

    def test_mixed_nightmare(self):
        _assert_deterministic("nightmare", 42, MIXED)

    def test_mixed_all_seeds(self):
        for seed in [42, 100, 9999, 123456]:
            _assert_deterministic("hard", seed, MIXED)


class TestDeterminismResourceExhaustion:
    """
    Specifically targets the resource-exhaustion conditional rng path.

    Before the fix, `sess.rng.random()` was only called when
    `_resource_exhausted` was True, making the RNG sequence branch-dependent.
    After the fix, the dice is pre-rolled unconditionally every defense step.
    """

    def test_exhaust_budget_hard(self):
        _assert_deterministic("hard", 42, EXHAUST_BUDGET)

    def test_exhaust_budget_easy(self):
        _assert_deterministic("easy", 42, EXHAUST_BUDGET)

    def test_exhaust_budget_multiple_seeds(self):
        for seed in [0, 7, 42, 999]:
            _assert_deterministic("hard", seed, EXHAUST_BUDGET)


class TestDeterminismFullEpisode:
    """Run full-length episodes and confirm reward sequences are identical."""

    def test_full_episode_easy(self):
        actions = (SCAN_ONLY + MIXED + EXHAUST_BUDGET)[:30]
        _assert_deterministic("easy", 42, actions)

    def test_full_episode_hard(self):
        actions = (MIXED + EXHAUST_BUDGET)[:30]
        _assert_deterministic("hard", 1337, actions)

    def test_different_seeds_differ(self):
        """Sanity: different seeds should (almost always) produce different outcomes."""
        r42   = _run_episode("hard", 42,   MIXED)
        r1337 = _run_episode("hard", 1337, MIXED)
        # Not asserting equality — just confirming seeds aren't silently ignored
        # (it's statistically near-impossible for these to be equal)
        assert r42 != r1337, "Seeds 42 and 1337 produced identical rewards — seeds may be ignored"
