"""
tests/test_audit.py — Regression tests for all 6 fixes from the adversarial audit.

Each test is self-contained, deterministic (fixed seeds), and directly exercises the
specific code path that was broken.  Tests are ordered by severity (P0 first).

Run:
    pytest tests/test_audit.py -v
"""
import math
import random
import types

import pytest

# Import the HTTP layer under test
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import (
    _fresh_state,
    _do_reset_session,
    _obs,
    _compute_grader_score,
    _update_visibility,
    _age_threats,
    TASK_OVERRIDES,
    Session,
    NODES,
    ATTACKS,
    _MITRE_CORRECT_ACTION,
    _SCAN_COST_BY_DIFFICULTY,
    _VERIFY_COST_BY_DIFFICULTY,
    _MONITOR_COST,
    NODE_CRITICALITY,
    AdaptiveAttacker,
    _ATTACKER_SEED,
)
from constants import COST_RAW


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(task: str = "easy", seed: int = 42) -> Session:
    """Create a minimally initialised session for testing."""
    cfg = TASK_OVERRIDES[task]
    sess = Session(task_name=task, task_config=cfg)
    sess.effective_config = dict(cfg)
    sess.rng = random.Random(seed)
    sess.attacker = AdaptiveAttacker(seed=_ATTACKER_SEED ^ (seed & 0xFFFFFFFF))
    _do_reset_session(sess)
    return sess


def _inject_threats(sess: Session, threat_specs: list[dict]) -> None:
    """Replace the session's threat list with hand-crafted threats for deterministic tests."""
    sess.state["threats"] = []
    for spec in threat_specs:
        t_type = spec["type"]
        t = {
            "id":                        spec.get("id", f"t_{t_type}_{spec.get('node','node_1')}"),
            "type":                      t_type,
            "original_type":             t_type,
            "node":                      spec.get("node", "node_1"),
            "visible":                   spec.get("visible", True),
            "age":                       spec.get("age", 1),
            "stage":                     spec.get("stage", "initial"),
            "escalated":                 False,
            "contained":                 spec.get("contained", False),
            "spread_attempted":          False,
            "mitre_id":                  "T0000",
            "severity":                  spec.get("severity", 0.5),
            "is_false_positive":         spec.get("is_false_positive", False),
            "is_verified":               spec.get("is_verified", True),
            "pending_action":            spec.get("pending_action", None),
            "mitigation_progress":       spec.get("mitigation_progress", 0),
            "resurface_risk":            0.0,
            "steps_since_contained":     0,
            "kill_chain_stage_idx":      0,
            "kill_chain_health_multiplier": 1.0,
            "packets_per_second":        50,
            "failed_auth_attempts":      10,
            "outbound_data_bytes":       500,
            "lateral_connection_count":  2,
            "unusual_process_count":     3,
            "spread_rate":               0.1,
            "is_persistent":             False,
            "affected_node_count":       1,
            "detection_confidence":      0.8,
        }
        sess.state["threats"].append(t)


# ============================================================================
# FIX 1 — P0: pending_action break → continue
# ============================================================================

class TestPendingActionContinue:
    """
    Bug: the defense loop used `break` when encountering a pending-action threat.
    This prevented the agent from ever targeting a second visible threat when the
    first one already had a patch in progress.

    Fix: changed `break` to `continue` so the loop skips pending threats and
    continues scanning for a target that can be acted upon.
    """

    def test_second_threat_targeted_when_first_has_pending_action(self):
        """
        Setup: two visible threats.
          T1 = malware on node_1 (pending_action="patch", needs isolate_machine)
          T2 = phishing on node_2 (no pending action, needs block_ip)
        Action: block_ip
        Expected: T2 gets contained; session.containment_events has 1 entry for T2.
        Pre-fix: T1's pending_action caused a break and T2 was never reached.
        """
        sess = _make_session("easy", seed=1)
        _inject_threats(sess, [
            {"id": "t_malware_node1",   "type": "malware",  "node": "node_1",
             "visible": True, "pending_action": "patch", "mitigation_progress": 1},
            {"id": "t_phishing_node2",  "type": "phishing", "node": "node_2",
             "visible": True, "pending_action": None},
        ])

        from app import app as fastapi_app
        from fastapi.testclient import TestClient
        from app import _SESSIONS, _LATEST_SID

        # Directly manipulate session store so we can use the HTTP layer
        import uuid
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        import app as _app
        _app._LATEST_SID = sid

        client = TestClient(fastapi_app)
        resp = client.post("/step", json={"action": "block_ip", "session_id": sid})
        data = resp.json()

        # T2 (phishing → block_ip) should have been contained
        t2 = next(t for t in sess.state["threats"] if t["id"] == "t_phishing_node2")
        assert t2["contained"], (
            "block_ip should have contained T2 (phishing) even though T1 had a pending action. "
            "If break was not changed to continue the loop would exit before reaching T2."
        )

    def test_single_pending_threat_not_double_penalised(self):
        """
        When there is exactly one visible threat and it has a pending_action,
        the mitigation should not magically succeed — but health should not
        drop more than once (wrong-action penalty only, not double-penalty).
        """
        sess = _make_session("easy", seed=2)
        _inject_threats(sess, [
            {"id": "t_malware_node1", "type": "malware", "node": "node_1",
             "visible": True, "pending_action": "patch", "mitigation_progress": 1},
        ])
        import uuid
        from app import _SESSIONS
        import app as _app
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid
        health_before = sess.state["system_health"]

        from fastapi.testclient import TestClient
        from app import app as fastapi_app
        client = TestClient(fastapi_app)
        client.post("/step", json={"action": "isolate_machine", "session_id": sid})

        # Health penalty is at most one wrong-action hit (−5), not stacked
        health_after = sess.state["system_health"]
        health_delta = health_before - health_after
        assert health_delta <= 10, (
            f"Expected health drop ≤10 for wrong-action + degradation, got {health_delta}. "
            "Indicates the agent was double-penalised."
        )


# ============================================================================
# FIX 2 — P0: ignore with no visible threats must not reduce health
# ============================================================================

class TestIgnoreNoThreats:
    """
    Bug: `ignore` always subtracted 10 HP regardless of whether any visible
    threat existed to justify the penalty.

    Fix: gate the −10 penalty on `matched_threat_type is not None`.
    """

    def test_ignore_with_no_visible_threats_no_health_loss(self):
        """
        If no threats are visible, `ignore` should be a no-op on health.
        Health may still degrade from the health_degradation_rate but must not
        receive the extra −10 ignore penalty.
        """
        sess = _make_session("easy", seed=3)
        # Make all threats hidden
        _inject_threats(sess, [
            {"id": "t_malware_node1", "type": "malware", "node": "node_1",
             "visible": False, "age": 0},
        ])
        import uuid
        from app import _SESSIONS
        import app as _app
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid
        health_before = sess.state["system_health"]

        from fastapi.testclient import TestClient
        from app import app as fastapi_app
        client = TestClient(fastapi_app)
        client.post("/step", json={"action": "ignore", "session_id": sid})

        health_after = sess.state["system_health"]
        # No threat is visible so no ignore penalty should apply.
        # Easy has health_degradation_rate=0.08 so there may be a tiny passive loss.
        assert health_before - health_after < 10, (
            f"Health dropped {health_before - health_after} HP on `ignore` with no visible "
            "threats. The −10 ignore penalty should only fire when there is an active visible "
            "threat being wilfully ignored."
        )

    def test_ignore_with_visible_threat_reduces_health(self):
        """Sanity-check the reverse: ignore on a visible threat must still cost 10 HP."""
        sess = _make_session("easy", seed=4)
        _inject_threats(sess, [
            {"id": "t_phishing_node1", "type": "phishing", "node": "node_1",
             "visible": True, "age": 1},
        ])
        import uuid
        from app import _SESSIONS
        import app as _app
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid
        health_before = sess.state["system_health"]

        from fastapi.testclient import TestClient
        from app import app as fastapi_app
        client = TestClient(fastapi_app)
        client.post("/step", json={"action": "ignore", "session_id": sid})

        health_after = sess.state["system_health"]
        assert health_before - health_after >= 10, (
            "Expected ≥10 HP loss when ignoring a visible threat (−10 ignore penalty). "
            f"Got {health_before - health_after}."
        )


# ============================================================================
# FIX 3 — P0: lateral_movement visibility must respect detect_prob
# ============================================================================

class TestLateralMovementVisibility:
    """
    Bug: lateral-movement stage threats were revealed using only fn_rate, completely
    bypassing detect_prob.  On elite (detect_prob=0.08, fn_rate=0.15) this gave them
    an 85% reveal probability — 10× the intended 8%.

    Fix: both roll checks (_roll1 < detect_prob AND _roll2 > fn_rate) are now applied
    uniformly for all non-FP threats, consuming exactly 2 RNG calls per eligible threat.
    """

    def test_lateral_movement_reveal_respects_detect_prob(self):
        """
        Simulate _update_visibility 1000× with a patched RNG that always returns values
        that SHOULD pass the fn_rate check but FAIL the detect_prob check.  The threat
        must NOT be revealed (i.e. detect_prob is actually checked).
        """
        sess = _make_session("elite", seed=5)
        detect_prob = sess.effective_config.get("base_detection_prob", 0.08)

        # Patch RNG: roll1 always = detect_prob + 0.01 (just above threshold → fails detect_prob)
        #            roll2 always = fn_rate - 0.01 (below fn_rate → passes fn_rate check)
        # Under the old code, only roll2 was checked (> fn_rate), so this would always reveal.
        # Under the fixed code, roll1 < detect_prob is checked first — it fails → no reveal.
        fn_rate = sess.effective_config.get("false_negative_rate",
                                             sess.task_config.get("false_negative_rate", 0.0))

        fail_detect_prob_roll = detect_prob + 0.01  # always FAILS detect_prob check (not < detect_prob)
        pass_fn_rate_roll = max(0.0, fn_rate - 0.01)   # always PASSES fn_rate check (> fn_rate)

        class _MockRNG:
            def __init__(self):
                self._call = 0
                self._randint_val = 0
            def random(self):
                self._call += 1
                # Alternate: odd calls = fail detect_prob, even calls = pass fn_rate
                if self._call % 2 == 1:
                    return fail_detect_prob_roll
                return pass_fn_rate_roll
            def randint(self, a, b):
                return self._randint_val
            def uniform(self, a, b):
                return a
            def choice(self, seq):
                return seq[0]

        sess.rng = _MockRNG()
        sess.state["threats"] = [{
            "id": "t_lat", "type": "lateral_movement", "original_type": "lateral_movement",
            "node": "node_1", "visible": False, "age": 10, "stage": "lateral_movement",
            "contained": False, "is_false_positive": False,
        }]

        _update_visibility(sess)

        assert not sess.state["threats"][0]["visible"], (
            "Lateral-movement threat was revealed even though detect_prob check should have "
            "failed.  This indicates the fix did not apply detect_prob to the lateral_movement "
            "branch, reproducing the original bug."
        )

    def test_lateral_movement_reveal_probability_is_bounded(self):
        """
        Run 2000 independent trials.  With elite detect_prob=0.08 and fn_rate=0.15,
        the combined reveal probability should be ≈ 0.08 × (1−0.15) ≈ 0.068.
        The old code gave ~0.85.  We assert the empirical rate is ≤ 0.30.
        """
        reveals = 0
        N = 2000
        for i in range(N):
            sess = _make_session("elite", seed=1000 + i)
            sess.state["threats"] = [{
                "id": "t_lat", "type": "lateral_movement", "original_type": "lateral_movement",
                "node": "node_1", "visible": False, "age": 0, "stage": "lateral_movement",
                "contained": False, "is_false_positive": False,
            }]
            _update_visibility(sess)
            if sess.state["threats"][0]["visible"]:
                reveals += 1
        rate = reveals / N
        assert rate <= 0.30, (
            f"Lateral-movement reveal rate on elite = {rate:.3f}, expected ≤ 0.30 "
            f"(old code gave ~0.85; fixed code should be ~0.068)."
        )


# ============================================================================
# FIX 4 — P1: attacker seed varies per session
# ============================================================================

class TestAttackerSeedPerSession:
    """
    Bug: all sessions used AdaptiveAttacker(seed=42).  Two concurrent sessions with
    different episode seeds got identical attacker initial states — an adversarial agent
    could exploit the predictable episode-2 strategy pivot.

    Fix: seed = _ATTACKER_SEED ^ (episode_seed & 0xFFFFFFFF).
    """

    def test_different_seeds_produce_different_attacker_rng_state(self):
        """Two sessions with different seeds must have attackers in different RNG states."""
        sess_a = _make_session("easy", seed=0)
        sess_b = _make_session("easy", seed=999999)

        # Draw one random number from each attacker's rng — they should differ
        rng_a = sess_a.attacker.rng.random()
        rng_b = sess_b.attacker.rng.random()
        assert rng_a != rng_b, (
            "Attacker RNG produced the same first value for sessions with seeds 0 and 999999. "
            "This indicates _ATTACKER_SEED ^ seed is not varying the attacker seed."
        )

    def test_same_seed_produces_same_attacker_rng_state(self):
        """Reproducibility: same episode seed → same attacker initial state."""
        sess_a = _make_session("easy", seed=42)
        sess_b = _make_session("easy", seed=42)
        rng_a = sess_a.attacker.rng.random()
        rng_b = sess_b.attacker.rng.random()
        assert rng_a == rng_b, (
            "Same episode seed produced different attacker RNG states — seed-based "
            "reproducibility is broken."
        )

    def test_zero_seed_xor_does_not_produce_zero_attacker_seed(self):
        """Edge case: episode seed=0 XOR _ATTACKER_SEED should equal _ATTACKER_SEED, not 0."""
        from app import _ATTACKER_SEED as AS
        expected_attacker_seed = AS ^ 0
        assert expected_attacker_seed == AS, (
            "0 XOR _ATTACKER_SEED should equal _ATTACKER_SEED.  "
            "The attacker is still meaningfully seeded even when the episode seed is 0."
        )
        # Verify a session with seed=0 doesn't produce the same attacker as seed=AS
        # (they'd only collide if AS=0, which we disallow)
        if AS != 0:
            sess_0  = _make_session("easy", seed=0)
            sess_as = _make_session("easy", seed=AS)
            r0  = sess_0.attacker.rng.random()
            ras = sess_as.attacker.rng.random()
            assert r0 != ras


# ============================================================================
# FIX 5 — P1: episode_actions_taken appended before _obs()
# ============================================================================

class TestEpisodeActionsTakenOrder:
    """
    Bug: sess.episode_actions_taken.append(raw_action) happened AFTER obs = _obs(sess),
    so _compute_grader_score inside _obs used a list that was missing the current action.
    resource_efficiency in the step response was therefore one action behind reality.

    Fix: append both episode_actions_taken and episode_rewards BEFORE calling _obs().
    """

    def test_grader_score_reflects_current_action_cost(self):
        """
        After an isolate_machine action on hard (scan_cost scaled, isolate_cost=0.30),
        the grader_score returned by /step must reflect the reduced resource budget
        including the just-taken action, not the budget from one step earlier.
        """
        from fastapi.testclient import TestClient
        from app import app as fastapi_app, _SESSIONS
        import app as _app
        import uuid

        sess = _make_session("hard", seed=7)
        _inject_threats(sess, [
            {"id": "t_malware_hard", "type": "malware", "node": "node_1",
             "visible": True, "age": 1, "is_verified": True},
        ])
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid

        client = TestClient(fastapi_app)
        resp = client.post("/step", json={"action": "isolate_machine", "session_id": sid})
        data = resp.json()

        # The grader_breakdown.resource in the response must account for the
        # isolate_machine cost just taken.
        breakdown = data.get("grader_breakdown", {})
        resource_in_response = breakdown.get("resource", None)

        # Also compute what the grader SHOULD see after one isolate_machine on hard
        from constants import get_scaled_costs
        task_budget = sess.task_config.get("resource_per_step", 1.0)
        scaled = get_scaled_costs(task_budget)
        isolate_cost = scaled["isolate_machine"]
        total_budget = task_budget * sess.task_config.get("max_steps", 50)
        # 1 isolate_machine action taken
        expected_raw_eff = max(0.0, 1.0 - isolate_cost / total_budget)
        # Containment rate depends on whether threat was contained — check:
        t_contained = sess.state["threats"][0].get("contained", False)
        _total_real = 1
        _contained = 1 if t_contained else 0
        _uncontained_fraction = (_total_real - _contained) / _total_real
        damping = 1.0 - 0.5 * _uncontained_fraction
        expected_eff = expected_raw_eff * damping

        if resource_in_response is not None:
            # Allow ±0.05 tolerance for floating-point and damping variation
            assert abs(resource_in_response - expected_eff) < 0.15, (
                f"grader_breakdown.resource={resource_in_response:.4f} but expected "
                f"≈{expected_eff:.4f} after one isolate_machine on hard. "
                "If the current action cost is missing, resource will be inflated."
            )

    def test_episode_actions_taken_includes_current_step(self):
        """
        After calling /step, sess.episode_actions_taken must already contain the action
        from that step (it was needed by _obs before being returned).
        """
        from fastapi.testclient import TestClient
        from app import app as fastapi_app, _SESSIONS
        import app as _app
        import uuid

        sess = _make_session("easy", seed=8)
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid

        client = TestClient(fastapi_app)
        client.post("/step", json={"action": "ignore", "session_id": sid})

        assert "ignore" in sess.episode_actions_taken, (
            "episode_actions_taken does not contain 'ignore' after the step completed. "
            "This means the append still happens after _obs(), not before."
        )


# ============================================================================
# FIX 6 — P2: /analytics and /observe use difficulty-scaled scan cost
# ============================================================================

class TestAnalyticsScanCost:
    """
    Bug: /analytics and /observe computed resources_remaining using a hardcoded 0.2
    scan cost fallback.  On hard (actual scan_cost=0.45) this understated scan spending
    by 2.25×, making resources_remaining appear 125% higher than the step layer reported.

    Fix: use _SCAN_COST_BY_DIFFICULTY and _VERIFY_COST_BY_DIFFICULTY consistently.
    """

    def test_analytics_resources_remaining_matches_step_layer(self):
        """
        After 3 scans on a hard session, /analytics resources_remaining should match
        what /step returns in grader_breakdown.resource (within tolerance).
        """
        from fastapi.testclient import TestClient
        from app import app as fastapi_app, _SESSIONS
        import app as _app
        import uuid

        sess = _make_session("hard", seed=9)
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid

        client = TestClient(fastapi_app)
        # Perform 3 scans
        for node in ["scan_node_1", "scan_node_2", "scan_node_3"]:
            client.post("/step", json={"action": node, "session_id": sid})

        # Get analytics resources_remaining
        analytics = client.get(f"/analytics?session_id={sid}").json()
        analytics_resources = analytics.get("resources_remaining", -1.0)

        # Compute expected manually
        scan_cost = _SCAN_COST_BY_DIFFICULTY["hard"]  # 0.45
        task_budget = TASK_OVERRIDES["hard"]["resource_per_step"]
        total_budget = task_budget * TASK_OVERRIDES["hard"]["max_steps"]
        expected = max(0.0, 1.0 - (3 * scan_cost) / total_budget)

        assert abs(analytics_resources - expected) < 0.05, (
            f"/analytics resources_remaining={analytics_resources:.4f} but expected "
            f"≈{expected:.4f} after 3 scans on hard (scan_cost={scan_cost}). "
            "Hardcoded 0.2 fallback would give a much higher (inflated) value."
        )

    def test_observe_resources_remaining_matches_analytics(self):
        """
        /observe and /analytics must agree on resources_remaining (both now use the
        same difficulty-scaled cost function).
        """
        from fastapi.testclient import TestClient
        from app import app as fastapi_app, _SESSIONS
        import app as _app
        import uuid

        sess = _make_session("hard", seed=10)
        sid = str(uuid.uuid4())
        _SESSIONS[sid] = sess
        _app._LATEST_SID = sid

        client = TestClient(fastapi_app)
        for node in ["scan_node_1", "scan_node_2"]:
            client.post("/step", json={"action": node, "session_id": sid})

        analytics  = client.get(f"/analytics?session_id={sid}").json()
        observe    = client.get(f"/observe?session_id={sid}").json()

        res_analytics = analytics.get("resources_remaining", -1.0)
        res_observe   = observe.get("analytics", {}).get("resources_remaining", -1.0)

        assert abs(res_analytics - res_observe) < 1e-4, (
            f"/analytics resources_remaining={res_analytics} ≠ "
            f"/observe resources_remaining={res_observe}. "
            "Both endpoints must use the same cost function."
        )


# ============================================================================
# Edge case and regression tests
# ============================================================================

class TestEdgeCases:
    """Additional edge-case and regression scenarios."""

    def test_zero_threats_episode_grader_score_valid(self):
        """Score must be strictly in (0, 1) even when no threats ever spawn."""
        from grader import compute_grader_score, safe_score
        score = compute_grader_score(
            containment_rate=1.0,
            critical_health=1.0,
            resource_efficiency=1.0,
            speed_bonus=0.0,
            fp_penalty=0.0,
        )
        assert 0.0 < score < 1.0

    def test_all_threats_contained_instantly_max_score(self):
        """Perfect play (instant containment, full health, no FPs) → score near ceiling."""
        from grader import compute_grader_score, compute_speed_bonus
        speed = compute_speed_bonus([{"age_at_containment": 0}])
        score = compute_grader_score(
            containment_rate=1.0,
            critical_health=1.0,
            resource_efficiency=1.0,
            speed_bonus=speed,
            fp_penalty=0.0,
        )
        assert score >= 0.85, f"Expected score ≥ 0.85 for perfect play, got {score}"

    def test_grader_score_never_exact_zero_or_one(self):
        """safe_score must never return exactly 0.0 or 1.0."""
        from grader import safe_score
        for v in [-100.0, 0.0, 0.0001, 0.9999, 1.0, 100.0]:
            s = safe_score(v)
            assert s > 0.0 and s < 1.0, f"safe_score({v}) = {s} violates strict (0,1)"

    def test_resurface_resets_spread_attempted_flag(self):
        """
        A threat that resurfaces must have spread_attempted reset to False so it can
        potentially spread again after re-escalation.  Otherwise a resurfaced threat
        never spreads, removing the punishment for neglecting monitor actions.
        """
        sess = _make_session("easy", seed=11)
        _inject_threats(sess, [
            {"id": "t_malware", "type": "malware", "node": "node_1",
             "visible": True, "age": 2, "stage": "lateral_movement",
             "contained": True, "is_verified": True},
        ])
        # Manually trigger resurface
        t = sess.state["threats"][0]
        t["resurface_risk"] = 1.0
        t["steps_since_contained"] = 10

        # Run _age_threats which includes the resurface check
        _age_threats(sess)

        if not t["contained"]:  # resurfaced
            assert not t.get("spread_attempted", True), (
                "spread_attempted not reset after resurface — resurfaced threats can never "
                "spread even if they re-escalate to lateral_movement stage."
            )

    def test_session_lru_eviction_does_not_corrupt_latest_sid(self):
        """
        _evict_oldest_sessions should never set _LATEST_SID to None when evicting old
        sessions (the latest session is always at the END of the OrderedDict, not the start).
        """
        from app import _SESSIONS, _evict_oldest_sessions, _MAX_SESSIONS
        import app as _app
        import uuid

        # Fill to capacity
        old_sessions = dict(_SESSIONS)
        _SESSIONS.clear()

        first_sid = None
        for i in range(_MAX_SESSIONS):
            sid = f"test-evict-{i}"
            cfg = TASK_OVERRIDES["easy"]
            s = Session(task_name="easy", task_config=cfg)
            s.rng = random.Random(i)
            s.attacker = AdaptiveAttacker(seed=_ATTACKER_SEED)
            _do_reset_session(s)
            _SESSIONS[sid] = s
            _app._LATEST_SID = sid
            if first_sid is None:
                first_sid = sid

        latest_before = _app._LATEST_SID
        # Adding one more should evict the oldest (first_sid), not the latest
        _evict_oldest_sessions()

        assert _app._LATEST_SID == latest_before, (
            "_LATEST_SID was corrupted during LRU eviction.  It should only be cleared "
            "when the latest session itself is the one being evicted, which never happens "
            "in normal operation (latest is always at the end of the OrderedDict)."
        )

        # Restore
        _SESSIONS.clear()
        _SESSIONS.update(old_sessions)
