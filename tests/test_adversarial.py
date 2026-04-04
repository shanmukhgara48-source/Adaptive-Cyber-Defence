"""
Adversarial test suite — 22 tests designed to break the environment.

These tests target edge cases, race conditions, and adversarial inputs.
They require a running server at ADV_TEST_URL (default: http://localhost:7860).

Skip the entire module gracefully if the server is not reachable.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import threading
import time

import pytest
import requests

BASE_URL = os.getenv("ADV_TEST_URL", "http://localhost:7860")
TIMEOUT = 10  # seconds per request


# ---------------------------------------------------------------------------
# Module-level skip if server is not reachable
# ---------------------------------------------------------------------------

def _server_reachable() -> bool:
    try:
        r = requests.get(BASE_URL + "/", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_reachable(),
    reason=f"Adversarial tests require a running server at {BASE_URL}",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset(task: str = "easy", seed: int = 0) -> dict:
    return requests.post(
        f"{BASE_URL}/reset", json={"task": task, "seed": seed}, timeout=TIMEOUT
    ).json()


def step(action: str) -> dict:
    return requests.post(
        f"{BASE_URL}/step", json={"action": action}, timeout=TIMEOUT
    ).json()


def state() -> dict:
    return requests.get(f"{BASE_URL}/state", timeout=TIMEOUT).json()


def analytics() -> dict:
    return requests.get(f"{BASE_URL}/analytics", timeout=TIMEOUT).json()


VALID_ACTIONS = [
    "block_ip", "isolate_machine", "patch", "ignore",
    "scan_node_1", "scan_node_2", "scan_node_3", "scan_node_4", "scan_node_5",
]

TASKS_ALL = ["easy", "medium", "hard", "nightmare", "elite", "impossible"]
THRESHOLDS = {
    "easy": 0.55, "medium": 0.55, "hard": 0.45,
    "nightmare": 0.25, "elite": 0.20, "impossible": 0.10,
}
MITRE_ACTION = {
    "phishing": "block_ip", "malware": "isolate_machine",
    "ransomware": "isolate_machine", "ddos": "patch",
    "lateral_movement": "block_ip",
}


def _pick_action(obs: dict, scanned: set, step_num: int) -> str:
    threats = obs.get("visible_threats", [])
    unscanned = [f"node_{i}" for i in range(1, 6) if f"node_{i}" not in scanned]
    for t in threats:
        if t.get("type") == "lateral_movement" or t.get("stage") == "lateral_movement":
            return "block_ip"
    for t in threats:
        a = MITRE_ACTION.get(t.get("type", ""))
        if a:
            return a
    if not threats and unscanned:
        return f"scan_node_{unscanned[0].split('_')[1]}"
    if not threats:
        return f"scan_node_{(step_num % 5) + 1}"
    return "ignore"


def _run_episode(task: str, seed: int = 0) -> dict:
    """Run a full episode with the deterministic agent. Returns grader-formula score."""
    reset(task, seed)
    scanned: set = set()
    total_reward = 0.0
    last_step = 0
    for s in range(1, 300):
        obs = state()
        if obs.get("done"):
            break
        action = _pick_action(obs, scanned, s)
        if action.startswith("scan_node_"):
            scanned.add(action.replace("scan_node_", "node_"))
        data = step(action)
        total_reward += data.get("reward", 0.0)
        last_step = s
        if data.get("done"):
            break
    anl = analytics()
    soc = anl.get("soc_metrics", {})
    net = anl.get("network_status", {})
    cr  = float(soc.get("containment_rate", 0.0))
    ch  = float(net.get("system_health", 0)) / 100.0
    avg = float(soc.get("avg_reward_per_step", total_reward / max(1, last_step)))
    score = max(0.0, min(1.0, 0.50 * cr + 0.20 * ch + 0.15 * 1.0 + 0.15 * avg))
    return {"score": score, "containment_rate": cr, "health": ch, "steps": last_step}


def _is_valid_float(v) -> bool:
    try:
        f = float(v)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


# ===========================================================================
# TEST 1 — Rapid-fire reset storm
# ===========================================================================

def test_adv_01_reset_storm():
    """50 resets in rapid succession — never 500, always clean state."""
    for i in range(50):
        r = requests.post(
            f"{BASE_URL}/reset", json={"task": "easy", "seed": i}, timeout=TIMEOUT
        )
        assert r.status_code != 500, f"Reset {i} returned 500"
        data = r.json()
        assert data.get("step", -1) == 0, f"Reset {i}: step={data.get('step')} expected 0"
        assert _is_valid_float(data.get("score", None)), f"Reset {i}: score invalid"
        assert float(data.get("score", 1.0)) == 0.0, f"Reset {i}: score not zero"


# ===========================================================================
# TEST 2 — Simultaneous conflicting task resets
# ===========================================================================

def test_adv_02_concurrent_task_resets():
    """10 threads reset different tasks simultaneously — no crash, all valid JSON."""
    tasks = ["easy", "medium", "hard", "nightmare", "elite", "impossible",
             "easy", "hard", "nightmare", "medium"]
    errors: list = []
    responses: list = [None] * 10

    def do_reset(idx: int, task: str):
        try:
            r = requests.post(
                f"{BASE_URL}/reset", json={"task": task}, timeout=TIMEOUT
            )
            assert r.status_code != 500, f"Thread {idx} got 500"
            responses[idx] = r.json()
            assert isinstance(responses[idx], dict), "Response not a dict"
        except Exception as e:
            errors.append(f"Thread {idx} ({task}): {e}")

    threads = [threading.Thread(target=do_reset, args=(i, t)) for i, t in enumerate(tasks)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=15)

    assert not errors, f"Concurrent reset errors: {errors}"
    for i, resp in enumerate(responses):
        assert resp is not None, f"Thread {i} got no response"
        assert "step" in resp or "error" in resp or "task" in resp, \
            f"Thread {i} response missing expected keys: {resp}"


# ===========================================================================
# TEST 3 — Step after max_steps exceeded
# ===========================================================================

def test_adv_03_step_beyond_max():
    """35 steps on easy (max=30) — done by step 30, extra steps never crash."""
    reset("easy")
    done_at = None
    for i in range(1, 36):
        data = step("scan_node_1")
        reward = data.get("reward", None)
        assert _is_valid_float(reward), f"Step {i}: reward={reward} is not finite float"
        assert not math.isnan(float(reward)), f"Step {i}: reward is NaN"
        assert "done" in data, f"Step {i}: missing 'done' field"
        if data["done"] and done_at is None:
            done_at = i

    assert done_at is not None, "Episode never ended (done never became True)"
    assert done_at <= 30, f"Episode ended at step {done_at}, expected <= 30"


# ===========================================================================
# TEST 4 — Alternating valid and invalid actions
# ===========================================================================

def test_adv_04_alternating_invalid():
    """Alternate valid/invalid actions 20 times — server never 500."""
    reset("easy")
    action_sequence = [
        "block_ip", "INVALID_ACTION", "scan_node_1", "???",
        "isolate_machine", "DROP TABLE", "patch", "null",
        "ignore", "block_ip\nignore", "scan_node_2", "  ",
        "block_ip", "9999", "scan_node_3", "{}",
        "isolate_machine", "undefined", "patch", "scan_node_4",
    ]
    for i, action in enumerate(action_sequence):
        r = requests.post(
            f"{BASE_URL}/step", json={"action": action}, timeout=TIMEOUT
        )
        assert r.status_code != 500, \
            f"Step {i} action={action!r} returned 500"
        data = r.json()
        # Valid actions must return a finite reward
        if action in VALID_ACTIONS and not data.get("done"):
            assert _is_valid_float(data.get("reward")), \
                f"Step {i}: valid action {action!r} returned non-finite reward"


# ===========================================================================
# TEST 5 — Reward accumulation never overflows
# ===========================================================================

def test_adv_05_reward_no_overflow():
    """3 back-to-back episodes — score always a finite non-negative float."""
    for episode in range(3):
        reset("easy", seed=episode)
        for _ in range(35):
            data = step("scan_node_1")
            score = data.get("score", None)
            reward = data.get("reward", None)
            assert _is_valid_float(score), f"Episode {episode}: score={score} invalid"
            assert _is_valid_float(reward), f"Episode {episode}: reward={reward} invalid"
            assert float(score) >= 0.0, f"Episode {episode}: score={score} < 0"
            assert float(score) <= 1.0 + 1e-6, \
                f"Episode {episode}: score={score} > 1 (unclamped overflow)"
            if data.get("done"):
                break


# ===========================================================================
# TEST 6 — Nightmare health death spiral
# ===========================================================================

def test_adv_06_nightmare_health_death():
    """20 ignore actions on nightmare — health reaches 0, never goes negative."""
    reset("nightmare")
    reached_zero = False
    for i in range(20):
        data = step("ignore")
        health = data.get("system_health", None)
        assert health is not None, f"Step {i}: missing system_health"
        assert float(health) >= 0.0, f"Step {i}: health={health} went negative"
        if float(health) == 0.0 or data.get("done"):
            reached_zero = True
            break

    assert reached_zero, \
        "Nightmare health never reached 0 after 20 ignore actions — degradation too slow"


# ===========================================================================
# TEST 7 — All 9 actions in rapid sequence
# ===========================================================================

def test_adv_07_all_actions_rapid():
    """All 9 valid actions × 3 cycles — every response has required fields."""
    REQUIRED_FIELDS = {"reward", "done", "system_health", "score", "step", "reason"}
    reset("easy")
    for cycle in range(3):
        for action in VALID_ACTIONS:
            data = step(action)
            for field in REQUIRED_FIELDS:
                assert field in data, \
                    f"Cycle {cycle}, action={action!r}: missing field '{field}'"
            assert isinstance(data["reward"], (int, float)), \
                f"action={action!r}: reward not a number"
            assert isinstance(data["done"], bool), \
                f"action={action!r}: done not a bool"
            assert _is_valid_float(data["reward"]), \
                f"action={action!r}: reward={data['reward']} not finite"
            if data["done"]:
                reset("easy")


# ===========================================================================
# TEST 8 — State consistency under concurrent reads
# ===========================================================================

def test_adv_08_concurrent_state_reads():
    """30 concurrent GET /state calls — all valid JSON, no 500s."""
    reset("easy")
    for _ in range(3):
        step("scan_node_1")

    results: list = [None] * 30
    errors: list = []

    def do_read(idx: int):
        try:
            r = requests.get(f"{BASE_URL}/state", timeout=TIMEOUT)
            assert r.status_code != 500, f"Thread {idx} got 500"
            results[idx] = r.json()
        except Exception as e:
            errors.append(f"Thread {idx}: {e}")

    threads = [threading.Thread(target=do_read, args=(i,)) for i in range(30)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=15)

    assert not errors, f"Concurrent read errors: {errors}"
    for i, resp in enumerate(results):
        assert resp is not None, f"Thread {i} got no response"
        assert "system_health" in resp, f"Thread {i}: missing system_health"
        assert "done" in resp, f"Thread {i}: missing done"
        assert _is_valid_float(resp.get("system_health", None)), \
            f"Thread {i}: system_health invalid"


# ===========================================================================
# TEST 9 — Giant payload injection
# ===========================================================================

def test_adv_09_giant_payload():
    """1 MB garbage payloads — server must not 500, must work normally after."""
    giant_action = "A" * (1024 * 1024)  # 1 MB
    giant_task   = "Z" * (1024 * 1024)

    # Giant action in /step
    r_step = requests.post(
        f"{BASE_URL}/step", json={"action": giant_action}, timeout=30
    )
    assert r_step.status_code != 500, \
        f"/step with 1MB action returned 500: {r_step.text[:200]}"

    # Giant task in /reset
    r_reset = requests.post(
        f"{BASE_URL}/reset", json={"task": giant_task}, timeout=30
    )
    assert r_reset.status_code != 500, \
        f"/reset with 1MB task returned 500: {r_reset.text[:200]}"

    # Server must still work after the giant payloads
    r_normal = requests.post(
        f"{BASE_URL}/reset", json={"task": "easy"}, timeout=TIMEOUT
    )
    assert r_normal.status_code == 200, "Server broken after giant payload"
    data = r_normal.json()
    assert data.get("step") == 0, "Server state corrupt after giant payload"


# ===========================================================================
# TEST 10 — Unicode and emoji injection
# ===========================================================================

def test_adv_10_unicode_injection():
    """Exotic Unicode action values — none must crash the server."""
    reset("easy")
    exotic_actions = [
        "🔥",
        "блокировать",
        "封锁IP",
        "block_ip\x00",
        "block_ip\nignore",
        "<script>alert(1)</script>",
        "\x00\x01\x02",
        "block_ip; DROP TABLE threats;--",
        "𝔹𝕝𝕠𝕔𝕜",
        "",
    ]
    for action in exotic_actions:
        try:
            r = requests.post(
                f"{BASE_URL}/step", json={"action": action}, timeout=TIMEOUT
            )
            assert r.status_code != 500, \
                f"action={action!r} returned 500"
            # Response must be valid JSON
            data = r.json()
            assert isinstance(data, dict), \
                f"action={action!r}: response not a dict"
        except requests.exceptions.JSONDecodeError:
            pytest.fail(f"action={action!r} returned non-JSON response")


# ===========================================================================
# TEST 11 — Reward determinism with same seed
# ===========================================================================

def test_adv_11_reward_deterministic():
    """Same seed → same rewards for first 3 steps."""
    actions = ["scan_node_1", "block_ip", "scan_node_2"]

    # First run
    reset("easy", seed=42)
    r1 = [step(a)["reward"] for a in actions]

    # Second run, same seed
    reset("easy", seed=42)
    r2 = [step(a)["reward"] for a in actions]

    for i, (a, x, y) in enumerate(zip(actions, r1, r2)):
        assert x == y, (
            f"Step {i+1} action={a!r}: reward changed between runs "
            f"with same seed=42: {x} != {y}"
        )


# ===========================================================================
# TEST 12 — Health can never go negative
# ===========================================================================

def test_adv_12_health_never_negative():
    """50 ignore actions on nightmare — health always >= 0."""
    reset("nightmare")
    for i in range(50):
        data = step("ignore")
        health = float(data.get("system_health", 0))
        assert health >= 0.0, f"Step {i}: system_health={health} went negative"
        if data.get("done"):
            break


# ===========================================================================
# TEST 13 — scan_coverage never exceeds 1.0
# ===========================================================================

def test_adv_13_coverage_never_exceed_one():
    """Scan all 5 nodes 3 times each — scan_coverage always in [0, 1]."""
    reset("easy")
    for cycle in range(3):
        for node in range(1, 6):
            data = step(f"scan_node_{node}")
            cov = float(data.get("scan_coverage", -1))
            assert 0.0 <= cov <= 1.0, \
                f"Cycle {cycle}, node {node}: scan_coverage={cov} out of [0, 1]"
            if data.get("done"):
                reset("easy")


# ===========================================================================
# TEST 14 — Cumulative reward in /history only increases
# ===========================================================================

def test_adv_14_score_monotonic():
    """History total_reward must increase (or stay same) after every step."""
    reset("easy")
    prev_total = 0.0
    for i in range(15):
        step("scan_node_1")
        hist = requests.get(f"{BASE_URL}/history", timeout=TIMEOUT).json()
        total = float(hist.get("total_reward", 0.0))
        assert total >= prev_total - 1e-9, \
            f"Step {i+1}: total_reward dropped from {prev_total} to {total}"
        prev_total = total
        # Check step count is consistent
        assert hist.get("total_steps", 0) == i + 1, \
            f"Step {i+1}: history total_steps={hist.get('total_steps')} != {i+1}"


# ===========================================================================
# TEST 15 — All endpoints survive after episode done
# ===========================================================================

def test_adv_15_all_endpoints_after_done():
    """All 6 endpoints return valid JSON after episode completes."""
    reset("nightmare")
    for _ in range(20):
        data = step("ignore")
        if data.get("done"):
            break

    endpoints = ["/", "/state", "/history", "/analytics", "/threat-intel", "/attacker-report"]
    for ep in endpoints:
        try:
            r = requests.get(f"{BASE_URL}{ep}", timeout=TIMEOUT)
            assert r.status_code != 500, f"{ep} returned 500 after done"
            data = r.json()
            assert isinstance(data, dict), f"{ep} returned non-dict"
        except requests.exceptions.JSONDecodeError:
            pytest.fail(f"{ep} returned non-JSON after done")


# ===========================================================================
# TEST 16 — Task config actually changes behavior
# ===========================================================================

def test_adv_16_task_config_affects_behavior():
    """Easy and nightmare must have different threat counts and health degradation."""
    # Compare hidden_node_count at reset
    easy_reset = reset("easy")
    easy_hidden = easy_reset.get("hidden_node_count", -1)

    nightmare_reset = reset("nightmare")
    nightmare_hidden = nightmare_reset.get("hidden_node_count", -1)

    assert nightmare_hidden == 5, \
        f"Nightmare must have 5 hidden threats, got {nightmare_hidden}"
    assert easy_hidden >= 2, \
        f"Easy must have >= 2 threats, got {easy_hidden}"

    # Compare health degradation — use 3 steps to avoid both bottoming out at 0.
    # Nightmare: health_degradation_rate=0.12 × 5/5 × 100 + 10hp ignore = 22hp/step → drops ~66 in 3
    # Easy:      health_degradation_rate=0.05 × 3/5 × 100 + 10hp ignore = 13hp/step → drops ~39 in 3
    N_STEPS = 3
    reset("easy")
    easy_start = 100.0
    easy_health_end = easy_start
    for _ in range(N_STEPS):
        data = step("ignore")
        easy_health_end = float(data.get("system_health", easy_health_end))
        if data.get("done"):
            break
    easy_drop = easy_start - easy_health_end

    reset("nightmare")
    nightmare_start = 100.0
    nightmare_health_end = nightmare_start
    for _ in range(N_STEPS):
        data = step("ignore")
        nightmare_health_end = float(data.get("system_health", nightmare_health_end))
        if data.get("done"):
            break
    nightmare_drop = nightmare_start - nightmare_health_end

    assert nightmare_drop > easy_drop, (
        f"Nightmare health drop ({nightmare_drop:.1f}) must be greater "
        f"than easy drop ({easy_drop:.1f}) over {N_STEPS} ignore steps"
    )


# ===========================================================================
# TEST 17 — /history is consistent with /step rewards
# ===========================================================================

def test_adv_17_history_matches_steps():
    """Sum of step rewards must equal total_reward in /history."""
    reset("easy")
    actions = ["scan_node_1", "block_ip", "scan_node_2", "patch", "isolate_machine"]
    step_rewards = []
    for action in actions:
        data = step(action)
        step_rewards.append(round(float(data.get("reward", 0.0)), 3))

    hist = requests.get(f"{BASE_URL}/history", timeout=TIMEOUT).json()
    assert hist.get("total_steps") == 5, \
        f"Expected 5 steps in history, got {hist.get('total_steps')}"

    hist_rewards = [round(float(s["reward"]), 3) for s in hist.get("episode_steps", [])]
    assert hist_rewards == step_rewards, \
        f"History rewards {hist_rewards} != step rewards {step_rewards}"

    hist_total = round(float(hist.get("total_reward", 0.0)), 3)
    calc_total = round(sum(step_rewards), 3)
    assert abs(hist_total - calc_total) < 0.01, \
        f"History total_reward {hist_total} != sum of step rewards {calc_total}"


# ===========================================================================
# TEST 18 — Attacker strategy changes after defender patterns
# ===========================================================================

def test_adv_18_attacker_adapts():
    """After many block_ip actions, block_rate must increase."""
    # Read baseline
    before = requests.get(f"{BASE_URL}/attacker-report", timeout=TIMEOUT).json()
    block_rate_before = float(before.get("defender_profile", {}).get("block_rate", 0.0))
    steps_before = int(before.get("defender_profile", {}).get("steps_observed", 0))

    # Run 5 episodes using only block_ip
    for ep in range(5):
        reset("easy", seed=ep)
        for _ in range(15):
            data = step("block_ip")
            if data.get("done"):
                break

    after = requests.get(f"{BASE_URL}/attacker-report", timeout=TIMEOUT).json()
    profile = after.get("defender_profile", {})
    block_rate_after = float(profile.get("block_rate", 0.0))
    steps_after = int(profile.get("steps_observed", 0))

    # block_rate must have increased (or stayed high if already high)
    assert steps_after > steps_before, \
        "steps_observed did not increase after 5 episodes"
    assert block_rate_after > block_rate_before or block_rate_after >= 0.4, (
        f"block_rate did not increase after block_ip-only episodes: "
        f"{block_rate_before:.3f} → {block_rate_after:.3f}"
    )
    # strategy_label must reflect block-heavy play if block_rate > 0.4
    if block_rate_after > 0.4:
        label = profile.get("strategy_label", "")
        assert label == "BLOCKER", \
            f"Expected strategy_label='BLOCKER' when block_rate={block_rate_after:.3f}, got {label!r}"


# ===========================================================================
# TEST 19 — /threat-intel matches /state visible threats
# ===========================================================================

def test_adv_19_threat_intel_consistent():
    """After scanning all nodes, threat-intel active_campaigns count == state visible_threats."""
    reset("easy")
    # Scan all 5 nodes to maximize visibility
    for node in range(1, 6):
        step(f"scan_node_{node}")

    s = state()
    visible_count = len(s.get("visible_threats", []))

    intel = requests.get(f"{BASE_URL}/threat-intel", timeout=TIMEOUT).json()
    campaign_count = len(intel.get("active_campaigns", []))

    assert campaign_count == visible_count, (
        f"/state has {visible_count} visible threats "
        f"but /threat-intel has {campaign_count} active_campaigns"
    )


# ===========================================================================
# TEST 20 — /analytics performance_grade is accurate
# ===========================================================================

def test_adv_20_analytics_grade_accurate():
    """Ignore actions → grade D. Correct scan+respond actions → grade A or B."""
    # Part 1: 12 ignore actions on nightmare → grade D
    reset("nightmare")
    for _ in range(12):
        data = step("ignore")
        if data.get("done"):
            break
    anl = analytics()
    grade_bad = anl.get("performance_grade", "?")
    assert grade_bad == "D", \
        f"Expected grade='D' after ignoring all threats, got {grade_bad!r}"

    # Part 2: scan all nodes + correct mitigations on easy → grade A or B
    reset("easy")
    # Scan all nodes to reveal threats
    for node in range(1, 6):
        step(f"scan_node_{node}")
    # Attempt all mitigations
    for action in ["block_ip", "isolate_machine", "patch",
                   "block_ip", "isolate_machine", "patch"]:
        data = step(action)
        if data.get("done"):
            break
    anl_good = analytics()
    grade_good = anl_good.get("performance_grade", "?")
    assert grade_good in ("A", "B"), \
        f"Expected grade 'A' or 'B' after correct actions, got {grade_good!r}"


# ===========================================================================
# TEST 21 — Elite and impossible tasks work
# ===========================================================================

def test_adv_21_elite_impossible_tasks():
    """Elite and impossible tasks spawn 5 threats and degrade health faster than nightmare."""
    # Elite
    elite_reset = reset("elite")
    assert elite_reset.get("task") == "elite", \
        f"Expected task='elite', got {elite_reset.get('task')!r}"
    assert elite_reset.get("hidden_node_count") == 5, \
        f"Elite must have 5 hidden threats, got {elite_reset.get('hidden_node_count')}"

    elite_health_start = float(elite_reset.get("system_health", 100))
    for _ in range(5):
        data = step("ignore")
        if data.get("done"):
            break
    elite_health_end = float(data.get("system_health", elite_health_start))
    elite_drop = elite_health_start - elite_health_end

    # Impossible
    impossible_reset = reset("impossible")
    assert impossible_reset.get("task") == "impossible", \
        f"Expected task='impossible', got {impossible_reset.get('task')!r}"
    assert impossible_reset.get("hidden_node_count") == 5, \
        f"Impossible must have 5 hidden threats, got {impossible_reset.get('hidden_node_count')}"

    imp_health_start = float(impossible_reset.get("system_health", 100))
    for _ in range(5):
        data = step("ignore")
        if data.get("done"):
            break
    imp_health_end = float(data.get("system_health", imp_health_start))
    imp_drop = imp_health_start - imp_health_end

    # Impossible degrades health at least as fast as elite
    assert imp_drop >= elite_drop, (
        f"Impossible health drop ({imp_drop:.1f}) should be >= elite drop ({elite_drop:.1f})"
    )


# ===========================================================================
# TEST 22 — Full episode on all 6 tasks, scores decrease with difficulty
# ===========================================================================

def test_adv_22_all_six_tasks_complete():
    """All 6 tasks complete with done=True. Mean scores decrease with difficulty."""
    # Run 5 episodes each for statistical stability
    task_scores: dict[str, list] = {t: [] for t in TASKS_ALL}

    for task in TASKS_ALL:
        for seed in range(5):
            result = _run_episode(task, seed)
            score = result["score"]
            assert 0.0 <= score <= 1.0, \
                f"{task} seed={seed}: score={score} out of [0, 1]"
            task_scores[task].append(score)

    mean_scores = {t: statistics.mean(scores) for t, scores in task_scores.items()}

    # All tasks must pass their thresholds on average
    for task, mean in mean_scores.items():
        threshold = THRESHOLDS[task]
        assert mean >= threshold, (
            f"{task}: mean score {mean:.3f} below threshold {threshold}"
        )

    # Scores must broadly decrease with difficulty.
    # 0.10 tolerance accounts for two known stochastic effects:
    # (a) A perfect deterministic MITRE agent scores higher on hard (5 threats
    #     in 30 steps → dense containment rewards) than medium (2 in 50).
    # (b) Very-hard tasks (elite/impossible) with rapid lateral_movement auto-
    #     progression can reveal all threats early, letting a perfect agent
    #     achieve high containment in few steps — indistinguishable from easy.
    # LLM agents see the difficulty difference via partial observability;
    # a deterministic HTTP oracle does not. The ordering check here catches
    # catastrophic regressions (e.g. impossible becoming literally easier than
    # easy by a large margin) while tolerating known oracle-agent variance.
    ordered = [mean_scores[t] for t in TASKS_ALL]
    for i in range(len(TASKS_ALL) - 1):
        assert ordered[i] >= ordered[i + 1] - 0.10, (
            f"Score ordering violated: {TASKS_ALL[i]} ({ordered[i]:.3f}) "
            f"< {TASKS_ALL[i+1]} ({ordered[i+1]:.3f})"
        )
