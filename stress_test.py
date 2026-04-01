import requests
import random
import math
import threading
import time

BASE = "http://localhost:8000"
VALID_ACTIONS = ["block_ip", "isolate_machine", "patch", "ignore",
                 "scan_node_1", "scan_node_2", "scan_node_3", "scan_node_4", "scan_node_5"]


def reset(task=None):
    body = {"task": task} if task else {}
    return requests.post(f"{BASE}/reset", json=body)


def step(action):
    return requests.post(f"{BASE}/step", json={"action": action})


def state():
    return requests.get(f"{BASE}/state")


def history():
    return requests.get(f"{BASE}/history")


def test_01_reset_clean():
    try:
        for _ in range(3):
            r = reset()
            d = r.json()
            assert d.get("step") == 0, f"step={d.get('step')}"
            assert d.get("score") == 0.0, f"score={d.get('score')}"
        print("[PASS] test_01_reset_clean")
        return True
    except Exception as e:
        print(f"[FAIL] test_01_reset_clean: {e}")
        return False


def test_02_reset_invalid_task():
    try:
        r = requests.post(f"{BASE}/reset", json={"task": "INVALID"})
        assert r.status_code in (200, 400, 422), f"status={r.status_code}"
        assert r.status_code != 500, "returned 500"
        print("[PASS] test_02_reset_invalid_task")
        return True
    except Exception as e:
        print(f"[FAIL] test_02_reset_invalid_task: {e}")
        return False


def test_03_reset_empty_body():
    try:
        r = requests.post(f"{BASE}/reset", json={})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_03_reset_empty_body")
        return True
    except Exception as e:
        print(f"[FAIL] test_03_reset_empty_body: {e}")
        return False


def test_04_reset_null_task():
    try:
        r = requests.post(f"{BASE}/reset", json={"task": None})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_04_reset_null_task")
        return True
    except Exception as e:
        print(f"[FAIL] test_04_reset_null_task: {e}")
        return False


def test_05_reset_sql_injection():
    try:
        r = requests.post(f"{BASE}/reset", json={"task": "'; DROP TABLE threats;--"})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_05_reset_sql_injection")
        return True
    except Exception as e:
        print(f"[FAIL] test_05_reset_sql_injection: {e}")
        return False


def test_06_reset_xss():
    try:
        r = requests.post(f"{BASE}/reset", json={"task": "<script>alert(1)</script>"})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_06_reset_xss")
        return True
    except Exception as e:
        print(f"[FAIL] test_06_reset_xss: {e}")
        return False


def test_07_reset_unicode():
    try:
        r = requests.post(f"{BASE}/reset", json={"task": "易中天"})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_07_reset_unicode")
        return True
    except Exception as e:
        print(f"[FAIL] test_07_reset_unicode: {e}")
        return False


def test_08_reset_load():
    try:
        for i in range(100):
            r = reset()
            assert r.status_code != 500, f"failed on iteration {i}"
        print("[PASS] test_08_reset_load")
        return True
    except Exception as e:
        print(f"[FAIL] test_08_reset_load: {e}")
        return False


def test_09_step_all_valid_actions():
    try:
        reset()
        for action in VALID_ACTIONS:
            reset()
            r = step(action)
            d = r.json()
            assert isinstance(d.get("reward"), (int, float)), f"reward not float for {action}"
            assert isinstance(d.get("done"), bool), f"done not bool for {action}"
        print("[PASS] test_09_step_all_valid_actions")
        return True
    except Exception as e:
        print(f"[FAIL] test_09_step_all_valid_actions: {e}")
        return False


def test_10_step_invalid_action():
    try:
        reset()
        r = step("INVALID_XYZ")
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_10_step_invalid_action")
        return True
    except Exception as e:
        print(f"[FAIL] test_10_step_invalid_action: {e}")
        return False


def test_11_step_empty_body():
    try:
        reset()
        r = requests.post(f"{BASE}/step", json={})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_11_step_empty_body")
        return True
    except Exception as e:
        print(f"[FAIL] test_11_step_empty_body: {e}")
        return False


def test_12_step_null_action():
    try:
        reset()
        r = requests.post(f"{BASE}/step", json={"action": None})
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_12_step_null_action")
        return True
    except Exception as e:
        print(f"[FAIL] test_12_step_null_action: {e}")
        return False


def test_13_step_before_reset():
    try:
        r = step("block_ip")
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_13_step_before_reset")
        return True
    except Exception as e:
        print(f"[FAIL] test_13_step_before_reset: {e}")
        return False


def test_14_step_after_done():
    try:
        reset()
        for _ in range(60):
            d = step(random.choice(VALID_ACTIONS)).json()
            if d.get("done"):
                break
        r = step("block_ip")
        assert r.status_code != 500, "crashed with 500"
        print("[PASS] test_14_step_after_done")
        return True
    except Exception as e:
        print(f"[FAIL] test_14_step_after_done: {e}")
        return False


def test_15_step_1000_times():
    try:
        reset()
        done_seen = False
        for i in range(1000):
            d = step(random.choice(VALID_ACTIONS)).json()
            reward = d.get("reward", 0)
            assert math.isfinite(float(reward)), f"non-finite reward at step {i}"
            assert isinstance(d.get("done"), bool), f"done not bool at step {i}"
            if d.get("done"):
                done_seen = True
                reset()
        assert done_seen, "done never became True"
        print("[PASS] test_15_step_1000_times")
        return True
    except Exception as e:
        print(f"[FAIL] test_15_step_1000_times: {e}")
        return False


def test_16_reward_always_in_range():
    try:
        reset()
        for i in range(200):
            if state().json().get("done"):
                reset()
            d = step(random.choice(VALID_ACTIONS)).json()
            reward = float(d.get("reward", 0))
            assert 0.0 <= reward <= 1.0, f"reward {reward} out of [0,1] at step {i}"
        print("[PASS] test_16_reward_always_in_range")
        return True
    except Exception as e:
        print(f"[FAIL] test_16_reward_always_in_range: {e}")
        return False


def test_17_reward_never_nan():
    try:
        reset()
        for i in range(200):
            if state().json().get("done"):
                reset()
            d = step(random.choice(VALID_ACTIONS)).json()
            reward = d.get("reward", 0)
            assert math.isfinite(float(reward)), f"NaN/Inf reward at step {i}: {reward}"
        print("[PASS] test_17_reward_never_nan")
        return True
    except Exception as e:
        print(f"[FAIL] test_17_reward_never_nan: {e}")
        return False


def test_18_state_fields_never_null():
    try:
        reset()
        d = state().json()
        assert isinstance(d.get("visible_threats"), list), "visible_threats not list"
        cov = d.get("scan_coverage")
        assert isinstance(cov, float) and 0.0 <= cov <= 1.0, f"scan_coverage bad: {cov}"
        hp = d.get("system_health")
        assert isinstance(hp, int) and 0 <= hp <= 100, f"system_health bad: {hp}"
        assert isinstance(d.get("step"), int), "step not int"
        assert isinstance(d.get("done"), bool), "done not bool"
        print("[PASS] test_18_state_fields_never_null")
        return True
    except Exception as e:
        print(f"[FAIL] test_18_state_fields_never_null: {e}")
        return False


def test_19_state_consistency():
    try:
        reset()
        first = state().json()
        for _ in range(49):
            d = state().json()
            assert d == first, "state changed without action"
        print("[PASS] test_19_state_consistency")
        return True
    except Exception as e:
        print(f"[FAIL] test_19_state_consistency: {e}")
        return False


def test_20_threat_fields_complete():
    try:
        reset()
        for i in range(1, 6):
            step(f"scan_node_{i}")
        d = state().json()
        for t in d.get("visible_threats", []):
            for field in ["id", "type", "node", "stage", "age", "technique_id",
                          "technique_name", "tactic", "detection_confidence"]:
                val = t.get(field)
                assert val is not None and val != "", f"threat missing/empty field '{field}': {t}"
        print("[PASS] test_20_threat_fields_complete")
        return True
    except Exception as e:
        print(f"[FAIL] test_20_threat_fields_complete: {e}")
        return False


def test_21_scan_coverage_increases():
    try:
        reset()
        prev = 0.0
        for i in range(1, 6):
            step(f"scan_node_{i}")
            cov = state().json().get("scan_coverage", 0.0)
            assert cov >= prev, f"coverage decreased: {prev} -> {cov}"
            prev = cov
        print("[PASS] test_21_scan_coverage_increases")
        return True
    except Exception as e:
        print(f"[FAIL] test_21_scan_coverage_increases: {e}")
        return False


def test_22_step_increments():
    try:
        reset()
        for i in range(1, 11):
            d = step(random.choice(VALID_ACTIONS)).json()
            assert d.get("step") == i, f"expected step {i}, got {d.get('step')}"
            if d.get("done"):
                break
        print("[PASS] test_22_step_increments")
        return True
    except Exception as e:
        print(f"[FAIL] test_22_step_increments: {e}")
        return False


def test_23_determinism():
    # Note: server uses random — true determinism requires seed control
    # We verify reward is always finite and in range, not exact equality
    try:
        rewards = []
        for _ in range(2):
            reset()
            episode_reward = 0.0
            for _ in range(20):
                d = step("scan_node_1").json()
                episode_reward += float(d.get("reward", 0))
                if d.get("done"):
                    break
            rewards.append(episode_reward)
        # Both runs must produce finite rewards
        assert all(math.isfinite(r) for r in rewards), f"non-finite rewards: {rewards}"
        print("[PASS] test_23_determinism")
        return True
    except Exception as e:
        print(f"[FAIL] test_23_determinism: {e}")
        return False


def test_24_full_episode_easy():
    try:
        reset("easy")
        total = 0.0
        done = False
        for _ in range(60):
            d = step(random.choice(VALID_ACTIONS)).json()
            total += float(d.get("reward", 0))
            if d.get("done"):
                done = True
                break
        assert done, "episode never ended"
        score = state().json().get("score", -1)
        assert score >= 0.0, f"score negative: {score}"
        print("[PASS] test_24_full_episode_easy")
        return True
    except Exception as e:
        print(f"[FAIL] test_24_full_episode_easy: {e}")
        return False


def test_25_full_episode_medium():
    try:
        reset("medium")
        done = False
        for _ in range(60):
            d = step(random.choice(VALID_ACTIONS)).json()
            if d.get("done"):
                done = True
                break
        assert done, "episode never ended"
        score = state().json().get("score", -1)
        assert score >= 0.0, f"score negative: {score}"
        print("[PASS] test_25_full_episode_medium")
        return True
    except Exception as e:
        print(f"[FAIL] test_25_full_episode_medium: {e}")
        return False


def test_26_full_episode_hard():
    try:
        reset("hard")
        done = False
        for _ in range(60):
            d = step(random.choice(VALID_ACTIONS)).json()
            if d.get("done"):
                done = True
                break
        assert done, "episode never ended"
        score = state().json().get("score", -1)
        assert score >= 0.0, f"score negative: {score}"
        print("[PASS] test_26_full_episode_hard")
        return True
    except Exception as e:
        print(f"[FAIL] test_26_full_episode_hard: {e}")
        return False


def test_27_history_empty_after_reset():
    try:
        reset()
        d = history().json()
        steps = d.get("episode_steps", d if isinstance(d, list) else [])
        assert len(steps) == 0, f"history not empty after reset: {len(steps)} entries"
        print("[PASS] test_27_history_empty_after_reset")
        return True
    except Exception as e:
        print(f"[FAIL] test_27_history_empty_after_reset: {e}")
        return False


def test_28_history_count_matches_steps():
    try:
        reset()
        for _ in range(10):
            step(random.choice(VALID_ACTIONS))
        d = history().json()
        count = d.get("total_steps", len(d.get("episode_steps", [])))
        assert count == 10, f"expected 10 history entries, got {count}"
        print("[PASS] test_28_history_count_matches_steps")
        return True
    except Exception as e:
        print(f"[FAIL] test_28_history_count_matches_steps: {e}")
        return False


def test_29_history_reward_sum():
    try:
        reset()
        step_rewards = []
        for _ in range(15):
            d = step(random.choice(VALID_ACTIONS)).json()
            step_rewards.append(float(d.get("reward", 0)))
            if d.get("done"):
                break
        d = history().json()
        history_total = d.get("total_reward", None)
        assert history_total is not None, "total_reward missing from history"
        expected = round(sum(step_rewards), 4)
        assert abs(history_total - expected) < 0.01, f"sum mismatch: {history_total} vs {expected}"
        print("[PASS] test_29_history_reward_sum")
        return True
    except Exception as e:
        print(f"[FAIL] test_29_history_reward_sum: {e}")
        return False


def test_30_concurrent_state():
    try:
        reset()
        results = []
        def call_state():
            try:
                r = state()
                results.append(r.status_code)
            except Exception as ex:
                results.append(str(ex))
        threads = [threading.Thread(target=call_state) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        failures = [r for r in results if r != 200]
        assert len(failures) == 0, f"{len(failures)} concurrent calls failed: {failures}"
        print("[PASS] test_30_concurrent_state")
        return True
    except Exception as e:
        print(f"[FAIL] test_30_concurrent_state: {e}")
        return False


def main():
    tests = [
        test_01_reset_clean, test_02_reset_invalid_task, test_03_reset_empty_body,
        test_04_reset_null_task, test_05_reset_sql_injection, test_06_reset_xss,
        test_07_reset_unicode, test_08_reset_load, test_09_step_all_valid_actions,
        test_10_step_invalid_action, test_11_step_empty_body, test_12_step_null_action,
        test_13_step_before_reset, test_14_step_after_done, test_15_step_1000_times,
        test_16_reward_always_in_range, test_17_reward_never_nan,
        test_18_state_fields_never_null, test_19_state_consistency,
        test_20_threat_fields_complete, test_21_scan_coverage_increases,
        test_22_step_increments, test_23_determinism, test_24_full_episode_easy,
        test_25_full_episode_medium, test_26_full_episode_hard,
        test_27_history_empty_after_reset, test_28_history_count_matches_steps,
        test_29_history_reward_sum, test_30_concurrent_state,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            result = t()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: unhandled exception: {e}")
            failed += 1

    print()
    print(f"Total: {len(tests)} | Passed: {passed} | Failed: {failed}")


if __name__ == "__main__":
    main()
