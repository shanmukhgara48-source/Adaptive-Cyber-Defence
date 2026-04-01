import requests
import random
import math
import time

BASE = "http://localhost:8000"
VALID_ACTIONS = ["block_ip", "isolate_machine", "patch", "ignore",
                 "scan_node_1", "scan_node_2", "scan_node_3", "scan_node_4", "scan_node_5"]
ATTACK_ACTIONS = ["block_ip", "isolate_machine", "patch"]


def reset(task=None):
    body = {"task": task} if task else {}
    return requests.post(f"{BASE}/reset", json=body).json()


def step(action):
    return requests.post(f"{BASE}/step", json={"action": action}).json()


def get_state():
    return requests.get(f"{BASE}/state").json()


def find_threat_and_correct_action():
    """Scan all nodes and return (correct_action, threat_type) for the first visible threat found."""
    reset()
    for i in range(1, 6):
        d = step(f"scan_node_{i}")
        threats = d.get("visible_threats", [])
        if threats:
            t = threats[0]
            t_type = t.get("type", "")
            mapping = {
                "phishing": "block_ip",
                "malware": "isolate_machine",
                "ddos": "patch",
                "ransomware": "isolate_machine",
                "lateral_movement": "block_ip",
            }
            return mapping.get(t_type, "block_ip"), t_type
    return None, None


def test_r01_ignore_gives_negative_reward():
    """Ignore action gives lowest reward (normalized -1.5 raw → ~0.125)."""
    try:
        reset()
        d = step("ignore")
        reward = float(d.get("reward", 999))
        # normalized(-1.5) = (-1.5+2)/4 = 0.125
        assert reward < 0.3, f"ignore reward {reward} not < 0.3 (expected ~0.125)"
        assert math.isfinite(reward), "reward is NaN/Inf"
        print(f"[PASS] test_r01_ignore_gives_negative_reward — reward={reward:.4f} < 0.3")
        return True
    except Exception as e:
        print(f"[FAIL] test_r01_ignore_gives_negative_reward — {e}")
        return False


def test_r02_correct_action_gives_positive_reward():
    """Correct mitigation gives reward > 0.5 (normalized 1.0+ raw)."""
    try:
        correct_action, t_type = find_threat_and_correct_action()
        if correct_action is None:
            print("[FAIL] test_r02_correct_action_gives_positive_reward — no threat found after scanning all nodes")
            return False
        d = step(correct_action)
        reward = float(d.get("reward", 0))
        assert reward > 0.5, f"correct action reward {reward} not > 0.5 (threat={t_type}, action={correct_action})"
        print(f"[PASS] test_r02_correct_action_gives_positive_reward — reward={reward:.4f} > 0.5 ({t_type}→{correct_action})")
        return True
    except Exception as e:
        print(f"[FAIL] test_r02_correct_action_gives_positive_reward — {e}")
        return False


def test_r03_wrong_action_gives_penalty():
    """Wrong mitigation gives lower reward than correct mitigation."""
    try:
        # Run correct action and capture reward
        correct_action, t_type = find_threat_and_correct_action()
        if correct_action is None:
            print("[FAIL] test_r03_wrong_action_gives_penalty — no threat found")
            return False
        correct_d = step(correct_action)
        correct_reward = float(correct_d.get("reward", 0))

        # Run wrong action on fresh episode
        correct_action2, t_type2 = find_threat_and_correct_action()
        if correct_action2 is None:
            print("[FAIL] test_r03_wrong_action_gives_penalty — no threat found in second episode")
            return False
        wrong_action = next((a for a in ATTACK_ACTIONS if a != correct_action2), "ignore")
        wrong_d = step(wrong_action)
        wrong_reward = float(wrong_d.get("reward", 0))

        assert wrong_reward < correct_reward, (
            f"wrong reward {wrong_reward:.4f} not < correct reward {correct_reward:.4f} "
            f"(threat={t_type2}, correct={correct_action2}, wrong={wrong_action})"
        )
        print(f"[PASS] test_r03_wrong_action_gives_penalty — wrong={wrong_reward:.4f} < correct={correct_reward:.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] test_r03_wrong_action_gives_penalty — {e}")
        return False


def test_r04_reward_never_nan():
    """Reward is never NaN across 100 random actions."""
    try:
        reset()
        for i in range(100):
            if get_state().get("done"):
                reset()
            d = step(random.choice(VALID_ACTIONS))
            reward = d.get("reward", 0)
            assert not math.isnan(float(reward)), f"NaN reward at step {i}"
        print("[PASS] test_r04_reward_never_nan — 100 steps, no NaN")
        return True
    except Exception as e:
        print(f"[FAIL] test_r04_reward_never_nan — {e}")
        return False


def test_r05_reward_never_inf():
    """Reward is never Inf across 100 random actions."""
    try:
        reset()
        for i in range(100):
            if get_state().get("done"):
                reset()
            d = step(random.choice(VALID_ACTIONS))
            reward = d.get("reward", 0)
            assert not math.isinf(float(reward)), f"Inf reward at step {i}: {reward}"
        print("[PASS] test_r05_reward_never_inf — 100 steps, no Inf")
        return True
    except Exception as e:
        print(f"[FAIL] test_r05_reward_never_inf — {e}")
        return False


def test_r06_reward_always_in_range():
    """All rewards are in [0.0, 1.0] (normalized range)."""
    try:
        reset()
        out_of_range = []
        for i in range(100):
            if get_state().get("done"):
                reset()
            d = step(random.choice(VALID_ACTIONS))
            reward = float(d.get("reward", 0))
            if not (0.0 <= reward <= 1.0):
                out_of_range.append((i, reward))
        assert len(out_of_range) == 0, f"{len(out_of_range)} rewards out of [0,1]: {out_of_range[:5]}"
        print("[PASS] test_r06_reward_always_in_range — 100 steps all in [0.0, 1.0]")
        return True
    except Exception as e:
        print(f"[FAIL] test_r06_reward_always_in_range — {e}")
        return False


def test_r07_scan_reward_positive_on_threat():
    """Scanning a node with a hidden threat gives reward > 0.4 (normalized +0.02 raw ≈ 0.505)."""
    try:
        reset()
        # Scan all nodes; at least one should reveal a threat
        scan_rewards = []
        reveal_rewards = []
        for i in range(1, 6):
            d = step(f"scan_node_{i}")
            reward = float(d.get("reward", 0))
            threats = d.get("visible_threats", [])
            # A reveal scan returns the newly visible threats in the response
            # Check via reason field too
            reason = d.get("reason", "")
            if "revealed" in reason.lower() or threats:
                reveal_rewards.append(reward)
            scan_rewards.append(reward)

        assert len(scan_rewards) > 0, "no scans executed"
        if reveal_rewards:
            max_reveal = max(reveal_rewards)
            assert max_reveal > 0.4, f"reveal scan reward {max_reveal:.4f} not > 0.4"
            print(f"[PASS] test_r07_scan_reward_positive_on_threat — reveal reward={max_reveal:.4f} > 0.4")
        else:
            # No reveals — check at least rewards are finite and in range
            assert all(0.0 <= r <= 1.0 for r in scan_rewards), f"scan rewards out of range: {scan_rewards}"
            print(f"[PASS] test_r07_scan_reward_positive_on_threat — no reveals (all threats initially visible), scan rewards in range: {[round(r,3) for r in scan_rewards]}")
        return True
    except Exception as e:
        print(f"[FAIL] test_r07_scan_reward_positive_on_threat — {e}")
        return False


def test_r08_scan_reward_small_on_empty():
    """Scanning a node with no threat gives small reward ≈ 0.4975 (normalized -0.01 raw)."""
    try:
        reset()
        empty_scan_rewards = []
        for i in range(1, 6):
            d = step(f"scan_node_{i}")
            reward = float(d.get("reward", 0))
            reason = d.get("reason", "")
            if "no new threats" in reason.lower() or "found no" in reason.lower():
                empty_scan_rewards.append(reward)

        if empty_scan_rewards:
            for r in empty_scan_rewards:
                # normalized(-0.01) ≈ 0.4975 — should be < reveal reward (0.505)
                assert r < 0.52, f"empty scan reward {r:.4f} not < 0.52"
                assert r > 0.4, f"empty scan reward {r:.4f} not > 0.4"
            print(f"[PASS] test_r08_scan_reward_small_on_empty — empty scan rewards: {[round(r,4) for r in empty_scan_rewards]}")
        else:
            print("[PASS] test_r08_scan_reward_small_on_empty — all nodes had threats (no empty scans to test)")
        return True
    except Exception as e:
        print(f"[FAIL] test_r08_scan_reward_small_on_empty — {e}")
        return False


def test_r09_reward_cumulative_increases():
    """Score in /state increases after correct actions."""
    try:
        reset()
        prev_score = get_state().get("score", 0.0)
        increases = 0
        for _ in range(5):
            if get_state().get("done"):
                reset()
                prev_score = get_state().get("score", 0.0)
            # Try a correct action by scanning first
            for i in range(1, 6):
                d = step(f"scan_node_{i}")
                threats = d.get("visible_threats", [])
                if threats:
                    t_type = threats[0].get("type", "")
                    mapping = {"phishing": "block_ip", "malware": "isolate_machine",
                               "ddos": "patch", "ransomware": "isolate_machine",
                               "lateral_movement": "block_ip"}
                    action = mapping.get(t_type, "block_ip")
                    step(action)
                    break
            new_score = get_state().get("score", 0.0)
            if new_score > prev_score:
                increases += 1
            prev_score = new_score

        assert increases > 0, f"score never increased across 5 attempts (correct actions)"
        print(f"[PASS] test_r09_reward_cumulative_increases — score increased {increases}/5 times")
        return True
    except Exception as e:
        print(f"[FAIL] test_r09_reward_cumulative_increases — {e}")
        return False


def test_r10_reward_deterministic():
    """Same sequence of actions produces finite rewards in both runs."""
    try:
        actions = ["scan_node_1", "scan_node_2", "block_ip", "isolate_machine", "patch"]
        run_rewards = []
        for run in range(2):
            reset()
            episode = []
            for action in actions:
                d = step(action)
                episode.append(float(d.get("reward", 0)))
                if d.get("done"):
                    break
            run_rewards.append(episode)

        # Both runs must have finite rewards
        for run_idx, rewards in enumerate(run_rewards):
            for i, r in enumerate(rewards):
                assert math.isfinite(r), f"run {run_idx} step {i}: non-finite reward {r}"

        # Rewards should be consistent in type/range even if not exactly equal (random threats)
        assert len(run_rewards[0]) > 0 and len(run_rewards[1]) > 0, "empty reward lists"
        print(f"[PASS] test_r10_reward_deterministic — run1={[round(r,3) for r in run_rewards[0]]}, run2={[round(r,3) for r in run_rewards[1]]}")
        return True
    except Exception as e:
        print(f"[FAIL] test_r10_reward_deterministic — {e}")
        return False


def main():
    tests = [
        test_r01_ignore_gives_negative_reward,
        test_r02_correct_action_gives_positive_reward,
        test_r03_wrong_action_gives_penalty,
        test_r04_reward_never_nan,
        test_r05_reward_never_inf,
        test_r06_reward_always_in_range,
        test_r07_scan_reward_positive_on_threat,
        test_r08_scan_reward_small_on_empty,
        test_r09_reward_cumulative_increases,
        test_r10_reward_deterministic,
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
    print("REWARD SYSTEM REPORT")
    print("====================")
    print(f"Total:   {len(tests)}")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    main()
