"""
verify_openenv_compliance.py — OpenEnv spec compliance checker.
Run from the project root (Documents/) or from within the package dir.
"""

import sys
import os

# Ensure both the package parent and package itself are on sys.path
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_PARENT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_compliance():
    print("=== OpenEnv Compliance Check ===\n")
    errors = []

    # ── Test 1: openenv.yaml exists and has required fields ─────────────────
    try:
        import yaml
        yaml_path = os.path.join(_HERE, "openenv.yaml")
        with open(yaml_path) as f:
            spec = yaml.safe_load(f)
        required = ["name", "version", "description", "tasks",
                    "observation_space", "action_space", "reward"]
        for field in required:
            if field not in spec:
                errors.append(f"openenv.yaml missing field: {field}")
        tasks = spec.get("tasks", [])
        if len(tasks) < 3:
            errors.append(f"Need 3+ tasks, found {len(tasks)}")
        for t in tasks:
            score = t.get("passing_score", 0)
            if not (0.0 <= score <= 1.0):
                errors.append(f"Task {t.get('id')} passing_score out of range")
        print(f"[{'PASS' if not [e for e in errors if 'openenv' in e.lower() or 'task' in e.lower()] else 'FAIL'}] openenv.yaml structure")
    except ImportError:
        # yaml not installed — try basic check
        try:
            with open(os.path.join(_HERE, "openenv.yaml")) as f:
                content = f.read()
            for field in ["name", "version", "description", "tasks"]:
                if field not in content:
                    errors.append(f"openenv.yaml missing field: {field}")
            print("[PASS] openenv.yaml structure (basic check — install pyyaml for full check)")
        except Exception as e:
            errors.append(f"openenv.yaml error: {e}")
            print(f"[FAIL] openenv.yaml: {e}")
    except Exception as e:
        errors.append(f"openenv.yaml error: {e}")
        print(f"[FAIL] openenv.yaml: {e}")

    # ── Test 2: CyberDefenseEnv imports and instantiates ────────────────────
    try:
        from adaptive_cyber_defense.environment import CyberDefenseEnv
        env = CyberDefenseEnv(task="easy", seed=42)
        print("[PASS] Environment imports and instantiates")
    except Exception as e:
        errors.append(f"Environment import failed: {e}")
        print(f"[FAIL] Environment import: {e}")
        print("\nAll errors so far:")
        for err in errors:
            print(f"  - {err}")
        return False

    # ── Test 3: reset() returns correct structure ────────────────────────────
    try:
        obs = env.reset()
        assert isinstance(obs, dict), f"reset() must return dict, got {type(obs)}"
        required_keys = ["active_threats", "network_state", "resources", "step", "score"]
        for k in required_keys:
            assert k in obs, f"reset() missing key: {k}"
        assert isinstance(obs["active_threats"], list), "active_threats must be list"
        assert isinstance(obs["network_state"], dict), "network_state must be dict"
        assert isinstance(obs["resources"], dict), "resources must be dict"
        assert obs["step"] == 0, f"step must be 0 on reset, got {obs['step']}"
        assert obs["score"] == 0.0, f"score must be 0.0 on reset, got {obs['score']}"
        print("[PASS] reset() returns correct structure")
    except Exception as e:
        errors.append(f"reset() failed: {e}")
        print(f"[FAIL] reset(): {e}")

    # ── Test 4: step() returns correct 4-tuple ───────────────────────────────
    try:
        from adaptive_cyber_defense.models.action import Action
        action = Action.SCAN
        result = env.step(action)
        assert isinstance(result, tuple), f"step() must return tuple, got {type(result)}"
        assert len(result) == 4, f"step() must return 4 items, got {len(result)}"
        obs2, reward, done, info = result
        assert isinstance(obs2, dict), f"observation must be dict, got {type(obs2)}"
        assert isinstance(reward, float), f"reward must be float, got {type(reward)}"
        assert isinstance(done, bool), f"done must be bool, got {type(done)}"
        assert isinstance(info, dict), f"info must be dict, got {type(info)}"
        assert -1.0 <= reward <= 1.0, f"reward {reward} out of range [-1, 1]"
        for k in ["active_threats", "network_state", "resources", "step", "score"]:
            assert k in obs2, f"step() obs missing key: {k}"
        print("[PASS] step() returns correct (obs, reward, done, info) tuple")
    except Exception as e:
        errors.append(f"step() failed: {e}")
        print(f"[FAIL] step(): {e}")

    # ── Test 5: state() returns correct structure ────────────────────────────
    try:
        state = env.state()
        assert isinstance(state, dict), f"state() must return dict, got {type(state)}"
        assert "observation" in state, "state() missing 'observation' key"
        assert "step" in state, "state() missing 'step' key"
        assert "score" in state, "state() missing 'score' key"
        assert "done" in state, "state() missing 'done' key"
        print("[PASS] state() returns correct structure")
    except Exception as e:
        errors.append(f"state() failed: {e}")
        print(f"[FAIL] state(): {e}")

    # ── Test 6: All 3 tasks runnable ─────────────────────────────────────────
    from adaptive_cyber_defense.models.action import Action
    for task in ["easy", "medium", "hard"]:
        try:
            env2 = CyberDefenseEnv(task=task, seed=0)
            obs = env2.reset()
            assert isinstance(obs, dict)
            result = env2.step(Action.SCAN)
            assert len(result) == 4
            print(f"[PASS] Task '{task}' runs correctly")
        except Exception as e:
            errors.append(f"Task '{task}' failed: {e}")
            print(f"[FAIL] Task '{task}': {e}")

    # ── Test 7: Reward always in [-1.0, 1.0] ────────────────────────────────
    try:
        env3 = CyberDefenseEnv(task="easy", seed=1)
        env3.reset()
        for _ in range(10):
            obs, r, done, info = env3.step(Action.SCAN)
            assert -1.0 <= r <= 1.0, f"reward {r} out of range"
            if done:
                break
        print("[PASS] Reward always in [-1.0, 1.0] range")
    except Exception as e:
        errors.append(f"Reward range failed: {e}")
        print(f"[FAIL] Reward range: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    if errors:
        print(f"FAILED — {len(errors)} errors found:")
        for e in errors:
            print(f"  - {e}")
        return False
    else:
        print("ALL CHECKS PASSED — fully OpenEnv compliant!")
        return True


if __name__ == "__main__":
    ok = test_compliance()
    sys.exit(0 if ok else 1)
