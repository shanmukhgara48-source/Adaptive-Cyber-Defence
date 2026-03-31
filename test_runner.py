import requests
import threading
import json
import time
import sys
from collections import defaultdict

BASE = "http://127.0.0.1:8000"
results = defaultdict(list)
FAIL_LOG = []


def reset():
    r = requests.post(f"{BASE}/reset")
    assert r.status_code == 200
    return r.json()


def step(action, raw=None):
    if raw is not None:
        return requests.post(f"{BASE}/step", data=raw, headers={"Content-Type": "application/json"})
    return requests.post(f"{BASE}/step", json={"action": action})


def record(cat, name, passed, detail=""):
    results[cat].append((name, passed, detail))
    if not passed:
        FAIL_LOG.append(f"[{cat}] {name}: {detail}")


def assert_valid_json_response(r, name, cat):
    try:
        j = r.json()
        record(cat, name, True)
        return j
    except Exception as e:
        record(cat, name, False, f"Non-JSON response: {r.text[:200]}")
        return None


def assert_no_crash(r, name, cat):
    if r.status_code in (200, 422, 400):
        record(cat, name, True)
    else:
        record(cat, name, False, f"HTTP {r.status_code}: {r.text[:200]}")


# ─── A: INVALID ACTIONS ───────────────────────────────────────────────────────

def test_invalid_actions():
    cat = "A_INVALID_ACTIONS"
    cases = [
        ("empty_string", ""),
        ("whitespace_only", "   "),
        ("single_space", " "),
        ("tab_char", "\t"),
        ("newline_char", "\n"),
        ("long_string_1k", "a" * 1000),
        ("long_string_10k", "x" * 10000),
        ("long_string_100k", "z" * 100000),
        ("unicode_emoji", "😈💀🔥"),
        ("unicode_cjk", "攻击系统"),
        ("unicode_arabic", "هجوم"),
        ("unicode_rtl_override", "\u202eattack"),
        ("null_byte", "action\x00here"),
        ("sql_injection_1", "' OR '1'='1"),
        ("sql_injection_2", "'; DROP TABLE threats; --"),
        ("xss_1", "<script>alert(1)</script>"),
        ("xss_2", "\"><img src=x onerror=alert(1)>"),
        ("path_traversal", "../../etc/passwd"),
        ("shell_injection_1", "; rm -rf /"),
        ("shell_injection_2", "$(whoami)"),
        ("shell_injection_3", "`id`"),
        ("json_injection", '{"action": "block_ip"}'),
        ("number_as_string", "12345"),
        ("float_string", "3.14"),
        ("negative_number", "-999"),
        ("bool_true", "true"),
        ("bool_false", "false"),
        ("none_string", "None"),
        ("null_string", "null"),
        ("scan_no_node", "scan"),
        ("scan_invalid_node", "scan_node_99"),
        ("scan_node_0", "scan_node_0"),
        ("scan_node_negative", "scan_node_-1"),
        ("scan_node_float", "scan_node_1.5"),
        ("scan_node_alpha", "scan_node_abc"),
        ("random_fuzz_1", "a!@#$%^&*()"),
        ("random_fuzz_2", "BLOCK_IP"),
        ("random_fuzz_3", "Block_IP"),
        ("random_fuzz_4", "block ip"),
        ("random_fuzz_5", "blockip"),
        ("random_fuzz_6", "block_ip_extra"),
        ("random_fuzz_7", "isolate_machine_now"),
        ("random_fuzz_8", "patch_it"),
        ("repeated_unknown", "unknown_action_xyz"),
        ("very_long_unicode", "α" * 5000),
        ("format_string", "%s%s%s%d%d%d"),
        ("env_var_injection", "${PATH}"),
        ("template_injection", "{{7*7}}"),
        ("xml_injection", "<foo>&xxe;</foo>"),
        ("null_char_sequence", "\x00\x01\x02\x03"),
        ("high_code_points", "\U0001F4A9" * 500),
    ]
    for name, action in cases:
        reset()
        try:
            r = step(action)
            assert_no_crash(r, f"action_{name}", cat)
            j = r.json()
            has_keys = all(k in j for k in ("system_health", "score", "step", "done"))
            record(cat, f"obs_keys_{name}", has_keys, "Missing obs keys" if not has_keys else "")
        except Exception as e:
            record(cat, f"action_{name}", False, str(e))


# ─── B: API MISUSE ────────────────────────────────────────────────────────────

def test_api_misuse():
    cat = "B_API_MISUSE"

    # Wrong HTTP methods
    for method, path in [("get", "/step"), ("put", "/step"), ("delete", "/step"),
                          ("patch", "/step"), ("put", "/reset"), ("delete", "/reset")]:
        try:
            r = getattr(requests, method)(f"{BASE}{path}", json={"action": "block_ip"})
            record(cat, f"wrong_method_{method}_{path.strip('/')}", r.status_code in (200, 405, 404, 307), f"HTTP {r.status_code}")
        except Exception as e:
            record(cat, f"wrong_method_{method}_{path.strip('/')}", False, str(e))

    # Missing JSON body
    try:
        r = requests.post(f"{BASE}/step")
        record(cat, "missing_body", r.status_code in (200, 422, 400), f"HTTP {r.status_code}")
    except Exception as e:
        record(cat, "missing_body", False, str(e))

    # Empty JSON body
    try:
        r = requests.post(f"{BASE}/step", json={})
        record(cat, "empty_json_body", r.status_code in (200, 422, 400), f"HTTP {r.status_code}")
    except Exception as e:
        record(cat, "empty_json_body", False, str(e))

    # Invalid JSON
    for idx, bad in enumerate(['not json at all', '{action: missing_quotes}', '{"action":}',
                                '{"action": null}', '[]', 'null', '0', '"string"']):
        try:
            r = requests.post(f"{BASE}/step", data=bad, headers={"Content-Type": "application/json"})
            record(cat, f"invalid_json_{idx}", r.status_code in (200, 422, 400), f"HTTP {r.status_code}: {r.text[:100]}")
        except Exception as e:
            record(cat, f"invalid_json_{idx}", False, str(e))

    # Wrong field name
    try:
        r = requests.post(f"{BASE}/step", json={"act": "block_ip"})
        record(cat, "wrong_field_name", r.status_code in (200, 422, 400), f"HTTP {r.status_code}")
    except Exception as e:
        record(cat, "wrong_field_name", False, str(e))

    # Extra fields
    try:
        r = requests.post(f"{BASE}/step", json={"action": "block_ip", "extra": "field", "evil": "<script>"})
        assert_no_crash(r, "extra_fields", cat)
    except Exception as e:
        record(cat, "extra_fields", False, str(e))

    # Wrong type for action
    for idx, val in enumerate([123, 3.14, True, False, None, [], {}, [1, 2], {"a": 1}]):
        try:
            r = requests.post(f"{BASE}/step", json={"action": val})
            record(cat, f"wrong_action_type_{idx}", r.status_code in (200, 422, 400), f"HTTP {r.status_code}")
        except Exception as e:
            record(cat, f"wrong_action_type_{idx}", False, str(e))

    # Large payloads
    try:
        r = requests.post(f"{BASE}/step", json={"action": "block_ip", "data": "x" * 1_000_000})
        assert_no_crash(r, "large_payload_1mb", cat)
    except Exception as e:
        record(cat, "large_payload_1mb", False, str(e))

    try:
        r = requests.post(f"{BASE}/step", json={"action": "block_ip", "nested": {"k": "v" * 100000}})
        assert_no_crash(r, "large_nested_payload", cat)
    except Exception as e:
        record(cat, "large_nested_payload", False, str(e))

    # Deeply nested JSON
    def deep_nest(depth):
        d = {"action": "block_ip"}
        for _ in range(depth):
            d = {"level": d}
        return d

    try:
        r = requests.post(f"{BASE}/step", json=deep_nest(500))
        assert_no_crash(r, "deep_nested_json_500", cat)
    except Exception as e:
        record(cat, "deep_nested_json_500", False, str(e))

    # Concurrent requests
    errors = []
    def concurrent_step(i):
        try:
            r = requests.post(f"{BASE}/step", json={"action": "block_ip"}, timeout=5)
            if r.status_code not in (200, 422, 400):
                errors.append(f"Thread {i}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"Thread {i}: {e}")

    reset()
    threads = [threading.Thread(target=concurrent_step, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()
    record(cat, "concurrent_20_requests", len(errors) == 0, "; ".join(errors))


# ─── C: STATE BREAKERS ────────────────────────────────────────────────────────

def test_state_breakers():
    cat = "C_STATE_BREAKERS"

    # Step before reset (fresh state is already initialized)
    try:
        r = step("block_ip")
        assert_no_crash(r, "step_before_explicit_reset", cat)
    except Exception as e:
        record(cat, "step_before_explicit_reset", False, str(e))

    # Repeated reset
    for i in range(20):
        try:
            r = reset()
            has_keys = all(k in r for k in ("system_health", "score", "step", "done"))
            if not has_keys:
                record(cat, f"repeated_reset_{i}", False, "Missing keys")
                break
        except Exception as e:
            record(cat, f"repeated_reset_{i}", False, str(e))
            break
    else:
        record(cat, "repeated_reset_20", True)

    # Step after done=True
    reset()
    for _ in range(60):
        r = step("ignore")
        j = r.json()
        if j.get("done"):
            break
    # Now step after done
    for i in range(5):
        try:
            r = step("block_ip")
            j = r.json()
            record(cat, f"step_after_done_{i}", r.status_code == 200, f"HTTP {r.status_code}")
            valid = all(k in j for k in ("system_health", "score", "step", "done"))
            record(cat, f"step_after_done_valid_obs_{i}", valid, "Missing obs keys")
        except Exception as e:
            record(cat, f"step_after_done_{i}", False, str(e))

    # Health underflow: hammer with ignore to push health below 0
    reset()
    for _ in range(30):
        r = step("ignore")
        j = r.json()
        h = j.get("system_health", -999)
        if h < 0:
            record(cat, "health_underflow", False, f"health={h}")
            break
    else:
        record(cat, "health_underflow", True)

    # Score overflow stress
    reset()
    for _ in range(200):
        step("block_ip")
    r = requests.get(f"{BASE}/state")
    j = r.json()
    score = j.get("score", 0)
    record(cat, "score_finite", isinstance(score, (int, float)) and score == score, f"score={score}")  # NaN check

    # Health never exceeds 100
    reset()
    r = requests.get(f"{BASE}/state")
    j = r.json()
    h = j.get("system_health", -1)
    record(cat, "health_max_100_after_reset", h <= 100, f"health={h}")

    # Done stays True after hitting it
    reset()
    for _ in range(60):
        r = step("ignore")
        j = r.json()
        if j.get("done"):
            break
    done_val = j.get("done")
    step("block_ip")
    r2 = requests.get(f"{BASE}/state")
    j2 = r2.json()
    record(cat, "done_persists", j2.get("done") == True, f"done={j2.get('done')}")

    # Negative step count impossible
    reset()
    r = requests.get(f"{BASE}/state")
    j = r.json()
    record(cat, "step_non_negative_after_reset", j.get("step", -1) >= 0, f"step={j.get('step')}")


# ─── D: PARTIAL OBSERVABILITY ─────────────────────────────────────────────────

def test_partial_observability():
    cat = "D_PARTIAL_OBS"

    # Scan same node repeatedly
    reset()
    for i in range(10):
        try:
            r = step("scan_node_1")
            assert_no_crash(r, f"scan_same_node_{i}", cat)
        except Exception as e:
            record(cat, f"scan_same_node_{i}", False, str(e))

    # Scan all valid nodes
    reset()
    for n in range(1, 6):
        try:
            r = step(f"scan_node_{n}")
            assert_no_crash(r, f"scan_valid_node_{n}", cat)
        except Exception as e:
            record(cat, f"scan_valid_node_{n}", False, str(e))

    # scan_coverage should be 1.0 after scanning all
    r = requests.get(f"{BASE}/state")
    j = r.json()
    record(cat, "scan_coverage_full", j.get("scan_coverage") == 1.0, f"coverage={j.get('scan_coverage')}")

    # Scan invalid nodes
    reset()
    for name, node in [("node_6", "scan_node_6"), ("node_100", "scan_node_100"),
                        ("node_neg", "scan_node_-1"), ("node_0", "scan_node_0"),
                        ("node_float", "scan_node_1.5"), ("node_alpha", "scan_node_abc"),
                        ("bare_scan", "scan"), ("scan_space", "scan node 1")]:
        try:
            r = step(node)
            assert_no_crash(r, f"invalid_scan_{name}", cat)
        except Exception as e:
            record(cat, f"invalid_scan_{name}", False, str(e))

    # No scan ever: hidden_node_count should be TOTAL_NODES
    reset()
    r = requests.get(f"{BASE}/state")
    j = r.json()
    record(cat, "no_scan_hidden_count", j.get("hidden_node_count") == 5, f"hidden={j.get('hidden_node_count')}")
    record(cat, "no_scan_coverage_zero", j.get("scan_coverage") == 0.0, f"coverage={j.get('scan_coverage')}")

    # Visibility consistency: visible_threats must only contain scanned or aged threats
    reset()
    r = requests.get(f"{BASE}/state")
    j = r.json()
    # Initially no scans, step=0, so visible_threats should be empty or only aged ones
    visible = j.get("visible_threats", [])
    record(cat, "initial_visible_threats_valid", isinstance(visible, list), f"type={type(visible)}")

    # hidden_node_count + scanned == TOTAL_NODES
    reset()
    step("scan_node_1")
    step("scan_node_2")
    r = requests.get(f"{BASE}/state")
    j = r.json()
    coverage = j.get("scan_coverage", -1)
    hidden = j.get("hidden_node_count", -1)
    record(cat, "coverage_hidden_consistent", abs(coverage - 0.4) < 0.01, f"coverage={coverage}")
    record(cat, "hidden_count_after_2_scans", hidden == 3, f"hidden={hidden}")


# ─── E: REWARD EDGE CASES ─────────────────────────────────────────────────────

def test_reward_edge_cases():
    cat = "E_REWARD"

    # Reward in valid range
    reset()
    for i in range(30):
        r = step("block_ip")
        j = r.json()
        rwd = j.get("reward", None)
        if rwd is not None:
            if not (-10 <= rwd <= 10):
                record(cat, f"reward_range_step_{i}", False, f"reward={rwd}")
                break
    else:
        record(cat, "reward_range_all_valid", True)

    # Reward is finite (not NaN, not Inf)
    reset()
    for i in range(10):
        r = step("ignore")
        j = r.json()
        rwd = j.get("reward", 0)
        if rwd != rwd or rwd == float('inf') or rwd == float('-inf'):
            record(cat, f"reward_finite_{i}", False, f"reward={rwd}")
            break
    else:
        record(cat, "reward_finite_all", True)

    # Score is finite
    reset()
    for _ in range(50):
        step("patch")
    r = requests.get(f"{BASE}/state")
    j = r.json()
    score = j.get("score", 0)
    record(cat, "score_finite", score == score and score not in (float('inf'), float('-inf')), f"score={score}")

    # Reward returned on every step
    reset()
    for i, action in enumerate(["block_ip", "isolate_machine", "patch", "ignore", "scan_node_1", "xyz"]):
        r = step(action)
        j = r.json()
        record(cat, f"reward_present_{i}_{action}", "reward" in j, f"keys={list(j.keys())}")


# ─── F: MULTI-THREAT CHAOS ────────────────────────────────────────────────────

def test_multi_threat_chaos():
    cat = "F_MULTI_THREAT"

    # Rapid sequential defense actions
    reset()
    actions = ["block_ip", "isolate_machine", "patch", "ignore"] * 30
    for i, action in enumerate(actions):
        try:
            r = step(action)
            assert_no_crash(r, f"rapid_seq_{i}", cat)
        except Exception as e:
            record(cat, f"rapid_seq_{i}", False, str(e))
            break
    record(cat, "rapid_120_sequential", True)

    # All valid actions cycle
    reset()
    valid_actions = ["block_ip", "isolate_machine", "patch", "ignore",
                     "scan_node_1", "scan_node_2", "scan_node_3", "scan_node_4", "scan_node_5"]
    for i, action in enumerate(valid_actions * 5):
        try:
            r = step(action)
            j = r.json()
            h = j.get("system_health", -1)
            if h < 0:
                record(cat, f"health_negative_action_{i}", False, f"health={h}")
                break
        except Exception as e:
            record(cat, f"cycle_actions_{i}", False, str(e))
            break
    else:
        record(cat, "cycle_all_valid_actions", True)

    # Alternating invalid/valid
    reset()
    for i in range(20):
        try:
            r1 = step("$$INVALID$$")
            r2 = step("block_ip")
            assert_no_crash(r1, f"alt_invalid_{i}", cat)
            assert_no_crash(r2, f"alt_valid_{i}", cat)
        except Exception as e:
            record(cat, f"alternating_{i}", False, str(e))
            break
    record(cat, "alternating_invalid_valid_20", True)


# ─── G: PERFORMANCE STRESS ────────────────────────────────────────────────────

def test_performance():
    cat = "G_PERFORMANCE"

    # 200 rapid /step calls
    reset()
    errors = []
    start = time.time()
    for i in range(200):
        try:
            r = step("block_ip")
            if r.status_code not in (200,):
                errors.append(f"step {i}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"step {i}: {e}")
    elapsed = time.time() - start
    record(cat, "200_rapid_steps_no_crash", len(errors) == 0, "; ".join(errors[:5]))
    record(cat, "200_rapid_steps_timing", elapsed < 30, f"took {elapsed:.1f}s")

    # 50 resets in loop
    errors = []
    for i in range(50):
        try:
            r = reset()
            if not all(k in r for k in ("system_health", "score", "step", "done")):
                errors.append(f"reset {i}: missing keys")
        except Exception as e:
            errors.append(f"reset {i}: {e}")
    record(cat, "50_resets_no_error", len(errors) == 0, "; ".join(errors[:5]))

    # Interleaved reset + step
    errors = []
    for i in range(30):
        try:
            reset()
            for _ in range(5):
                r = step("block_ip")
                if r.status_code != 200:
                    errors.append(f"iter {i}: HTTP {r.status_code}")
                    break
        except Exception as e:
            errors.append(f"iter {i}: {e}")
    record(cat, "interleaved_reset_step_30x5", len(errors) == 0, "; ".join(errors[:5]))

    # Concurrent resets
    errors = []
    def do_reset(i):
        try:
            r = requests.post(f"{BASE}/reset", timeout=5)
            if r.status_code != 200:
                errors.append(f"Thread {i}: HTTP {r.status_code}")
        except Exception as e:
            errors.append(f"Thread {i}: {e}")

    threads = [threading.Thread(target=do_reset, args=(i,)) for i in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()
    record(cat, "concurrent_10_resets", len(errors) == 0, "; ".join(errors))

    # State always valid after stress
    r = requests.get(f"{BASE}/state")
    j = r.json()
    valid = all(k in j for k in ("visible_threats", "system_health", "score", "step", "done",
                                  "hidden_node_count", "scan_coverage"))
    record(cat, "state_valid_after_stress", valid, f"keys={list(j.keys())}")


# ─── H: SECURITY TESTS ────────────────────────────────────────────────────────

def test_security():
    cat = "H_SECURITY"

    injections = [
        ("sqli_1", "' OR 1=1 --"),
        ("sqli_2", "'; DROP TABLE state; --"),
        ("sqli_3", "1; SELECT * FROM users"),
        ("xss_1", "<script>alert('xss')</script>"),
        ("xss_2", "<img src=x onerror=alert(1)>"),
        ("xss_3", "javascript:alert(1)"),
        ("path_trav_1", "../../../../etc/passwd"),
        ("path_trav_2", "..\\..\\..\\windows\\system32"),
        ("shell_1", "; cat /etc/passwd"),
        ("shell_2", "| ls -la"),
        ("shell_3", "&& whoami"),
        ("shell_4", "`uname -a`"),
        ("shell_5", "$(cat /etc/shadow)"),
        ("null_byte_1", "block_ip\x00; rm -rf /"),
        ("null_byte_2", "\x00\x00\x00"),
        ("unicode_attack_1", "\u0000\u0001\u0002"),
        ("unicode_attack_2", "\ufeff\ufffe"),
        ("unicode_attack_3", "\u202e\u202d"),
        ("format_str_1", "%n%n%n%n"),
        ("format_str_2", "%x%x%x%x"),
        ("ldap_inject", "*()|%26"),
        ("xml_inject", "<?xml version='1.0'?><!DOCTYPE root [<!ENTITY test SYSTEM 'file:///etc/passwd'>]>"),
        ("json_proto_pollute", "__proto__"),
        ("json_constructor", "constructor"),
        ("regex_dos", "a" * 100 + "!" * 100),
    ]

    for name, payload in injections:
        reset()
        try:
            r = step(payload)
            record(cat, f"inject_{name}_no_crash", r.status_code in (200, 400, 422), f"HTTP {r.status_code}")
            try:
                j = r.json()
                record(cat, f"inject_{name}_returns_json", True)
            except Exception:
                record(cat, f"inject_{name}_returns_json", False, f"Non-JSON: {r.text[:100]}")
        except Exception as e:
            record(cat, f"inject_{name}_no_crash", False, str(e))

    # Verify no sensitive data leaked in responses
    reset()
    r = step("'; DROP TABLE state; --")
    try:
        j = r.json()
        text = json.dumps(j).lower()
        sensitive_leaked = any(s in text for s in ["passwd", "shadow", "secret", "traceback", "exception", "error stack"])
        record(cat, "no_sensitive_data_leak", not sensitive_leaked, f"Possible leak in: {text[:200]}")
    except Exception:
        record(cat, "no_sensitive_data_leak", False, "Non-JSON response")


# ─── ENDPOINT HEALTH CHECKS ───────────────────────────────────────────────────

def test_endpoints():
    cat = "Z_ENDPOINTS"

    # GET /
    try:
        r = requests.get(f"{BASE}/")
        record(cat, "root_get", r.status_code == 200, f"HTTP {r.status_code}")
        r.json()
        record(cat, "root_returns_json", True)
    except Exception as e:
        record(cat, "root_get", False, str(e))

    # GET /state
    try:
        r = requests.get(f"{BASE}/state")
        record(cat, "state_get", r.status_code == 200, f"HTTP {r.status_code}")
        j = r.json()
        required = ["visible_threats", "hidden_node_count", "scan_coverage",
                    "system_health", "score", "step", "done"]
        for key in required:
            record(cat, f"state_has_{key}", key in j, f"missing in {list(j.keys())}")
    except Exception as e:
        record(cat, "state_get", False, str(e))

    # POST /reset
    try:
        r = requests.post(f"{BASE}/reset")
        record(cat, "reset_post", r.status_code == 200, f"HTTP {r.status_code}")
        j = r.json()
        record(cat, "reset_health_100", j.get("system_health") == 100, f"health={j.get('system_health')}")
        record(cat, "reset_score_0", j.get("score") == 0, f"score={j.get('score')}")
        record(cat, "reset_step_0", j.get("step") == 0, f"step={j.get('step')}")
        record(cat, "reset_done_false", j.get("done") == False, f"done={j.get('done')}")
    except Exception as e:
        record(cat, "reset_post", False, str(e))

    # POST /step valid
    reset()
    try:
        r = requests.post(f"{BASE}/step", json={"action": "block_ip"})
        record(cat, "step_post_valid", r.status_code == 200, f"HTTP {r.status_code}")
        j = r.json()
        record(cat, "step_has_reward", "reward" in j, f"keys={list(j.keys())}")
        record(cat, "step_has_action", "action" in j, f"keys={list(j.keys())}")
    except Exception as e:
        record(cat, "step_post_valid", False, str(e))


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run_all():
    print("=" * 70)
    print("ADAPTIVE CYBER DEFENSE — STRESS TEST SUITE")
    print("=" * 70)

    # Check server is up
    try:
        r = requests.get(f"{BASE}/", timeout=3)
        print(f"Server UP at {BASE}\n")
    except Exception:
        print(f"ERROR: Server not reachable at {BASE}")
        print("Start with: uvicorn app:app --host 127.0.0.1 --port 8000")
        sys.exit(1)

    suites = [
        ("A: Invalid Actions", test_invalid_actions),
        ("B: API Misuse", test_api_misuse),
        ("C: State Breakers", test_state_breakers),
        ("D: Partial Observability", test_partial_observability),
        ("E: Reward Edge Cases", test_reward_edge_cases),
        ("F: Multi-Threat Chaos", test_multi_threat_chaos),
        ("G: Performance Stress", test_performance),
        ("H: Security Tests", test_security),
        ("Z: Endpoint Health", test_endpoints),
    ]

    total_pass = 0
    total_fail = 0

    for label, fn in suites:
        print(f"\n{'─'*60}")
        print(f"  {label}")
        print(f"{'─'*60}")
        try:
            fn()
        except Exception as e:
            print(f"  [SUITE CRASH] {e}")

    # Print results per category
    for cat, tests in sorted(results.items()):
        passed = sum(1 for _, p, _ in tests if p)
        failed = sum(1 for _, p, _ in tests if not p)
        total_pass += passed
        total_fail += failed
        status = "OK" if failed == 0 else "FAIL"
        print(f"\n[{status}] {cat}: {passed}/{passed+failed} passed")
        for name, passed_, detail in tests:
            if not passed_:
                print(f"      FAIL  {name}: {detail}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {total_pass} PASSED  |  {total_fail} FAILED")
    print("=" * 70)

    if FAIL_LOG:
        print("\nFAILURE SUMMARY:")
        for line in FAIL_LOG:
            print(f"  - {line}")

    return total_fail == 0


if __name__ == "__main__":
    ok = run_all()
    sys.exit(0 if ok else 1)
