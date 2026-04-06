import os
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

BASE_URL     = os.getenv("BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
API_KEY      = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("API_KEY")

# Defer validation to run-time so that `python inference.py` (without env vars)
# prints a clear usage message and exits cleanly rather than crashing with a
# raw traceback at import time.
def _check_env_vars() -> bool:
    """Return True if all required env vars are present, False otherwise."""
    missing = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not API_KEY:
        missing.append("HF_TOKEN / OPENAI_API_KEY / API_KEY")
    if missing:
        print("ERROR: Missing required environment variable(s):")
        for m in missing:
            print(f"  - {m}")
        print(
            "\nUsage:\n"
            "  export API_BASE_URL=<openai-compatible-endpoint>\n"
            "  export MODEL_NAME=<model-id>\n"
            "  export HF_TOKEN=<api-key>   # or OPENAI_API_KEY / API_KEY\n"
            "  python inference.py"
        )
        return False
    return True

client = OpenAI(api_key=API_KEY or "placeholder", base_url=API_BASE_URL or "http://localhost:8000")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "block_ip", "isolate_machine", "patch", "ignore",
    *[f"scan_node_{i}" for i in range(1, 6)],
]

TEMPERATURE = 0.0
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

TASKS = ["easy", "medium", "hard", "nightmare", "elite", "impossible"]

# Per-task step limits matching task Python configs exactly
# easy=30, medium=50, hard=30, nightmare=15, elite=15, impossible=10
TASK_MAX_STEPS: dict[str, int] = {
    "easy":       30,
    "medium":     50,
    "hard":       30,
    "nightmare":  15,
    "elite":      15,
    "impossible": 10,
}

# IOC-based action inference thresholds.
# Threat types are NOT exposed in observations — agents must reason from behavioral signals.
# These thresholds match the IOC profile separation built into the environment:
#   packets_per_second > 500   → volumetric flood (DDoS-like)       → patch
#   outbound_data_bytes > 2000 → data staging/exfil (malware/ransom) → isolate_machine
#   unusual_process_count > 5  → code execution (malware/ransom)     → isolate_machine
#   lateral_connection_count > 8 → network traversal               → block_ip
#   failed_auth_attempts > 20  → credential attack (phish/lateral)  → block_ip
IOC_THRESHOLDS = {
    "patch":           {"packets_per_second": 500},
    "isolate_machine": {"outbound_data_bytes": 2000, "unusual_process_count": 5},
    "block_ip":        {"lateral_connection_count": 8, "failed_auth_attempts": 20},
}

# ---------------------------------------------------------------------------
# Observation enrichment
# ---------------------------------------------------------------------------

def get_enriched_observation(base_url: str, obs: dict, session_id: str = "") -> dict:
    """Fetch /threat-intel and /analytics to enrich the observation."""
    enriched = dict(obs)
    params = {"session_id": session_id} if session_id else {}

    try:
        intel = requests.get(f"{base_url}/threat-intel", params=params, timeout=5).json()
        enriched["threat_intel"]         = intel.get("active_campaigns", [])
        enriched["risk_level"]           = intel.get("risk_level", "UNKNOWN")
    except Exception:
        enriched["threat_intel"]         = []
        enriched["risk_level"]           = "UNKNOWN"

    try:
        analytics = requests.get(f"{base_url}/analytics", params=params, timeout=5).json()
        enriched["performance_grade"]    = analytics.get("performance_grade", "?")
        enriched["recommended_next"]     = analytics.get("recommended_next_action", "ignore")
        enriched["containment_rate"]     = (
            analytics.get("soc_metrics", {}).get("containment_rate", 0.0)
        )
        enriched["resources_remaining"]  = analytics.get("resources_remaining", 0.5)
    except Exception:
        enriched["performance_grade"]    = "?"
        enriched["recommended_next"]     = "ignore"
        enriched["containment_rate"]     = 0.0
        enriched["resources_remaining"]  = 0.5

    return enriched


# ---------------------------------------------------------------------------
# IOC-based action inference (no LLM needed when signals are unambiguous)
# ---------------------------------------------------------------------------

def _ioc_action(t: dict) -> str | None:
    """Infer the best action for a single threat from its behavioral IOC signals.
    Returns None when the signals are too ambiguous to act without LLM reasoning.
    Signal separation is intentional: each class has at least one uniquely high axis.
    """
    pps     = t.get("packets_per_second", 0)
    outbound = t.get("outbound_data_bytes", 0)
    lateral = t.get("lateral_connection_count", 0)
    auth    = t.get("failed_auth_attempts", 0)
    procs   = t.get("unusual_process_count", 0)

    # Volumetric flood: uniquely extreme packet rate
    if pps > IOC_THRESHOLDS["patch"]["packets_per_second"]:
        return "patch"
    # Data staging / code execution: high outbound OR high process churn
    if (outbound > IOC_THRESHOLDS["isolate_machine"]["outbound_data_bytes"]
            or procs > IOC_THRESHOLDS["isolate_machine"]["unusual_process_count"]):
        return "isolate_machine"
    # Credential harvesting / network traversal: high lateral connections OR auth failures
    if (lateral > IOC_THRESHOLDS["block_ip"]["lateral_connection_count"]
            or auth > IOC_THRESHOLDS["block_ip"]["failed_auth_attempts"]):
        return "block_ip"
    return None


def deterministic_action(
    enriched: dict,
    scanned_nodes: set,
    step_num: int = 0,
) -> str | None:
    """Return an action from IOC signals without calling the LLM.
    Returns None when signals are too ambiguous and the LLM should decide.
    """
    threats   = enriched.get("visible_threats", [])
    all_nodes = [f"node_{i}" for i in range(1, 6)]
    unscanned = [n for n in all_nodes if n not in scanned_nodes]

    # ── 1. Escalated threats — highest urgency, act immediately ───────────────
    for t in threats:
        if t.get("escalated") or t.get("stage") == "lateral_movement":
            action = _ioc_action(t)
            if action:
                return action

    # ── 2. High-severity visible threats ─────────────────────────────────────
    for t in threats:
        if float(t.get("severity", 0)) > 0.7:
            action = _ioc_action(t)
            if action:
                return action

    # ── 3. Any visible threat with clear IOC signal ───────────────────────────
    for t in threats:
        action = _ioc_action(t)
        if action:
            return action

    # ── 4. No visible threats — scan unscanned nodes ─────────────────────────
    if not threats and unscanned:
        return f"scan_node_{unscanned[0].split('_')[1]}"

    if not threats:
        return "ignore"

    # Threats visible but all signals below thresholds — let LLM reason
    return None


# ---------------------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------------------

def choose_action(
    obs: dict,
    step_num: int,
    last_action: str,
    last_reward: float,
    scanned_nodes: set,
    base_url: str,
    session_id: str = "",
) -> str:
    """Enrich obs, try deterministic path first, then ask LLM."""

    # ── Speed bonus: act on young visible threats before enrichment round-trip ──
    # Containing a threat at age < 3 gives +0.1 early bonus.
    # Use IOC-based inference for the fast path — no type lookup needed.
    young_threats = [t for t in obs.get("visible_threats", []) if t.get("age", 99) < 3]
    if young_threats:
        youngest = min(young_threats, key=lambda t: t.get("age", 99))
        immediate = _ioc_action(youngest)
        if immediate:
            print(f"[speed] young threat age={youngest.get('age')} "
                  f"pps={youngest.get('packets_per_second')} "
                  f"ob={youngest.get('outbound_data_bytes')} "
                  f"auth={youngest.get('failed_auth_attempts')} "
                  f"→ {immediate} (speed bonus)")
            return immediate

    enriched      = get_enriched_observation(base_url, obs, session_id)
    threats       = enriched.get("visible_threats", [])
    coverage      = enriched.get("scan_coverage", 0.0)
    system_health = enriched.get("system_health", 100)
    all_nodes     = [f"node_{i}" for i in range(1, 6)]
    unscanned     = [n for n in all_nodes if n not in scanned_nodes]

    # Fast path: deterministic answer
    fast = deterministic_action(enriched, scanned_nodes, step_num)
    if fast is not None and fast in VALID_ACTIONS:
        return fast

    # ── Build compact IOC-based threat summary ────────────────────────────────
    threat_lines = []
    for t in threats:
        urgency = "URGENT" if t.get("age", 0) >= 3 else "monitor"
        threat_lines.append(
            f"  [{urgency}] node={t.get('node','?')}  stage={t.get('stage','?')}"
            f"  age={t.get('age',0)}  severity={t.get('severity','?')}"
            f"  pps={t.get('packets_per_second',0)}"
            f"  auth_fails={t.get('failed_auth_attempts',0)}"
            f"  outbound={t.get('outbound_data_bytes',0)}B"
            f"  lateral_conns={t.get('lateral_connection_count',0)}"
            f"  procs={t.get('unusual_process_count',0)}"
            f"  spread={t.get('spread_rate',0)}"
            f"  persistent={t.get('is_persistent',False)}"
        )
    threat_summary = "\n".join(threat_lines) if threat_lines else "  (none visible)"

    recommended_next = enriched.get("recommended_next", "ignore")
    risk_level       = enriched.get("risk_level", "UNKNOWN")
    grade            = enriched.get("performance_grade", "?")
    containment_rate = enriched.get("containment_rate", 0.0)
    next_scan        = f"scan_node_{unscanned[0].split('_')[1]}" if unscanned else "ignore"

    prompt = f"""You are an expert SOC analyst. Choose ONE defensive action based on behavioral IOC signals.
Threat types are NOT shown. You must INFER the threat class from the signals below.

SITUATION (Step {step_num}):
  Health: {system_health}/100  |  Risk: {risk_level}  |  Grade: {grade}
  Scan coverage: {coverage:.0%}  |  Containment rate: {containment_rate:.0%}
  Scanned nodes: {sorted(scanned_nodes) or 'none'}
  Unscanned nodes: {unscanned}
  Last action: {last_action}  →  reward: {last_reward:.3f}

ACTIVE ALERTS (behavioral IOCs — NO type labels):
{threat_summary}

IOC → THREAT CLASS → CORRECT ACTION (learn this pattern):
  packets_per_second > 500             → volumetric flood       → patch
  outbound_data_bytes > 2000           → data staging/exfil     → isolate_machine
  unusual_process_count > 5            → code execution         → isolate_machine
  lateral_connection_count > 8         → network traversal      → block_ip
  failed_auth_attempts > 20 + low pps  → credential harvesting  → block_ip

PENALTY GUIDE:
  Correct mitigation  →  +1.0 reward (early containment age<3: +1.1)
  Wrong mitigation    →  -0.5 penalty
  ignore              →  -1.5 penalty  AND  -10 health

REASONING EXAMPLE:
  outbound=35000B, procs=18, spread=0.8, pps=12
  → extreme outbound + high process count + high spread = data encryption/exfil
  → action: isolate_machine

PRIORITY ORDER:
  1. URGENT escalated threat (stage=lateral_movement or age>=3) → infer from IOCs above
  2. Any visible threat with clear signal                        → infer from IOCs above
  3. Unscanned nodes remain                                      → {next_scan}
  4. No threats visible                                          → ignore

SOC ANALYTICS HINT: {recommended_next}

VALID ACTIONS (output EXACTLY one):
  block_ip  isolate_machine  patch  ignore
  scan_node_1  scan_node_2  scan_node_3  scan_node_4  scan_node_5

OUTPUT ONLY THE ACTION. One word. No explanation."""

    model = MODEL_NAME  # single model; MODELS list kept for future expansion
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=TEMPERATURE,
            )
            raw    = response.choices[0].message.content.strip().lower()
            action = raw.split()[0].strip(".,!?:") if raw else "ignore"
            if action not in VALID_ACTIONS:
                print(f"[model] invalid '{action}' → fallback to intel recommendation")
                action = recommended_next if recommended_next in VALID_ACTIONS else "ignore"
            return action
        except Exception as e:
            print(f"[model] attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    # Final fallback: deterministic (no LLM)
    print("[model] exhausted retries — using deterministic fallback")
    return deterministic_action(enriched, scanned_nodes, step_num) or "ignore"


# ---------------------------------------------------------------------------
# Single-task interaction loop
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> dict:
    """Run one full episode for the given task. Returns a result summary dict."""
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'=' * 60}")

    try:
        reset_data = requests.post(
            f"{BASE_URL}/reset", json={"task": task_name}
        ).json()
    except Exception as e:
        print(f"[error] /reset failed for task '{task_name}': {e}")
        return {
            "task_id": task_name, "steps": 0,
            "total_reward": 0.0, "score": 0.0, "status": "reset_failed",
        }

    session_id = reset_data.get("session_id", "")
    if not session_id:
        print(f"[warn] /reset did not return session_id for task '{task_name}'")

    total_reward  = 0.0
    final_status  = "in_progress"
    last_action   = "none"
    last_reward   = 0.0
    scanned_nodes: set = set()
    step_num      = 0

    print(f"{'Step':<6} {'Action':<22} {'Reward':<8} {'Reason'}")
    print("-" * 70)

    max_steps = TASK_MAX_STEPS.get(task_name, 30)
    for step_num in range(1, max_steps + 1):
        try:
            obs = requests.get(f"{BASE_URL}/state",
                               params={"session_id": session_id}).json()
        except Exception as e:
            print(f"[error] /state failed at step {step_num}: {e}")
            final_status = "error"
            break

        if obs.get("done"):
            final_status = "done"
            break

        action = choose_action(
            obs, step_num, last_action, last_reward,
            scanned_nodes, BASE_URL, session_id,
        )

        if action.startswith("scan_node_"):
            node_id = action.replace("scan_node_", "node_")
            scanned_nodes.add(node_id)

        try:
            data = requests.post(f"{BASE_URL}/step",
                                 json={"action": action, "session_id": session_id}).json()
        except Exception as e:
            print(f"[error] /step failed at step {step_num}: {e}")
            final_status = "error"
            break

        last_reward   = data.get("reward", 0.0)
        last_action   = action
        total_reward += last_reward
        reason        = data.get("reason", "")
        step_label    = data.get("step", step_num)

        print(f"{step_label:<6} {action:<22} {last_reward:<8.3f} {reason[:60]}")

        if data.get("done"):
            final_status = "done"
            break
    else:
        final_status = "max_steps_reached"

    # ── Grader formula — EXACT match to tasks/base.py _compute_episode_score ────
    # score = 0.50 × containment_rate   (threats_contained / threats_total_spawned)
    #       + 0.20 × critical_health    (system_health / 100 — best HTTP-API proxy)
    #       + 0.15 × avg_resource_left  (1 - total_spent / total_budget, from /analytics)
    #       + 0.15 × speed_bonus        (mean early-containment score per threat)
    try:
        analytics = requests.get(f"{BASE_URL}/analytics",
                                 params={"session_id": session_id}, timeout=5).json()
        soc = analytics.get("soc_metrics", {})
        net = analytics.get("network_status", {})
        # containment_rate: threats_contained / threats_total_spawned — matches base.py exactly.
        containment_rate  = float(soc.get("containment_rate", 0.0))
        # critical_health: system_health/100 — HTTP API proxy for base.py avg critical-asset health.
        critical_health   = float(net.get("system_health", 0)) / 100.0
        # avg_resource_left: fraction of cumulative budget unused
        avg_resource_left = float(analytics.get("resources_remaining", 0.0))
    except Exception:
        containment_rate  = 0.0
        critical_health   = 0.0
        avg_resource_left = 0.5  # neutral fallback on analytics failure

    # Speed bonus: fetch containment_events from /state episode_info
    try:
        final_state = requests.get(f"{BASE_URL}/state", params={"session_id": session_id}, timeout=5).json()
        containment_events = final_state.get("episode_info", {}).get("containment_events", [])
        speed_bonus = 0.0
        if containment_events:
            sb_scores = []
            for ev in containment_events:
                age = ev.get("age_at_containment", 99)
                if age < 3:
                    sb_scores.append(1.0)
                elif age < 5:
                    sb_scores.append(0.5)
                else:
                    sb_scores.append(0.0)
            speed_bonus = sum(sb_scores) / len(sb_scores) if sb_scores else 0.0
    except Exception:
        speed_bonus = 0.0

    score = max(0.0, min(1.0,
        0.50 * containment_rate
        + 0.20 * critical_health
        + 0.15 * avg_resource_left
        + 0.15 * speed_bonus
    ))

    return {
        "task_id":          task_name,
        "steps":            step_num,
        "total_reward":     round(total_reward, 3),
        "containment_rate": round(containment_rate, 3),
        "critical_health":  round(critical_health, 3),
        "avg_resource_left": round(avg_resource_left, 3),
        "speed_bonus":      round(speed_bonus, 3),
        "score":            round(score, 3),
        "status":           final_status,
    }


# ---------------------------------------------------------------------------
# Main: run all tasks sequentially and print summary
# ---------------------------------------------------------------------------

# Per-task passing thresholds matching task Python configs
TASK_THRESHOLDS: dict[str, float] = {
    "easy":       0.50,
    "medium":     0.60,
    "hard":       0.45,
    "nightmare":  0.25,
    "elite":      0.20,
    "impossible": 0.10,
}


def run():
    if not _check_env_vars():
        raise SystemExit(1)
    results = []

    for task_name in TASKS:
        result = run_task(task_name)
        results.append((task_name, result))

    sep  = "=" * 80
    dash = "-" * 80
    print(f"\n{sep}")
    print("BASELINE RESULTS  (grader: 0.50×contain + 0.20×health + 0.15×resource + 0.15×speed_bonus)")
    print(sep)
    print(f"{'Task':<12} {'Steps':<7} {'Contain%':<10} {'Health%':<10} {'SpeedBonus':<12} {'Score':<8} {'Threshold':<11} {'Result'}")
    print(dash)
    scores = []
    for task_name, r in results:
        threshold = TASK_THRESHOLDS.get(task_name, 0.50)
        result_label = "PASS ✓" if r["score"] >= threshold else "FAIL ✗"
        scores.append(r["score"])
        print(
            f"{task_name:<12} {r['steps']:<7} "
            f"{r['containment_rate']:<10.3f} {r['critical_health']:<10.3f} "
            f"{r['speed_bonus']:<12.3f} {r['score']:<8.3f} {threshold:<11.2f} {result_label}"
        )
    print(dash)

    non_increasing = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    passes = sum(
        1 for (task_name, r) in results
        if r["score"] >= TASK_THRESHOLDS.get(task_name, 0.50)
    )
    print(f"\nScores non-increasing with difficulty: {'YES ✓' if non_increasing else 'NO ✗'}")
    print(f"Tasks passed: {passes}/{len(results)}")
    print(sep)


if __name__ == "__main__":
    if not _check_env_vars():
        raise SystemExit(1)
    run()
