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

if not API_BASE_URL:
    raise ValueError("Missing required environment variable: API_BASE_URL")
if not MODEL_NAME:
    raise ValueError("Missing required environment variable: MODEL_NAME")
if not API_KEY:
    raise ValueError("Missing required environment variable: HF_TOKEN or API_KEY")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

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

# MITRE → correct action lookup (used for deterministic fallback)
MITRE_ACTION = {
    "phishing":         "block_ip",
    "malware":          "isolate_machine",
    "ransomware":       "isolate_machine",
    "ddos":             "patch",
    "lateral_movement": "block_ip",
    "T1566":            "block_ip",
    "T1204":            "isolate_machine",
    "T1486":            "isolate_machine",
    "T1499":            "patch",
    "T1021":            "block_ip",
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
        enriched["recommended_actions"]  = intel.get("recommended_actions", [])
        enriched["risk_level"]           = intel.get("risk_level", "UNKNOWN")
    except Exception:
        enriched["threat_intel"]         = []
        enriched["recommended_actions"]  = []
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
# Deterministic fallback (no LLM needed when answer is obvious)
# ---------------------------------------------------------------------------

def deterministic_action(
    enriched: dict,
    scanned_nodes: set,
    step_num: int = 0,
) -> str | None:
    """
    Return an action without calling the LLM when the answer is unambiguous.
    Returns None when the situation is genuinely ambiguous and needs the LLM.
    """
    threats       = enriched.get("visible_threats", [])
    all_nodes     = [f"node_{i}" for i in range(1, 6)]
    unscanned     = [n for n in all_nodes if n not in scanned_nodes]

    # ── 1. lateral_movement → block_ip immediately (spreads fast) ─────────────
    for t in threats:
        if t.get("type") == "lateral_movement" or t.get("stage") == "lateral_movement":
            return "block_ip"

    # ── 2. CRITICAL severity threats — use MITRE lookup ───────────────────────
    for t in threats:
        if float(t.get("severity", 0)) > 0.7:
            action = (
                MITRE_ACTION.get(t.get("type", ""))
                or MITRE_ACTION.get(t.get("technique_id", ""))
            )
            if action:
                return action

    # ── 3. Any visible threat with known type — MITRE lookup ──────────────────
    for t in threats:
        action = (
            MITRE_ACTION.get(t.get("type", ""))
            or MITRE_ACTION.get(t.get("technique_id", ""))
        )
        if action:
            return action

    # ── 4. No visible threats — scan unscanned nodes first ────────────────────
    if not threats and unscanned:
        return f"scan_node_{unscanned[0].split('_')[1]}"

    # ── 5. All nodes scanned, no threats — ignore (empty rescan is penalized) ──
    # Server penalizes empty rescans at 0.425; ignore is neutral when no threats exist.
    if not threats:
        return "ignore"

    # Threats visible but type/technique unknown — let LLM decide
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

    # ── CHANGE 3: speed bonus hunting — act immediately on young threats ───────
    # Containing a threat at age < 3 gives +0.1 early bonus.
    # Skip enrichment entirely for maximum speed on obvious decisions.
    young_threats = [
        t for t in obs.get("visible_threats", [])
        if t.get("age", 99) < 3 and t.get("type")
    ]
    if young_threats:
        youngest = min(young_threats, key=lambda t: t.get("age", 99))
        threat_type = youngest.get("type", "").lower()
        immediate = MITRE_ACTION.get(threat_type)
        if immediate:
            print(f"[speed] young {threat_type} age={youngest.get('age')} → {immediate} (speed bonus)")
            return immediate

    enriched      = get_enriched_observation(base_url, obs, session_id)
    threats       = enriched.get("visible_threats", [])
    coverage      = enriched.get("scan_coverage", 0.0)
    system_health = enriched.get("system_health", 100)
    all_nodes     = [f"node_{i}" for i in range(1, 6)]
    unscanned     = [n for n in all_nodes if n not in scanned_nodes]

    # Fast path: deterministic answer
    fast = deterministic_action(enriched, scanned_nodes, step_num)
    if fast is not None:
        # Normalise scan_node_X format (deterministic returns "scan_nodeX" without underscore)
        if fast.startswith("scan_node_") and fast in VALID_ACTIONS:
            return fast
        if fast.startswith("scan_") and not fast.startswith("scan_node_"):
            # convert "scan_node_1" shorthand if needed
            fast = fast.replace("scan_", "scan_node_", 1) if "node" not in fast else fast
        if fast in VALID_ACTIONS:
            return fast

    # ── Build compact, structured threat summary ───────────────────────────────
    threat_lines = []
    for t in threats:
        urgency = "URGENT" if t.get("age", 0) >= 3 else "monitor"
        threat_lines.append(
            f"  [{urgency}] type={t.get('type','?')}  node={t.get('node','?')}"
            f"  stage={t.get('stage','?')}  age={t.get('age',0)}"
            f"  technique={t.get('technique_id','?')}"
            f"  severity={t.get('severity','?')}"
        )
    threat_summary = "\n".join(threat_lines) if threat_lines else "  (none visible)"

    # Intel from /threat-intel
    recommended_actions = enriched.get("recommended_actions", [])
    recommended_next    = enriched.get("recommended_next", "ignore")
    risk_level          = enriched.get("risk_level", "UNKNOWN")
    grade               = enriched.get("performance_grade", "?")
    containment_rate    = enriched.get("containment_rate", 0.0)

    # Next unscanned node action
    next_scan = f"scan_node_{unscanned[0].split('_')[1]}" if unscanned else "ignore"

    prompt = f"""You are an expert SOC analyst. Choose ONE action RIGHT NOW.

SITUATION (Step {step_num}):
  Health: {system_health}/100  |  Risk: {risk_level}  |  Grade: {grade}
  Scan coverage: {coverage:.0%}  |  Containment rate: {containment_rate:.0%}
  Scanned nodes: {sorted(scanned_nodes) or 'none'}
  Unscanned nodes: {unscanned}
  Last action: {last_action}  →  reward: {last_reward:.3f}

ACTIVE THREATS:
{threat_summary}

MITRE ATT&CK RESPONSE — follow EXACTLY, these give +1.0 reward:
  phishing        / T1566  →  block_ip
  malware         / T1204  →  isolate_machine
  ransomware      / T1486  →  isolate_machine
  ddos            / T1499  →  patch
  lateral_movement/ T1021  →  block_ip
  Wrong action             →  -0.5 penalty
  ignore                   →  -1.5 penalty

PRIORITY ORDER (strict):
  1. lateral_movement threat visible  →  block_ip  (spreads to other nodes!)
  2. CRITICAL severity threat visible →  use MITRE action above
  3. HIGH severity threat visible     →  use MITRE action above
  4. Any threat visible with known type → use MITRE action above
  5. Unscanned nodes remain           →  {next_scan}
  6. All nodes clean                  →  ignore

THREAT INTEL SAYS:
  Recommended: {recommended_next}
  Intel actions: {recommended_actions or 'none'}

VALID ACTIONS (output EXACTLY one of these):
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

    # ── Grader formula (mirrors tasks/base.py _compute_episode_score) ──────────
    # score = 0.50×containment_rate + 0.20×critical_health
    #       + 0.15×resources_remaining + 0.15×avg_step_reward
    try:
        analytics = requests.get(f"{BASE_URL}/analytics",
                                 params={"session_id": session_id}, timeout=5).json()
        soc = analytics.get("soc_metrics", {})
        net = analytics.get("network_status", {})
        containment_rate    = float(soc.get("containment_rate", 0.0))
        critical_health     = float(net.get("system_health", 0)) / 100.0
        avg_step_reward     = float(soc.get("avg_reward_per_step",
                                            total_reward / max(1, step_num)))
        resources_remaining = float(analytics.get("resources_remaining", 0.5))
    except Exception:
        containment_rate    = 0.0
        critical_health     = 0.0
        avg_step_reward     = total_reward / max(1, step_num)
        resources_remaining = 0.0  # conservative: don't inflate score on analytics failure

    score = max(0.0, min(1.0,
        0.50 * containment_rate
        + 0.20 * critical_health
        + 0.15 * resources_remaining
        + 0.15 * avg_step_reward
    ))

    return {
        "task_id":           task_name,
        "steps":             step_num,
        "total_reward":      round(total_reward, 3),
        "containment_rate":  round(containment_rate, 3),
        "critical_health":   round(critical_health, 3),
        "avg_step_reward":   round(avg_step_reward, 3),
        "score":             round(score, 3),
        "status":            final_status,
    }


# ---------------------------------------------------------------------------
# Main: run all tasks sequentially and print summary
# ---------------------------------------------------------------------------

# Per-task passing thresholds matching task Python configs
TASK_THRESHOLDS: dict[str, float] = {
    "easy":       0.55,
    "medium":     0.55,
    "hard":       0.45,
    "nightmare":  0.25,
    "elite":      0.20,
    "impossible": 0.10,
}


def run():
    results = []

    for task_name in TASKS:
        result = run_task(task_name)
        results.append((task_name, result))

    sep  = "=" * 80
    dash = "-" * 80
    print(f"\n{sep}")
    print("BASELINE RESULTS  (grader: 0.50×contain + 0.20×health + 0.15×res + 0.15×avg_r)")
    print(sep)
    print(f"{'Task':<12} {'Steps':<7} {'Contain%':<10} {'Health%':<10} {'AvgRew':<9} {'Score':<8} {'Threshold':<11} {'Result'}")
    print(dash)
    scores = []
    for task_name, r in results:
        threshold = TASK_THRESHOLDS.get(task_name, 0.50)
        result_label = "PASS ✓" if r["score"] >= threshold else "FAIL ✗"
        scores.append(r["score"])
        print(
            f"{task_name:<12} {r['steps']:<7} "
            f"{r['containment_rate']:<10.3f} {r['critical_health']:<10.3f} "
            f"{r['avg_step_reward']:<9.3f} {r['score']:<8.3f} {threshold:<11.2f} {result_label}"
        )
    print(dash)

    decreasing = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    passes = sum(
        1 for (task_name, r) in results
        if r["score"] >= TASK_THRESHOLDS.get(task_name, 0.50)
    )
    print(f"\nScores decrease with difficulty: {'YES ✓' if decreasing else 'NO ✗'}")
    print(f"Tasks passed: {passes}/{len(results)}")
    print(sep)


if __name__ == "__main__":
    run()
