import json
import os
import sys
import time
import requests
from openai import OpenAI
from grader import TASK_PASSING_SCORES, compute_grader_score as _compute_grader_formula

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

BASE_URL     = os.getenv("BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("OPENAI_API_KEY") or HF_TOKEN or os.getenv("API_KEY")

# Print LLM reasoning every step so judges can watch the agent think.
# Set VERBOSE=false to suppress.
VERBOSE = os.getenv("VERBOSE", "true").lower() != "false"


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


TIMEOUT = 30  # seconds — judge evaluation timeout

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=HF_TOKEN or API_KEY,
            base_url=API_BASE_URL,
        )
    return _client

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "block_ip", "isolate_machine", "patch", "ignore",
    *[f"scan_node_{i}" for i in range(1, 6)],
]

# max_tokens must fit: {"action":"PATCH_VULNERABILITY","target":"node_5","reasoning":"...~80 chars..."}
LLM_MAX_TOKENS = 120
TEMPERATURE    = 0.2   # slight creativity helps with step-by-step reasoning
MAX_RETRIES    = 3
RETRY_DELAY    = 2     # seconds between retries

# Single source of truth — imported from grader.py, matches openenv.yaml.
TASK_THRESHOLDS: dict[str, float] = TASK_PASSING_SCORES

TASKS = ["easy", "medium", "hard", "nightmare", "elite"]

TASK_MAX_STEPS: dict[str, int] = {
    "easy":       30,
    "medium":     50,
    "hard":       30,
    "nightmare":  15,
    "elite":      15,
    "impossible": 10,
}

# Map LLM action names → environment action strings.
# The LLM uses human-readable names that make the reasoning more legible;
# the environment expects lowercase snake_case strings.
ACTION_MAP: dict[str, str] = {
    "ISOLATE_NODE":        "isolate_machine",
    "ISOLATE_MACHINE":     "isolate_machine",
    "BLOCK_IP":            "block_ip",
    "PATCH":               "patch",
    "PATCH_VULNERABILITY": "patch",
    "DO_NOTHING":          "ignore",
    "IGNORE":              "ignore",
    # SCAN is handled separately — requires the target node field
}

# ---------------------------------------------------------------------------
# System prompt (constant across all steps)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an autonomous SOC analyst defending a 5-node corporate network \
against live cyber attacks. You do NOT receive threat type labels. Instead you receive \
behavioral Indicators of Compromise (IOCs) for each active alert and must infer what \
kind of attack is occurring, then choose the single best defensive action.

Available actions:
  ISOLATE_MACHINE — network isolation of a compromised node
  BLOCK_IP        — block a source IP address
  PATCH           — apply a security patch to a node
  SCAN_NODE_1 through SCAN_NODE_5 — reveal hidden threats
  DO_NOTHING      — take no action (heavy penalty)

Your job is to reason from the IOC signals which action is most appropriate. \
The correct action depends on what the signals suggest about the attack class.

Reasoning approach:
Each threat exposes behavioral IOC signals. Your job is to classify the attack from
these signals and choose the correct mitigation. Consider:
- Which IOCs are elevated versus baseline?
- What combination of signals suggests the attack class?
- High network volume suggests a different threat class than high authentication
  failures or high data egress.
- Lateral spread indicators suggest different mitigations than encryption or
  exfiltration indicators.
- Severity and detection_confidence tell you urgency, not attack type.
- A threat with ambiguous IOCs may be a false positive — consider the cost of
  acting versus waiting for more signal via SCAN.

Think step by step before answering:
  1. What does each threat's IOC profile suggest about its attack class?
  2. Which threat is most dangerous right now (highest severity, most escalated)?
  3. What action best neutralises it given the current resource budget?

Reply ONLY with a single JSON object — no markdown, no prose:
{"action": "ACTION_NAME", "target": "node_id", "reasoning": "one sentence"}

Examples of valid responses:
{"action": "ISOLATE_MACHINE", "target": "node_3", "reasoning": "combined high outbound bytes and process anomalies with rapid spread — encryption behavior pattern, isolate to stop propagation"}
{"action": "BLOCK_IP", "target": "node_1", "reasoning": "authentication failures dominate with minimal network volume — credential attack pattern, block source IP"}
{"action": "SCAN", "target": "node_4", "reasoning": "hidden_threat_count=2 but node_4 not yet scanned — need visibility before committing resources"}
{"action": "PATCH", "target": "node_2", "reasoning": "packets_per_second is the dominant signal with minimal auth failures or process anomalies — applying patch to neutralise the threat"}
"""

# ---------------------------------------------------------------------------
# Observation enrichment
# ---------------------------------------------------------------------------

def get_enriched_observation(base_url: str, obs: dict, session_id: str = "") -> dict:
    """Enrich the base observation with analytics and threat-intel data.

    Fetches /analytics and /threat-intel in parallel (two calls) and merges
    the results into the observation dict. Falls back gracefully if either
    endpoint is unavailable.
    """
    enriched = dict(obs)
    enriched.setdefault("risk_level",          "UNKNOWN")
    enriched.setdefault("threat_summary",      {})
    enriched.setdefault("performance_grade",   "?")
    enriched.setdefault("containment_rate",    0.0)
    enriched.setdefault("resources_remaining", 0.5)
    enriched.setdefault("attacker_strategy",   "UNKNOWN")

    params = {"session_id": session_id} if session_id else {}
    try:
        analytics = requests.get(f"{base_url}/analytics", params=params, timeout=TIMEOUT).json()
        enriched["performance_grade"]   = analytics.get("performance_grade", "?")
        enriched["containment_rate"]    = analytics.get("soc_metrics", {}).get("containment_rate", 0.0)
        enriched["resources_remaining"] = analytics.get("resources_remaining", 0.5)
        enriched["attacker_strategy"]   = analytics.get("agent_behavior", {}).get("attacker_strategy", "UNKNOWN")
    except Exception:
        pass

    try:
        intel = requests.get(f"{base_url}/threat-intel", params=params, timeout=TIMEOUT).json()
        enriched["risk_level"]     = intel.get("risk_level", "UNKNOWN")
        enriched["threat_summary"] = intel.get("threat_summary", {})
    except Exception:
        pass

    return enriched


# ---------------------------------------------------------------------------
# Action parsing and mapping
# ---------------------------------------------------------------------------

def _parse_llm_response(raw: str) -> tuple[str, str, str]:
    """Extract (action, target, reasoning) from the LLM's JSON response.
    Handles markdown code fences and trailing punctuation.
    Returns ("", "", "") on any parse failure.
    """
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()

    # Find the first JSON object in the response
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        return "", "", ""

    try:
        obj = json.loads(text[start:end])
    except json.JSONDecodeError:
        return "", "", ""

    action    = str(obj.get("action", "")).strip().upper().replace(" ", "_")
    target    = str(obj.get("target", "")).strip().lower()
    reasoning = str(obj.get("reasoning", "")).strip()
    return action, target, reasoning


def _map_to_env_action(action: str, target: str, scanned_nodes: set) -> str | None:
    """Convert the LLM's (action, target) pair to a valid environment action string.
    Returns None if the mapping is impossible.
    """
    if action.upper().startswith("SCAN_NODE_"):
        candidate = action.lower()
        if candidate in VALID_ACTIONS:
            return candidate

    if action == "SCAN":
        # Prefer the LLM's target node if it hasn't been scanned yet
        if target.startswith("node_") and target not in scanned_nodes:
            candidate = f"scan_{target}"
            if candidate in VALID_ACTIONS:
                return candidate
        # Target already scanned or invalid — redirect to first unscanned node
        for i in range(1, 6):
            node_id   = f"node_{i}"
            candidate = f"scan_node_{i}"
            if node_id not in scanned_nodes and candidate in VALID_ACTIONS:
                return candidate
        # All nodes scanned — allow re-scan of LLM's target (yields low reward but valid)
        if target.startswith("node_"):
            candidate = f"scan_{target}"
            if candidate in VALID_ACTIONS:
                return candidate
        return None

    env_action = ACTION_MAP.get(action)
    if env_action and env_action in VALID_ACTIONS:
        return env_action

    return None


def _fallback_action(threats: list, scanned_nodes: set) -> str:
    """Error fallback: scan the node hosting the highest-severity visible threat.
    If no threats are visible, scan the next unscanned node.
    Never uses a MITRE type lookup.
    """
    if threats:
        # Find the node with the most severe active alert
        worst = max(threats, key=lambda t: float(t.get("severity", 0.0)))
        node = worst.get("node", "")
        if node.startswith("node_"):
            candidate = f"scan_{node}"
            if candidate in VALID_ACTIONS:
                return candidate

    # Scan next unscanned node
    for i in range(1, 6):
        node_id = f"node_{i}"
        if node_id not in scanned_nodes:
            return f"scan_node_{i}"

    return "ignore"


# ---------------------------------------------------------------------------
# LLM-driven action selection — called every step, no rule-based bypass
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
    """Always call the LLM to choose an action.
    The only non-LLM path is the error fallback (API timeout / parse failure).
    """
    enriched       = get_enriched_observation(base_url, obs, session_id)
    threats        = enriched.get("visible_threats", [])
    coverage       = enriched.get("scan_coverage", 0.0)
    system_health  = enriched.get("system_health", 100)
    hidden_count   = enriched.get("hidden_threat_count", 0)
    risk_level     = enriched.get("risk_level", "UNKNOWN")
    grade          = enriched.get("performance_grade", "?")
    containment    = enriched.get("containment_rate", 0.0)
    resources      = enriched.get("resources_remaining", 1.0)
    attacker_strat = enriched.get("attacker_strategy", "UNKNOWN")

    all_nodes  = [f"node_{i}" for i in range(1, 6)]
    unscanned  = [n for n in all_nodes if n not in scanned_nodes]

    # ── Build per-threat IOC block ────────────────────────────────────────────
    if threats:
        threat_lines = []
        for i, t in enumerate(threats, 1):
            urgency = "!! URGENT" if (t.get("age", 0) >= 3 or t.get("escalated")) else "   active"
            threat_lines.append(
                f"  Alert {i} [{urgency}]\n"
                f"    node={t.get('node','?')}  stage={t.get('stage','initial')}"
                f"  age/dwell={t.get('age',0)} steps"
                f"  severity={t.get('severity',0):.2f}"
                f"  confidence={t.get('detection_confidence',0):.2f}\n"
                f"    packets_per_second={t.get('packets_per_second',0)}"
                f"  failed_auth_attempts={t.get('failed_auth_attempts',0)}"
                f"  outbound_data_bytes={t.get('outbound_data_bytes',0)}\n"
                f"    lateral_connection_count={t.get('lateral_connection_count',0)}"
                f"  unusual_process_count={t.get('unusual_process_count',0)}"
                f"  spread_rate={t.get('spread_rate',0):.2f}"
                f"  is_persistent={t.get('is_persistent',False)}"
                f"  affected_nodes={t.get('affected_node_count',1)}"
            )
        threats_block = "\n".join(threat_lines)
    else:
        threats_block = "  (no visible threats — hidden threats may exist on unscanned nodes)"

    user_prompt = f"""STEP {step_num} — LIVE NETWORK STATUS
  Health: {system_health}/100  |  Risk: {risk_level}  |  Grade: {grade}
  Scan coverage: {coverage:.0%}  |  Hidden threats: {hidden_count}
  Containment rate: {containment:.0%}  |  Resources remaining: {resources:.0%}
  Attacker strategy: {attacker_strat}
  Scanned nodes: {sorted(scanned_nodes) if scanned_nodes else 'none'}
  Unscanned nodes: {unscanned if unscanned else 'all scanned'}
  Last action: {last_action}  →  reward: {last_reward:+.3f}

ACTIVE ALERTS — behavioral IOCs (no type labels):
{threats_block}

TASK: Analyse the IOC profiles above, identify the most dangerous threat, and respond \
with the single best defensive action as JSON.
Remember: DO_NOTHING costs -10 health and -1.5 reward. Wrong mitigation costs -0.5. \
Correct mitigation earns +1.0 (age<3 earns +1.1 speed bonus).
"""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            raw = response.choices[0].message.content or ""
            raw = raw.lower().strip()

            action_name, target, reasoning = _parse_llm_response(raw)

            if not action_name:
                if VERBOSE:
                    print(f"  [llm] attempt {attempt}: parse failed — raw={raw!r:.80}")
                continue

            env_action = _map_to_env_action(action_name, target, scanned_nodes)

            if env_action is None:
                if VERBOSE:
                    print(f"  [llm] attempt {attempt}: unmappable action={action_name!r} target={target!r}")
                continue

            if VERBOSE:
                print(f"  [llm] {action_name}({target}) → {env_action}")
                print(f"  [reasoning] {reasoning}")

            return env_action, target

        except Exception as e:
            if VERBOSE:
                print(f"  [llm] attempt {attempt}/{MAX_RETRIES} error: {e}")
            err_str = str(e)
            if "403" in err_str or "401" in err_str:
                print(f"ERROR: LLM authentication failed (HTTP {401 if '401' in err_str else 403}). "
                      f"Check HF_TOKEN / OPENAI_API_KEY / API_KEY and API_BASE_URL.")
                return "ignore", ""   # signals caller to abort cleanly via done
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    # ── Error fallback: scan highest-severity node — NO MITRE lookup ──────────
    fallback = _fallback_action(threats, scanned_nodes)
    if VERBOSE:
        print(f"  [llm] exhausted retries — fallback scan: {fallback}")
    return fallback, ""


# ---------------------------------------------------------------------------
# Single-task interaction loop
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> dict:
    """Run one full episode for the given task. Returns a result summary dict."""
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"TASK: {task_name.upper()}")
    print(sep)

    all_rewards: list[float] = []
    print(f"[START] task={task_name} env=adaptive-cyber-defense model={MODEL_NAME}")
    sys.stdout.flush()

    try:
        reset_data = requests.post(
            f"{BASE_URL}/reset", json={"task": task_name}, timeout=TIMEOUT
        ).json()
    except requests.exceptions.Timeout:
        print(f"[TIMEOUT] /reset timed out after {TIMEOUT}s for task '{task_name}'")
        print(f"[END]   success=false steps=0 rewards=")
        sys.stdout.flush()
        return {
            "task_id": task_name, "steps": 0,
            "total_reward": 0.0, "score": 0.0, "status": "timeout",
        }
    except Exception as e:
        print(f"[error] /reset failed for task '{task_name}': {e}")
        print(f"[END]   success=false steps=0 rewards=")
        sys.stdout.flush()
        return {
            "task_id": task_name, "steps": 0,
            "total_reward": 0.0, "score": 0.0, "status": "reset_failed",
        }

    session_id = reset_data.get("session_id", "")
    if not session_id:
        print(f"[warn] /reset did not return session_id for task '{task_name}'")

    total_reward    = 0.0
    final_status    = "in_progress"
    last_action     = "none"
    last_reward     = 0.0
    scanned_nodes: set = set()
    step_num        = 0

    print(f"{'Step':<6} {'Env Action':<22} {'Reward':<8} {'Env Reason'}")
    print("-" * 70)

    max_steps = TASK_MAX_STEPS.get(task_name, 30)
    for step_num in range(1, max_steps + 1):
        try:
            obs = requests.get(
                f"{BASE_URL}/state", params={"session_id": session_id},
                timeout=TIMEOUT,
            ).json()
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] /state timed out after {TIMEOUT}s at step {step_num}")
            final_status = "timeout"
            break
        except Exception as e:
            print(f"[error] /state failed at step {step_num}: {e}")
            final_status = "error"
            break

        if obs.get("done"):
            final_status = "done"
            break

        if VERBOSE:
            print(f"\nStep {step_num}:")

        action, target_node = choose_action(
            obs, step_num, last_action, last_reward,
            scanned_nodes, BASE_URL, session_id,
        )

        if action.startswith("scan_node_"):
            scanned_nodes.add(action.replace("scan_node_", "node_"))

        step_payload = {"action": action, "session_id": session_id}
        if target_node:
            step_payload["target_node"] = target_node

        try:
            data = requests.post(
                f"{BASE_URL}/step",
                json=step_payload,
                timeout=TIMEOUT,
            ).json()
        except requests.exceptions.Timeout:
            print(f"[TIMEOUT] /step timed out after {TIMEOUT}s at step {step_num}")
            final_status = "timeout"
            break
        except Exception as e:
            print(f"[error] /step failed at step {step_num}: {e}")
            final_status = "error"
            break

        last_reward   = data.get("reward", 0.0)
        last_action   = action
        total_reward += last_reward
        all_rewards.append(last_reward)
        reason        = data.get("reason", "")
        step_label    = data.get("step", step_num)
        done_flag     = data.get("done", False)

        print(f"{step_label:<6} {action:<22} {last_reward:<8.3f} {reason[:55]}")
        print(f"[STEP]  step={step_num} action={action} reward={last_reward:.2f} done={str(done_flag).lower()} error=null")
        sys.stdout.flush()

        if done_flag:
            final_status = "done"
            break
    else:
        final_status = "max_steps_reached"

    # ── Grader formula ────────────────────────────────────────────────────────
    containment_rate  = 0.0
    critical_health   = 0.0
    avg_resource_left = 0.0
    speed_bonus       = 0.0
    score             = 0.0

    try:
        final_state  = requests.get(
            f"{BASE_URL}/state",
            params={"session_id": session_id}, timeout=TIMEOUT,
        ).json()
        episode_info = final_state.get("episode_info", {})

        # Read all four components from the server's authoritative episode_info.
        # The server computes these correctly at episode end using the same
        # grader.py formula; do NOT post-process or correct these values.
        containment_rate  = float(episode_info.get("containment_rate",   0.0))
        critical_health   = float(episode_info.get("critical_health",    0.0))
        avg_resource_left = float(episode_info.get("resources_remaining", 0.0))
        speed_bonus       = float(episode_info.get("speed_bonus",         0.0))

        # Use the authoritative grader formula — single source of truth.
        score = _compute_grader_formula(
            containment_rate, critical_health, avg_resource_left, speed_bonus
        )
    except Exception:
        pass

    threshold   = TASK_THRESHOLDS.get(task_name, 0.50)
    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    success_str = "true" if score >= threshold else "false"
    print(f"[END]   success={success_str} steps={step_num} rewards={rewards_str}")
    sys.stdout.flush()

    return {
        "task_id":           task_name,
        "steps":             step_num,
        "total_reward":      round(total_reward, 3),
        "containment_rate":  round(containment_rate, 3),
        "critical_health":   round(critical_health, 3),
        "avg_resource_left": round(avg_resource_left, 3),
        "speed_bonus":       round(speed_bonus, 3),
        "score":             round(score, 3),
        "status":            final_status,
    }


# ---------------------------------------------------------------------------
# Main: run all tasks and print summary
# ---------------------------------------------------------------------------


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
    print("RESULTS  (grader: 0.50×contain + 0.20×health + 0.15×resource + 0.15×speed)")
    print(sep)
    print(f"{'Task':<12} {'Steps':<7} {'Contain%':<10} {'Health%':<10} "
          f"{'Speed':<8} {'Score':<8} {'Thresh':<8} {'Result'}")
    print(dash)
    scores = []
    for task_name, r in results:
        threshold = TASK_THRESHOLDS.get(task_name, 0.50)
        score = r.get("score", 0.0)
        label = "PASS ✓" if score >= threshold else "FAIL ✗"
        scores.append(score)
        print(
            f"{task_name:<12} {r.get('steps', 0):<7} "
            f"{r.get('containment_rate', 0.0):<10.3f} {r.get('critical_health', 0.0):<10.3f} "
            f"{r.get('speed_bonus', 0.0):<8.3f} {score:<8.3f} {threshold:<8.2f} {label}"
        )
    print(dash)
    non_increasing = all(scores[i] >= scores[i+1] for i in range(len(scores)-1))
    passes = sum(
        1 for (t, r) in results if r.get("score", 0.0) >= TASK_THRESHOLDS.get(t, 0.50)
    )
    print(f"\nScores non-increasing: {'YES ✓' if non_increasing else 'NO ✗'}")
    print(f"Tasks passed: {passes}/{len(results)}")
    print(sep)

    print("\n[CEILING BENCHMARK]")
    print("Running impossible task as ceiling reference...")
    # Run 1 episode of impossible — not included in pass/fail calculation
    imp_result = run_task("impossible")
    print(f"Impossible raw score: {imp_result.get('score', 0.0):.4f}  "
          f"(no threshold — any score > 0.0 is meaningful)")


if __name__ == "__main__":
    if not _check_env_vars():
        raise SystemExit(1)
    run()
