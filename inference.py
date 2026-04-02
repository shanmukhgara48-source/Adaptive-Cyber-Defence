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
MAX_STEPS   = 20
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds between retries

MODELS = ["meta-llama/Meta-Llama-3-8B-Instruct"]
TASKS  = ["easy", "medium", "hard", "nightmare"]

# ---------------------------------------------------------------------------
# LLM-based action selection
# ---------------------------------------------------------------------------

def choose_action(
    obs: dict,
    step_num: int,
    last_action: str,
    last_reward: float,
    scanned_nodes: set,
) -> str:
    """Build a context-rich prompt and ask the LLM for exactly one action."""
    threats      = obs.get("visible_threats", [])
    coverage     = obs.get("scan_coverage", 1.0)
    system_state = obs.get("system_state", {})

    all_nodes = [f"node_{i}" for i in range(1, 6)]
    unscanned = [n for n in all_nodes if n not in scanned_nodes]

    prompt = f"""You are a cybersecurity defense agent.
Choose exactly ONE action to take this step.

Current State:
- Step: {step_num}/{MAX_STEPS}
- Visible threats: {threats}
- Scan coverage: {coverage:.2f}
- System state: {system_state}
- Already scanned nodes: {sorted(scanned_nodes)}
- Unscanned nodes: {unscanned}
- Last action: {last_action}
- Last reward: {last_reward:.3f}

Threat-to-action mapping (use this exactly):
- phishing / T1566                        → block_ip
- ddos / T1499                            → patch
- malware / T1204 / T1059                 → isolate_machine
- ransomware / T1486                      → isolate_machine
- lateral_movement / T1021                → block_ip
- no threats + scan_coverage < 0.9       → scan an unscanned node
- no threats + scan_coverage >= 0.9      → ignore

Decision rules (follow strictly in this order):
1. If visible_threats contains phishing, T1566, lateral_movement, T1021 → respond: block_ip
2. If visible_threats contains ddos, T1499 → respond: patch
3. If visible_threats contains malware, ransomware, T1204, T1059, T1486 → respond: isolate_machine
4. If visible_threats is not empty but type is unknown → respond: block_ip
5. If any node in system_state is 'compromised' → respond: isolate_machine
6. If scan_coverage < 0.9 AND unscanned nodes exist → respond with the scan action for the first unscanned node (e.g. scan_node_2 if node_2 is unscanned)
7. Otherwise → respond: ignore

Available actions: block_ip, isolate_machine, patch, ignore,
scan_node_1, scan_node_2, scan_node_3, scan_node_4, scan_node_5

Respond with ONLY the action string. No explanation. No punctuation. No extra words."""

    for model in MODELS:
        print(f"[model] trying: {model}")
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=20,
                    temperature=TEMPERATURE,
                )
                action = response.choices[0].message.content.strip().lower()
                action = action.split()[0] if action else "ignore"
                action = action.strip(".,!?")
                if action not in VALID_ACTIONS:
                    print(f"[model] invalid action '{action}' — falling back to ignore")
                    action = "ignore"
                return action
            except Exception as e:
                print(f"[model] {model} attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)

        print(f"[model] {model} exhausted retries — trying next model")

    print("[model] all models failed — returning fallback: ignore")
    return "ignore"


# ---------------------------------------------------------------------------
# Single-task interaction loop
# ---------------------------------------------------------------------------

def run_task(task_name: str) -> dict:
    """Run one full episode for the given task. Returns a result summary dict."""
    print(f"\n{'=' * 60}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'=' * 60}")

    # Reset environment for this task
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

    task_id = reset_data.get("task_id", task_name)

    total_reward  = 0.0
    final_status  = "in_progress"
    last_action   = "none"
    last_reward   = 0.0
    scanned_nodes = set()
    step_num      = 0

    print(f"{'Step':<6} {'Action':<22} {'Reward':<8} {'Reason'}")
    print("-" * 70)

    for step_num in range(1, MAX_STEPS + 1):
        # Fetch current observation
        try:
            obs = requests.get(f"{BASE_URL}/state").json()
        except Exception as e:
            print(f"[error] /state failed at step {step_num}: {e}")
            final_status = "error"
            break

        if obs.get("done"):
            final_status = "done"
            break

        # LLM picks the action
        action = choose_action(obs, step_num, last_action, last_reward, scanned_nodes)

        # Track scanned nodes
        if action.startswith("scan_node_"):
            node_id = action.replace("scan_node_", "node_")
            scanned_nodes.add(node_id)

        # Execute action in environment
        try:
            data = requests.post(f"{BASE_URL}/step", json={"action": action}).json()
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

    score = max(0.0, min(1.0, total_reward / MAX_STEPS))

    return {
        "task_id":      task_id,
        "steps":        step_num,
        "total_reward": round(total_reward, 3),
        "score":        round(score, 3),
        "status":       final_status,
    }


# ---------------------------------------------------------------------------
# Main: run all tasks sequentially and print summary
# ---------------------------------------------------------------------------

def run():
    results = []

    for task_name in TASKS:
        result = run_task(task_name)
        results.append((task_name, result))

    # Final summary table
    sep  = "=" * 60
    dash = "-" * 60
    print(f"\n{sep}")
    print("BASELINE RESULTS")
    print(sep)
    print(f"{'Task':<12} {'Steps':<8} {'Total Reward':<16} {'Score':<8} {'Status'}")
    print(dash)
    for task_name, r in results:
        print(
            f"{task_name:<12} {r['steps']:<8} {r['total_reward']:<16} "
            f"{r['score']:<8} {r['status']}"
        )
    print(sep)


if __name__ == "__main__":
    run()
