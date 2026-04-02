import random
import logging
import math
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from models import Observation, ActionRequest, Reward
import importlib.util as _ilu, sys as _sys, os as _os
_aa_spec = _ilu.spec_from_file_location(
    "adaptive_attacker",
    _os.path.join(_os.path.dirname(__file__), "engines", "adaptive_attacker.py"),
)
_aa_mod = _ilu.module_from_spec(_aa_spec)
_sys.modules["adaptive_attacker"] = _aa_mod   # must register before exec for @dataclass
_aa_spec.loader.exec_module(_aa_mod)
AdaptiveAttacker = _aa_mod.AdaptiveAttacker

# ─── DEBUG ────────────────────────────────────────────────────────────────────
DEBUG = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.WARNING)
log = logging.getLogger("cyber_defense")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATTACKS = ["phishing", "malware", "ddos", "ransomware", "lateral_movement"]

CORRECT_ACTION = {
    "phishing": "block_ip",
    "malware": "isolate_machine",
    "ddos": "patch",
    "ransomware": "isolate_machine",
    "lateral_movement": "block_ip",
}

MITRE_MAP = {
    "phishing": "T1566",
    "malware": "T1204",
    "ddos": "T1499",
    "ransomware": "T1486",
    "lateral_movement": "T1021",
}

EXPLAIN = {
    "phishing": {
        "correct": "Phishing attack detected (T1566). Blocking the source IP prevents credential theft and halts the initial access vector.",
        "wrong": "Phishing requires blocking the source IP. Other mitigations do not stop credential harvesting.",
        "ignore": "Ignoring phishing allows the attacker to harvest credentials — severe health impact.",
    },
    "malware": {
        "correct": "Malware execution detected (T1204). Isolating the machine cuts off C2 communication and stops lateral spread.",
        "wrong": "Malware requires machine isolation. Blocking IPs or patching alone does not stop active execution.",
        "ignore": "Ignoring active malware allows it to spread to adjacent nodes — critical health loss.",
    },
    "ddos": {
        "correct": "DDoS attack detected (T1499). Patching the exposed service mitigates the volumetric impact.",
        "wrong": "DDoS requires patching the target service. Isolation or IP blocking does not absorb volumetric traffic.",
        "ignore": "Ignoring a DDoS degrades service availability rapidly.",
    },
    "ransomware": {
        "correct": "Ransomware detected (T1486). Isolating the machine prevents encryption from spreading to network shares.",
        "wrong": "Ransomware requires machine isolation to stop file encryption propagation.",
        "ignore": "Ignoring ransomware allows full disk encryption — catastrophic health loss.",
    },
    "lateral_movement": {
        "correct": "Lateral movement detected (T1021). Blocking the attacker IP stops traversal to new hosts.",
        "wrong": "Lateral movement requires IP blocking to cut the attacker's pivot path.",
        "ignore": "Ignoring lateral movement allows the attacker to compromise additional nodes.",
    },
}

TOTAL_NODES = 5
NODES = [f"node_{i}" for i in range(1, TOTAL_NODES + 1)]

VALID_ACTIONS = frozenset(
    ["block_ip", "isolate_machine", "patch", "ignore"]
    + [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)]
)

MAX_ACTION_LEN = 64
MAX_REWARD = 2.0
MIN_REWARD = -2.0

TECHNIQUE_DEFAULTS = {
    "phishing":         ("T1566", "Phishing",                       "Initial Access"),
    "malware":          ("T1204", "User Execution",                  "Execution"),
    "ddos":             ("T1499", "Endpoint Denial of Service",      "Impact"),
    "ransomware":       ("T1486", "Data Encrypted for Impact",       "Impact"),
    "lateral_movement": ("T1021", "Remote Services",                 "Lateral Movement"),
}

TASKS = [
    {"id": 1, "difficulty": "easy",      "goal": "Detect and block a phishing attack before it harvests credentials"},
    {"id": 2, "difficulty": "medium",    "goal": "Stop malware spread before it reaches lateral movement stage"},
    {"id": 3, "difficulty": "hard",      "goal": "Handle multi-stage attacks across multiple nodes simultaneously"},
    {"id": 4, "difficulty": "nightmare", "goal": "Defend against multiple hidden threats under limited visibility and no scan budget"},
]

# Per-task overrides applied at reset time
TASK_OVERRIDES = {
    "easy":      {"threat_count": 3, "age_visibility_threshold": 5},
    "medium":    {"threat_count": 4, "age_visibility_threshold": 5},
    "hard":      {"threat_count": 5, "age_visibility_threshold": 5},
    "nightmare": {"threat_count": 5, "age_visibility_threshold": 8},  # threats stay hidden longer
}

# Active task config (mutated on each /reset)
current_task_config: dict = TASK_OVERRIDES["easy"]
current_task_name:   str  = "easy"

# Action translation: app.py lowercase → attacker uppercase
ACTION_TRANSLATION = {
    "block_ip":        "BLOCK_IP",
    "isolate_machine": "ISOLATE_NODE",
    "patch":           "PATCH_VULNERABILITY",
    "ignore":          "IGNORE",
}

def translate_action(raw_action: str) -> str:
    """Translate a lowercase app.py action to the uppercase name AdaptiveAttacker expects."""
    if raw_action.startswith("scan"):
        return "SCAN"
    return ACTION_TRANSLATION.get(raw_action, raw_action.upper())

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Adaptive Cyber Defense", version="2.0.0")

# ─── RED TEAM ─────────────────────────────────────────────────────────────────
adaptive_attacker = AdaptiveAttacker(seed=42)
current_attack_plan: dict = {}

# ─── STATE ────────────────────────────────────────────────────────────────────
def _make_threats():
    return [
        {
            "type": random.choice(ATTACKS),
            "node": random.choice(NODES),
            "visible": False,
            "age": 0,
            "stage": "initial",
            "contained": False,
            "mitre_id": MITRE_MAP.get(random.choice(ATTACKS), "T0000"),
        }
        for _ in range(3)
    ]


def _make_threats_fixed():
    """Make threats with correct mitre_id per type (called after type is set)."""
    threats = []
    count = current_task_config.get("threat_count", 3)
    for _ in range(count):
        t_type = random.choice(ATTACKS)
        threats.append({
            "type": t_type,
            "node": random.choice(NODES),
            "visible": False,
            "age": 0,
            "stage": "initial",
            "contained": False,
            "mitre_id": MITRE_MAP[t_type],
        })
    return threats


def _fresh_state():
    return {
        "threats": _make_threats_fixed(),
        "scanned_nodes": set(),
        "system_health": 100,
        "score": 0.0,
        "step": 0,
        "done": False,
    }


state = _fresh_state()
history = []
episode_history = []

# ── Analytics tracking ────────────────────────────────────────────────────────
episode_actions_taken: list = []
episode_rewards: list = []
threats_detected_this_episode: set = set()
threats_contained_this_episode: set = set()
false_positive_actions: int = 0


def _reset_state():
    global history, episode_history
    global episode_actions_taken, episode_rewards
    global threats_detected_this_episode, threats_contained_this_episode
    global false_positive_actions
    fresh = _fresh_state()
    state.update(fresh)
    history = []
    episode_history = []
    episode_actions_taken = []
    episode_rewards = []
    threats_detected_this_episode = set()
    threats_contained_this_episode = set()
    false_positive_actions = 0


def _validate_state():
    try:
        assert isinstance(state["threats"], list)
        assert isinstance(state["scanned_nodes"], set)
        assert isinstance(state["system_health"], (int, float))
        assert isinstance(state["score"], (int, float))
        assert isinstance(state["step"], int)
        assert isinstance(state["done"], bool)
        if not math.isfinite(state["system_health"]):
            raise ValueError("system_health non-finite")
        if not math.isfinite(state["score"]):
            raise ValueError("score non-finite")
    except Exception as e:
        log.error(f"State corruption detected — auto-resetting: {e}")
        _reset_state()


def _clamp_health():
    state["system_health"] = max(0, min(100, state["system_health"]))


def _clamp_reward(r: float) -> float:
    if not math.isfinite(r):
        return 0.5
    # Normalized reward to [0,1] for OpenEnv compliance
    normalized_reward = (float(r) + 2.0) / 4.0
    normalized_reward = max(0.0, min(1.0, normalized_reward))
    return normalized_reward


def _clamp_score():
    if not math.isfinite(state["score"]):
        state["score"] = 0.0


# ─── LOGIC ────────────────────────────────────────────────────────────────────
def enrich_threat(threat: dict) -> dict:
    """Ensure every visible threat has all required fields with safe defaults."""
    if not isinstance(threat, dict):
        return threat

    t_type = threat.get("type", "malware")

    tech_id, tech_name, tactic = TECHNIQUE_DEFAULTS.get(
        t_type,
        ("T1204", "User Execution", "Execution"),
    )

    threat["id"]    = str(threat.get("id", f"{t_type}_{threat.get('node', 'unknown')}"))
    threat["type"]  = str(t_type)
    threat["node"]  = str(threat.get("node", "node_1"))
    threat["stage"] = str(threat.get("stage", "initial"))

    threat["age"]      = int(threat.get("age", 0))
    threat["severity"] = float(threat.get("severity", 0.5))

    threat["technique_id"]   = threat.get("technique_id")   or tech_id
    threat["technique_name"] = threat.get("technique_name") or tech_name
    threat["tactic"]         = threat.get("tactic")         or tactic

    threat["detection_confidence"] = float(threat.get("detection_confidence", 1.0))

    return threat


def _update_visibility():
    age_thresh = current_task_config.get("age_visibility_threshold", 5)
    for t in state["threats"]:
        if (
            t["node"] in state["scanned_nodes"]
            or t["age"] >= age_thresh
            or t["stage"] == "lateral_movement"
        ):
            t["visible"] = True


def _age_threats():
    for t in state["threats"]:
        if not t.get("contained"):
            t["age"] += 1
            if t["age"] >= 8 and t["stage"] == "initial":
                t["stage"] = "lateral_movement"


def _visible_threats():
    out = []
    for t in state["threats"]:
        if t["visible"] and not t.get("contained"):
            out.append({
                "type": t["type"],
                "node": t["node"],
                "stage": t["stage"],
                "age": t["age"],
                "mitre_id": t.get("mitre_id", MITRE_MAP.get(t["type"], "T0000")),
            })
    return [enrich_threat(t) for t in out]


def _obs():
    scanned = state["scanned_nodes"]
    return {
        "visible_threats": _visible_threats(),
        "hidden_node_count": TOTAL_NODES - len(scanned),
        "scan_coverage": round(len(scanned) / TOTAL_NODES, 2),
        "system_health": state["system_health"],
        "score": round(state["score"], 2),
        "step": state["step"],
        "done": state["done"],
    }


def _build_reason(action: str, matched: bool, threat_type: str | None, early: bool) -> tuple[str, float]:
    """Return (reason_string, confidence)."""
    if action.startswith("scan"):
        return ("Scanning reveals hidden threats before they escalate. Essential under partial observability.", 0.85)

    if threat_type is None:
        return ("No visible threat to act on. Scanning unexplored nodes is recommended.", 0.60)

    ex = EXPLAIN.get(threat_type, {})
    if matched:
        base_conf = 0.92
        reason = ex.get("correct", f"Correct mitigation for {threat_type} ({MITRE_MAP.get(threat_type, '')}).")
        if early:
            reason += " Early neutralization bonus applied."
            base_conf = min(1.0, base_conf + 0.05)
        return (reason, base_conf)
    elif action == "ignore":
        return (ex.get("ignore", f"Ignoring {threat_type} allows escalation."), 0.20)
    else:
        return (ex.get("wrong", f"Wrong mitigation for {threat_type}. Check MITRE technique {MITRE_MAP.get(threat_type, '')}."), 0.35)


def safe_response(obs, action, reward=0.0, reason="", confidence=0.0, error=None):
    resp = {
        "action": action,
        "reward": round(float(reward), 3),
        "visible_threats": obs.get("visible_threats", []),
        "hidden_node_count": obs.get("hidden_node_count", TOTAL_NODES),
        "scan_coverage": obs.get("scan_coverage", 0.0),
        "system_health": obs.get("system_health", 100),
        "score": obs.get("score", 0.0),
        "step": obs.get("step", 0),
        "done": obs.get("done", False),
        "reason": reason,
        "confidence": round(float(confidence), 2),
    }
    if error is not None:
        resp["error"] = error
    return resp


# ─── EXCEPTION HANDLERS ───────────────────────────────────────────────────────
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if DEBUG:
        log.debug(f"Validation error: {exc.errors()}")
    _validate_state()
    obs = _obs()
    return JSONResponse(
        status_code=200,
        content=safe_response(obs, action="", reward=-0.5,
                               reason="Invalid input received. Action rejected.",
                               confidence=0.0, error="invalid action"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    try:
        _validate_state()
        obs = _obs()
    except Exception:
        _reset_state()
        obs = _obs()
    return JSONResponse(
        status_code=200,
        content=safe_response(obs, action="", reward=-0.5,
                               reason="Internal error. State preserved.",
                               confidence=0.0, error="internal error"),
    )


# ─── INPUT MODELS ─────────────────────────────────────────────────────────────
# Action (request body) and Observation/Reward (response) are imported from models.

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: int = 0


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Adaptive Cyber Defense API v2.0 — OpenEnv compatible"}


@app.get("/tasks")
def get_tasks():
    return TASKS


@app.get("/history")
def get_history():
    return {
        "episode_steps": episode_history,
        "total_steps": len(episode_history),
        "total_reward": round(sum(s["reward"] for s in episode_history), 4),
        "final_status": "done" if episode_history and episode_history[-1]["done"] else "in_progress",
    }


@app.get("/reset")
@app.get("/reset/")
@app.post("/reset")
@app.post("/reset/")
def reset(req: ResetRequest = None):
    global current_attack_plan, current_task_config, current_task_name
    task_name = (req.task.lower().strip() if req and req.task else "easy")
    if task_name not in TASK_OVERRIDES:
        task_name = "easy"
    current_task_name   = task_name
    current_task_config = TASK_OVERRIDES[task_name]
    _reset_state()
    current_attack_plan = adaptive_attacker.on_episode_start()
    obs = _obs()
    obs["task"]        = task_name
    obs["attack_plan"] = current_attack_plan
    return obs


@app.get("/state", response_model=Observation)
@app.get("/state/", response_model=Observation)
@app.post("/state", response_model=Observation)
@app.post("/state/", response_model=Observation)
def get_state():
    _validate_state()
    return Observation(**_obs())


@app.post("/step")
def step(req: ActionRequest):
    try:
        _validate_state()

        if state["done"]:
            if DEBUG:
                log.debug("step() called after done=True")
            return safe_response(
                _obs(), action="", reward=0.0,
                reason="Episode is over. Call /reset to start a new episode.",
                confidence=1.0, error="Episode over — call /reset",
            )

        raw_action = req.action.strip().lower()

        # Unknown action — safe fallback, full obs returned
        if raw_action not in VALID_ACTIONS:
            if DEBUG:
                log.debug(f"Unknown action: {raw_action!r}")
            reward = _clamp_reward(-0.5)
            state["system_health"] = max(0, state["system_health"] - 5)
            _age_threats()
            _update_visibility()
            _clamp_health()
            state["score"] += reward
            _clamp_score()
            state["step"] += 1
            if state["system_health"] <= 0 or state["step"] >= 50:
                state["done"] = True
            obs = _obs()
            history.append({"step": state["step"], "action": raw_action,
                             "reward": round(reward, 3), "attack": None})
            return safe_response(
                obs, action=raw_action, reward=reward,
                reason=f"'{raw_action}' is not a recognised action. Valid: block_ip, isolate_machine, patch, ignore, scan_node_1..5.",
                confidence=0.0, error="invalid action",
            )

        reward = 0.0
        reason = ""
        confidence = 0.5
        matched = False
        early_bonus = False
        matched_threat_type = None

        # ── SCAN ──
        if raw_action.startswith("scan"):
            parts = raw_action.replace("scan_", "scan ").split()
            node = parts[1] if len(parts) > 1 else ""
            if node in NODES:
                state["scanned_nodes"].add(node)
                revealed = any(
                    t["node"] == node and not t["visible"] and not t.get("contained")
                    for t in state["threats"]
                )
                for t in state["threats"]:
                    if t["node"] == node and not t["visible"] and not t.get("contained"):
                        t["visible"] = True
                reward = 0.02 if revealed else -0.01
                reason, confidence = _build_reason(raw_action, False, None, False)
                if revealed:
                    reason = f"Scan of {node} revealed a hidden threat. Partial observability lifted for this node."
                    confidence = 0.90
                else:
                    reason = f"Scan of {node} found no new threats. Coverage improved."
                    confidence = 0.75
            else:
                reward = -0.01
                reason = f"'{node}' is not a valid node. Valid nodes: node_1 through node_5."
                confidence = 0.10

        # ── DEFENSE ──
        else:
            for t in state["threats"]:
                if t["visible"] and not t.get("contained"):
                    correct = CORRECT_ACTION.get(t["type"], "")
                    if raw_action == correct:
                        t["contained"] = True
                        matched_threat_type = t["type"]
                        matched = True
                        # Reward: 1 + health bonus
                        base = 1.0 + (state["system_health"] / 100.0)
                        # Early neutralization bonus (age < 3)
                        if t["age"] < 3:
                            base += 0.1
                            early_bonus = True
                        reward += base
                        break

            if not matched:
                # Find first visible uncontained threat type for context
                for t in state["threats"]:
                    if t["visible"] and not t.get("contained"):
                        matched_threat_type = t["type"]
                        break

                if raw_action == "ignore":
                    reward = -1.5
                    state["system_health"] = max(0, state["system_health"] - 10)
                else:
                    # Wrong action: -0.5 - health penalty component
                    reward = -0.5 - (10 - min(state["system_health"], 10)) / 20.0
                    state["system_health"] = max(0, state["system_health"] - 5)

            reason, confidence = _build_reason(raw_action, matched, matched_threat_type, early_bonus)

        reward = _clamp_reward(reward)
        _age_threats()
        _update_visibility()
        _clamp_health()

        state["score"] += reward
        _clamp_score()
        state["step"] += 1

        if state["system_health"] <= 0 or state["step"] >= 50:
            state["done"] = True

        obs = _obs()

        history.append({
            "step": state["step"],
            "action": raw_action,
            "reward": round(reward, 3),
            "attack": matched_threat_type,
        })

        global episode_history
        episode_history.append({
            "step": state["step"],
            "action": raw_action,
            "reward": float(reward),
            "done": bool(state["done"]),
            "reason": reason,
        })

        # ── Analytics tracking ────────────────────────────────────────────────
        global episode_actions_taken, episode_rewards
        global threats_detected_this_episode, threats_contained_this_episode
        global false_positive_actions
        episode_actions_taken.append(raw_action)
        episode_rewards.append(reward)
        _MITIGATIONS = {"block_ip", "isolate_machine", "patch"}
        if raw_action in _MITIGATIONS and reward < _clamp_reward(0.0):
            false_positive_actions += 1
        for threat in obs.get("visible_threats", []):
            tid = threat.get("id", "")
            if tid:
                threats_detected_this_episode.add(tid)
            if threat.get("stage") == "contained":
                threats_contained_this_episode.add(tid)
        # also mark threats contained via matched flag
        if matched and matched_threat_type:
            for t in state["threats"]:
                if t.get("contained") and t.get("type") == matched_threat_type:
                    tid = str(t.get("id", f"{t['type']}_{t['node']}"))
                    threats_contained_this_episode.add(tid)

        # ── Red team: observe defender action ─────────────────────────────────
        translated = translate_action(raw_action)
        threat_ctx = (matched_threat_type or "UNKNOWN").upper()
        adaptive_attacker.observe_defender_action(translated, threat_ctx)

        # ── Red team: episode end update ──────────────────────────────────────
        if state["done"]:
            contained = sum(1 for t in state["threats"] if t.get("contained"))
            total     = max(1, len(state["threats"]))
            defender_won = (contained / total) >= 0.8
            adaptive_attacker.on_episode_end(
                defender_won=defender_won,
                score=round(state["score"], 4),
            )

        return safe_response(obs, action=raw_action, reward=reward,
                             reason=reason, confidence=confidence)

    except Exception as e:
        log.error(f"/step unhandled exception: {e}", exc_info=True)
        try:
            _validate_state()
            obs = _obs()
        except Exception:
            _reset_state()
            obs = _obs()
        return safe_response(obs, action="", reward=_clamp_reward(-0.5),
                             reason="Unexpected error. State preserved.",
                             confidence=0.0, error="invalid action")


@app.get("/attacker-report")
def attacker_report():
    p = adaptive_attacker.defender_profile
    return {
        "episode_count":     adaptive_attacker.episode_count,
        "current_strategy":  adaptive_attacker.current_strategy,
        "defender_profile": {
            "strategy_label":   p.get_defender_strategy_label(),
            "isolation_rate":   round(p.isolation_rate, 3),
            "block_rate":       round(p.block_rate,     3),
            "scan_rate":        round(p.scan_rate,      3),
            "patch_rate":       round(p.patch_rate,     3),
            "most_used_action": p.get_most_used_action(),
            "steps_observed":   p.steps_observed,
            "action_counts":    dict(p.action_counts),
        },
        "strategy_history": adaptive_attacker.strategy_history[-5:],
        "adaptation_log":   adaptive_attacker.adaptation_log[-10:],
        "full_report":      adaptive_attacker.get_full_adaptation_report(),
    }


MITRE_INTEL = {
    "phishing": {
        "technique_id":   "T1566",
        "technique_name": "Phishing",
        "tactic":         "Initial Access",
        "tactic_id":      "TA0001",
        "severity":       "HIGH",
        "recommended_action": "block_ip",
        "description": "Adversary sending malicious emails to gain initial access",
        "indicators": ["suspicious_email", "malicious_attachment", "fake_link"],
        "mitigation": "Block source IP, enable email filtering, user awareness training",
        "kill_chain_phase": "delivery",
        "similar_incidents": 3,
    },
    "malware": {
        "technique_id":   "T1204",
        "technique_name": "User Execution",
        "tactic":         "Execution",
        "tactic_id":      "TA0002",
        "severity":       "CRITICAL",
        "recommended_action": "isolate_machine",
        "description": "Malicious code executing on compromised endpoint",
        "indicators": ["unusual_process", "file_modification", "registry_change"],
        "mitigation": "Isolate affected machine, run forensic analysis, restore from backup",
        "kill_chain_phase": "exploitation",
        "similar_incidents": 7,
    },
    "ddos": {
        "technique_id":   "T1499",
        "technique_name": "Endpoint Denial of Service",
        "tactic":         "Impact",
        "tactic_id":      "TA0040",
        "severity":       "HIGH",
        "recommended_action": "patch",
        "description": "Flooding target service to cause denial of service",
        "indicators": ["high_traffic", "service_unavailable", "cpu_spike"],
        "mitigation": "Apply rate limiting patch, enable DDoS protection, null-route attacker",
        "kill_chain_phase": "actions_on_objectives",
        "similar_incidents": 2,
    },
    "ransomware": {
        "technique_id":   "T1486",
        "technique_name": "Data Encrypted for Impact",
        "tactic":         "Impact",
        "tactic_id":      "TA0040",
        "severity":       "CRITICAL",
        "recommended_action": "isolate_machine",
        "description": "Encrypting files to extort ransom from victim organization",
        "indicators": ["file_encryption", "ransom_note", "shadow_copy_deletion"],
        "mitigation": "Immediately isolate machine, do not pay ransom, restore from backup",
        "kill_chain_phase": "actions_on_objectives",
        "similar_incidents": 5,
    },
    "lateral_movement": {
        "technique_id":   "T1021",
        "technique_name": "Remote Services",
        "tactic":         "Lateral Movement",
        "tactic_id":      "TA0008",
        "severity":       "HIGH",
        "recommended_action": "block_ip",
        "description": "Adversary moving through network using remote services",
        "indicators": ["unusual_login", "remote_access", "credential_reuse"],
        "mitigation": "Block internal IP, reset credentials, enable MFA",
        "kill_chain_phase": "lateral_movement",
        "similar_incidents": 4,
    },
}

SEVERITY_ORDER = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "UNKNOWN": 4}


@app.get("/threat-intel")
def threat_intel():
    """Returns MITRE ATT&CK enriched threat intelligence about active threats."""
    try:
        _validate_state()
        visible_threats = _visible_threats()
        scan_coverage   = round(len(state["scanned_nodes"]) / TOTAL_NODES, 3)
        system_health   = state["system_health"]

        active_campaigns = []
        for threat in visible_threats:
            threat_type = threat.get("type", "unknown").lower()
            intel = MITRE_INTEL.get(threat_type, {})
            active_campaigns.append({
                "threat_id":          threat.get("id", f"{threat_type}_{threat.get('node', 'unknown')}"),
                "threat_type":        threat_type,
                "node":               threat.get("node", "unknown"),
                "stage":              threat.get("stage", "unknown"),
                "age":                threat.get("age", 0),
                "severity":           intel.get("severity", "UNKNOWN"),
                "technique_id":       intel.get("technique_id", threat.get("technique_id", "")),
                "technique_name":     intel.get("technique_name", threat.get("technique_name", "")),
                "tactic":             intel.get("tactic", threat.get("tactic", "")),
                "tactic_id":          intel.get("tactic_id", ""),
                "recommended_action": intel.get("recommended_action", "ignore"),
                "description":        intel.get("description", ""),
                "indicators":         intel.get("indicators", []),
                "mitigation":         intel.get("mitigation", ""),
                "kill_chain_phase":   intel.get("kill_chain_phase", ""),
                "similar_incidents":  intel.get("similar_incidents", 0),
                "confidence":         round(threat.get("detection_confidence", 1.0), 3),
                "urgency":            "IMMEDIATE" if threat.get("age", 0) >= 3 else "MONITOR",
            })

        active_campaigns.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"], 4))

        # Derive compromised nodes from lateral_movement threats
        compromised = list({
            t["node"] for t in state["threats"]
            if t.get("stage") == "lateral_movement" and not t.get("contained")
        })

        if system_health < 30 or len(compromised) >= 3:
            risk_level = "CRITICAL"
        elif system_health < 60 or len(compromised) >= 2:
            risk_level = "HIGH"
        elif system_health < 80 or len(compromised) >= 1:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "timestamp": state["step"],
            "risk_level": risk_level,
            "active_campaigns": active_campaigns,
            "threat_summary": {
                "total_visible":    len(visible_threats),
                "critical_count":   sum(1 for t in active_campaigns if t["severity"] == "CRITICAL"),
                "high_count":       sum(1 for t in active_campaigns if t["severity"] == "HIGH"),
                "immediate_action": sum(1 for t in active_campaigns if t["urgency"] == "IMMEDIATE"),
            },
            "network_assessment": {
                "risk_level":      risk_level,
                "compromised_nodes": compromised,
                "scan_coverage":   scan_coverage,
                "system_health":   system_health,
                "unscanned_nodes": max(0, TOTAL_NODES - len(state["scanned_nodes"])),
            },
            "recommended_actions": [
                t["recommended_action"]
                for t in active_campaigns
                if t["urgency"] == "IMMEDIATE"
            ][:3],
            "mitre_framework": "ATT&CK v14.0",
        }

    except Exception as e:
        log.error(f"/threat-intel error: {e}", exc_info=True)
        return {
            "timestamp":          0,
            "risk_level":         "UNKNOWN",
            "active_campaigns":   [],
            "threat_summary":     {"total_visible": 0},
            "network_assessment": {},
            "recommended_actions": [],
            "error":              str(e),
            "mitre_framework":    "ATT&CK v14.0",
        }


@app.get("/analytics")
def get_analytics():
    """Returns real SOC metrics for the current episode."""
    try:
        _validate_state()
        visible_threats = _visible_threats()
        total_steps   = state["step"]
        system_health = state["system_health"]
        scan_coverage = round(len(state["scanned_nodes"]) / TOTAL_NODES, 3)

        n_detected  = max(1, len(threats_detected_this_episode))
        n_contained = len(threats_contained_this_episode)

        mttd = round(total_steps / n_detected, 2)
        mttr = round(total_steps / max(1, n_contained), 2)

        # Detection rate: detected / (detected + remaining hidden)
        hidden_count = TOTAL_NODES - len(state["scanned_nodes"])
        detection_rate = round(
            len(threats_detected_this_episode) /
            max(1, len(threats_detected_this_episode) + hidden_count), 3
        )

        containment_rate = round(n_contained / n_detected, 3)

        total_mitigations = sum(
            1 for a in episode_actions_taken
            if a in {"block_ip", "isolate_machine", "patch"}
        )
        false_positive_rate = round(
            false_positive_actions / max(1, total_mitigations), 3
        )

        avg_reward = round(
            sum(episode_rewards) / max(1, len(episode_rewards)), 4
        )

        if len(episode_rewards) >= 10:
            first_5 = sum(episode_rewards[:5]) / 5
            last_5  = sum(episode_rewards[-5:]) / 5
            trend = "IMPROVING" if last_5 > first_5 else "DECLINING"
        else:
            trend = "INSUFFICIENT_DATA"

        action_counts: dict = {}
        for a in episode_actions_taken:
            action_counts[a] = action_counts.get(a, 0) + 1
        most_used = max(action_counts, key=action_counts.get) if action_counts else "none"

        if containment_rate >= 0.8 and system_health >= 70:
            grade = "A"
        elif containment_rate >= 0.6 and system_health >= 50:
            grade = "B"
        elif containment_rate >= 0.4 and system_health >= 30:
            grade = "C"
        else:
            grade = "D"

        # Recommended next action
        types_visible = [t.get("type") for t in visible_threats]
        if "phishing" in types_visible:
            recommended = "block_ip"
        elif any(t in types_visible for t in ("malware", "ransomware")):
            recommended = "isolate_machine"
        elif "ddos" in types_visible:
            recommended = "patch"
        elif scan_coverage < 1.0:
            scanned = state["scanned_nodes"]
            unscanned = [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)
                         if f"node_{i}" not in scanned]
            recommended = unscanned[0] if unscanned else "ignore"
        else:
            recommended = "ignore"

        compromised = list({
            t["node"] for t in state["threats"]
            if t.get("stage") == "lateral_movement" and not t.get("contained")
        })

        return {
            "episode_step":      total_steps,
            "performance_grade": grade,
            "soc_metrics": {
                "mean_time_to_detect":  mttd,
                "mean_time_to_respond": mttr,
                "detection_rate":       detection_rate,
                "containment_rate":     containment_rate,
                "false_positive_rate":  false_positive_rate,
                "avg_reward_per_step":  avg_reward,
                "reward_trend":         trend,
            },
            "threat_tracking": {
                "threats_detected":      len(threats_detected_this_episode),
                "threats_contained":     n_contained,
                "threats_active":        len(visible_threats),
                "threats_ids_detected":  list(threats_detected_this_episode),
            },
            "network_status": {
                "system_health":     system_health,
                "scan_coverage":     scan_coverage,
                "compromised_nodes": compromised,
                "nodes_at_risk":     len(compromised),
            },
            "agent_behavior": {
                "total_actions":    len(episode_actions_taken),
                "action_breakdown": action_counts,
                "most_used_action": most_used,
                "false_positives":  false_positive_actions,
            },
            "recommended_next_action": recommended,
            "attacker_strategy": adaptive_attacker.current_strategy,
        }

    except Exception as e:
        log.error(f"/analytics error: {e}", exc_info=True)
        return {
            "episode_step":      0,
            "performance_grade": "UNKNOWN",
            "soc_metrics":       {},
            "threat_tracking":   {},
            "network_status":    {},
            "agent_behavior":    {},
            "recommended_next_action": "ignore",
            "error": str(e),
        }


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
