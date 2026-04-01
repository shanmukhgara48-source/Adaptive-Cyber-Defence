import random
import logging
import math
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from models import Observation, Action, Reward

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

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="Adaptive Cyber Defense", version="2.0.0")


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
    for _ in range(3):
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


def _reset_state():
    global history, episode_history
    fresh = _fresh_state()
    state.update(fresh)
    history = []
    episode_history = []


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
    for t in state["threats"]:
        if (
            t["node"] in state["scanned_nodes"]
            or t["age"] >= 5
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


# ─── INPUT MODEL ──────────────────────────────────────────────────────────────
# Action (request body) and Observation/Reward (response) are imported from models.


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


@app.get("/reset", response_model=Observation)
@app.get("/reset/", response_model=Observation)
@app.post("/reset", response_model=Observation)
@app.post("/reset/", response_model=Observation)
def reset():
    _reset_state()
    return Observation(**_obs())


@app.get("/state", response_model=Observation)
@app.get("/state/", response_model=Observation)
@app.post("/state", response_model=Observation)
@app.post("/state/", response_model=Observation)
def get_state():
    _validate_state()
    return Observation(**_obs())


@app.post("/step")
def step(req: Action):
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


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
