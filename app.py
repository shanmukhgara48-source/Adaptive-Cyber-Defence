import random
import logging
import math
import uuid
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator
from models import Observation
import importlib.util as _ilu, sys as _sys, os as _os


# ─── TASK CONFIG IMPORTS ──────────────────────────────────────────────────────
from tasks.easy import EasyTask
from tasks.medium import MediumTask
from tasks.hard import HardTask
from tasks.nightmare import NightmareTask
from tasks.elite import EliteTask
from tasks.impossible import ImpossibleTask

TASK_MAP = {
    "easy":       EasyTask,
    "medium":     MediumTask,
    "hard":       HardTask,
    "nightmare":  NightmareTask,
    "elite":      EliteTask,
    "impossible": ImpossibleTask,
}
_aa_spec = _ilu.spec_from_file_location(
    "adaptive_attacker",
    _os.path.join(_os.path.dirname(__file__), "engines", "adaptive_attacker.py"),
)
_aa_mod = _ilu.module_from_spec(_aa_spec)
_sys.modules["adaptive_attacker"] = _aa_mod   # must register before exec for @dataclass
_aa_spec.loader.exec_module(_aa_mod)
AdaptiveAttacker = _aa_mod.AdaptiveAttacker

# ─── DEBUG ────────────────────────────────────────────────────────────────────
import os as _os
DEBUG = _os.getenv("DEBUG", "false").lower() == "true"
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.WARNING)
log = logging.getLogger("cyber_defense")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATTACKS = ["phishing", "malware", "ddos", "ransomware", "lateral_movement"]

MITRE_MAP = {
    "phishing": "T1566",
    "malware": "T1204",
    "ddos": "T1499",
    "ransomware": "T1486",
    "lateral_movement": "T1021",
}

# Pure MITRE ATT&CK mitigation lookup — matches openenv.yaml and inference.py prompt exactly.
# phishing/T1566 → block_ip, malware/T1204 → isolate_machine,
# ransomware/T1486 → isolate_machine, ddos/T1499 → patch,
# lateral_movement/T1021 → block_ip
_MITRE_CORRECT_ACTION: dict[str, str] = {
    "phishing":         "block_ip",
    "malware":          "isolate_machine",
    "ransomware":       "isolate_machine",
    "ddos":             "patch",
    "lateral_movement": "block_ip",
}


def _get_correct_action(threat_type: str, severity: float, stage: str) -> str:
    """Return the correct MITRE-aligned mitigation action for a threat type.
    Severity and stage parameters are accepted for API compatibility but not used —
    the mapping is deterministic by type only, matching the spec and agent prompts.
    """
    return _MITRE_CORRECT_ACTION.get(threat_type, "ignore")

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
    {"id": 1, "difficulty": "easy",       "passing_score": 0.50, "goal": "Three simultaneous attacks. High detection, generous resources. Contain all before lateral spread."},
    {"id": 2, "difficulty": "medium",     "passing_score": 0.60, "goal": "Two intrusions with limited resources, FP noise. Prioritise threats."},
    {"id": 3, "difficulty": "hard",       "passing_score": 0.45, "goal": "APT across 5 nodes. Low detection, scarce resources, fast progression."},
    {"id": 4, "difficulty": "nightmare",  "passing_score": 0.25, "goal": "Nation-state APT. Near-zero detection, 15 steps. Designed for frontier LLMs."},
    {"id": 5, "difficulty": "elite",      "passing_score": 0.20, "goal": "Persistent threat with insider access. All nodes pre-compromised. Kill chain advances every step."},
    {"id": 6, "difficulty": "impossible", "passing_score": 0.10, "goal": "AI-driven attacker with perfect counter-strategy. Exists to show environment has no ceiling."},
]


# Per-task config derived from task classes — all difficulty params now live there.
# age_visibility_threshold is HTTP-API-specific (not in TaskConfig).
def _derive_task_overrides() -> dict:
    out = {}
    for name, cls in TASK_MAP.items():
        cfg = cls.config
        out[name] = {
            "threat_count":              cfg.initial_threat_count,
            "max_steps":                 cfg.max_steps,
            "false_negative_rate":       cfg.false_negative_rate,
            "base_detection_prob":       cfg.base_detection_prob,
            "attack_progression_prob":   cfg.attack_progression_prob,
            "lateral_spread_base_prob":  cfg.lateral_spread_base_prob,
            "health_degradation_rate":   cfg.health_degradation_rate,
            # Previously missing — these are critical for difficulty scaling
            "resource_per_step":         cfg.resource_per_step,
            "natural_severity_growth":   cfg.natural_severity_growth,
            "false_positive_rate":       cfg.false_positive_rate,
            "passing_score":             cfg.passing_score,
            # Visibility threshold: nightmare/elite/impossible use 8 (longer hidden window)
            "age_visibility_threshold":  8 if name in ("nightmare", "elite", "impossible") else 5,
        }
    return out

TASK_OVERRIDES = _derive_task_overrides()

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
adaptive_attacker = AdaptiveAttacker(seed=int(_os.getenv("ATTACKER_SEED", "42")))

# ─── SESSION ──────────────────────────────────────────────────────────────────
# Each call to /reset creates an isolated Session.  Concurrent users/judges
# each hold their own session_id and never share state.

@dataclass
class Session:
    """All mutable per-episode state, isolated per user/judge."""
    task_name:   str
    task_config: dict
    state:       dict = field(default_factory=dict)
    history:     list = field(default_factory=list)
    episode_history: list = field(default_factory=list)
    episode_actions_taken: list = field(default_factory=list)
    episode_rewards:       list = field(default_factory=list)
    threats_detected:      set  = field(default_factory=set)
    threats_contained:     set  = field(default_factory=set)
    false_positive_actions: int = 0
    attack_plan: dict = field(default_factory=dict)


# Session store — keyed by UUID string.
# No default session — every client must use session_id from /reset.
_SESSIONS: dict[str, Session] = {}
# Maximum live sessions (evict oldest after this limit to prevent memory growth)
_MAX_SESSIONS = 256

# Kept for backward-compat in exception handlers that need a session key
_DEFAULT_SESSION_ID = "default"


def _evict_oldest_sessions():
    """Keep session count under _MAX_SESSIONS by dropping the oldest entries."""
    while len(_SESSIONS) >= _MAX_SESSIONS:
        oldest = next(iter(_SESSIONS))
        del _SESSIONS[oldest]


def _resolve_sid(session_id: str | None, sess: "Session") -> str:
    """Return the canonical session_id string for a response."""
    return session_id if session_id else _DEFAULT_SESSION_ID


def _get_session(session_id: str | None) -> Session | None:
    """Return the session for the given id, or None if not found.
    No default session is created — callers must handle None.
    """
    if not session_id:
        return None
    return _SESSIONS.get(session_id.strip())


# ─── STATE HELPERS (session-scoped) ──────────────────────────────────────────

# Initial severity ranges per attack type — vary so severity-based logic fires
_SEVERITY_RANGES = {
    "phishing":         (0.3, 0.6),
    "malware":          (0.5, 0.8),
    "ransomware":       (0.7, 1.0),
    "ddos":             (0.4, 0.7),
    "lateral_movement": (0.6, 0.9),
}


def _initial_severity(t_type: str) -> float:
    lo, hi = _SEVERITY_RANGES.get(t_type, (0.4, 0.7))
    return round(random.uniform(lo, hi), 3)


def _make_threats_fixed(task_config: dict) -> list:
    """Make threats for a new episode using the given task config."""
    threats = []
    count = task_config.get("threat_count", 3)
    for idx in range(count):
        t_type = random.choice(ATTACKS)
        node   = random.choice(NODES)
        threats.append({
            "id":            f"{t_type}_{node}_{idx}",
            "type":          t_type,
            "original_type": t_type,  # preserved even after stage escalation
            "node":          node,
            "visible":       False,
            "age":           0,
            "stage":         "initial",
            "escalated":     False,
            "contained":     False,
            "mitre_id":      MITRE_MAP[t_type],
            "severity":      _initial_severity(t_type),
        })
    return threats


def _fresh_state(task_config: dict) -> dict:
    return {
        "threats": _make_threats_fixed(task_config),
        "scanned_nodes": set(),
        "system_health": 100,
        "score": 0.0,
        "step": 0,
        "done": False,
    }


def _do_reset_session(sess: Session) -> None:
    """Reset all mutable fields on an existing Session object in-place."""
    sess.state                  = _fresh_state(sess.task_config)
    sess.history                = []
    sess.episode_history        = []
    sess.episode_actions_taken  = []
    sess.episode_rewards        = []
    sess.threats_detected       = set()
    sess.threats_contained      = set()
    sess.false_positive_actions = 0
    sess.attack_plan            = {}


def _validate_session_state(sess: Session) -> None:
    s = sess.state
    try:
        assert isinstance(s["threats"], list)
        assert isinstance(s["scanned_nodes"], set)
        assert isinstance(s["system_health"], (int, float))
        assert isinstance(s["score"], (int, float))
        assert isinstance(s["step"], int)
        assert isinstance(s["done"], bool)
        if not math.isfinite(s["system_health"]):
            raise ValueError("system_health non-finite")
        if not math.isfinite(s["score"]):
            raise ValueError("score non-finite")
    except Exception as e:
        log.error(f"State corruption in session — auto-resetting: {e}")
        _do_reset_session(sess)


def _clamp_health(sess: Session) -> None:
    sess.state["system_health"] = int(
        max(0, min(100, round(sess.state["system_health"])))
    )


def _clamp_reward(r: float) -> float:
    if not math.isfinite(r):
        raise ValueError(f"non-finite reward passed to _clamp_reward: {r!r}")
    normalized_reward = (float(r) + 2.0) / 4.0
    return max(0.0, min(1.0, normalized_reward))


def _clamp_score(sess: Session) -> None:
    if not math.isfinite(sess.state["score"]):
        sess.state["score"] = 0.0
    sess.state["score"] = max(0.0, min(1.0, sess.state["score"]))


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


def _update_visibility(sess: Session) -> None:
    """Auto-reveal threats based on age or lateral movement, gated by task difficulty."""
    cfg          = sess.task_config
    age_thresh   = cfg.get("age_visibility_threshold", 5)
    detect_prob  = cfg.get("base_detection_prob", 1.0)
    fn_rate      = cfg.get("false_negative_rate", 0.0)
    for t in sess.state["threats"]:
        if t.get("contained") or t.get("visible"):
            continue
        if t["stage"] == "lateral_movement":
            if random.random() > fn_rate:
                t["visible"] = True
        elif t["age"] >= age_thresh:
            if random.random() < detect_prob and random.random() > fn_rate:
                t["visible"] = True


def _age_threats(sess: Session) -> None:
    prog_prob = sess.task_config.get("attack_progression_prob", 0.15)
    sev_growth = sess.task_config.get("natural_severity_growth", 0.05)
    for t in sess.state["threats"]:
        if not t.get("contained"):
            t["age"] += 1
            # Severity grows each step — makes severity-based action logic meaningful
            t["severity"] = round(min(1.0, t.get("severity", 0.5) + sev_growth), 3)
            if t["stage"] == "initial" and random.random() < prog_prob:
                # Stage escalates to lateral_movement but original type is PRESERVED.
                # The correct mitigation is always determined by original_type so the
                # agent is never penalized for correctly identifying the threat before escalation.
                t["stage"] = "lateral_movement"
                # Keep t["type"] = original_type (do NOT mutate it)
                if "original_type" not in t:
                    t["original_type"] = t["type"]
                t["escalated"] = True


def _visible_threats(sess: Session) -> list:
    out = []
    for t in sess.state["threats"]:
        if t["visible"] and not t.get("contained"):
            out.append({
                "type":          t["type"],
                "original_type": t.get("original_type", t["type"]),
                "escalated":     t.get("escalated", False),
                "node":          t["node"],
                "stage":         t["stage"],
                "age":           t["age"],
                "mitre_id":      t.get("mitre_id", MITRE_MAP.get(t.get("original_type", t["type"]), "T0000")),
            })
    return [enrich_threat(t) for t in out]


def _obs(sess: Session) -> dict:
    scanned = sess.state["scanned_nodes"]
    hidden_count = sum(
        1 for t in sess.state["threats"] if not t["visible"] and not t.get("contained")
    )
    score = round(sess.state["score"], 4)
    return {
        "visible_threats":  _visible_threats(sess),
        "hidden_node_count": hidden_count,
        "scan_coverage":    round(len(scanned) / TOTAL_NODES, 2),
        "system_health":    sess.state["system_health"],
        "score":            score,
        "normalized_score": score,   # running average ∈ [0,1] — proper learning signal
        "step":             sess.state["step"],
        "done":             sess.state["done"],
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
    score = obs.get("score", 0.0)
    resp = {
        "action":           action,
        "reward":           round(float(reward), 3),
        "visible_threats":  obs.get("visible_threats", []),
        "hidden_node_count": obs.get("hidden_node_count", TOTAL_NODES),
        "scan_coverage":    obs.get("scan_coverage", 0.0),
        "system_health":    obs.get("system_health", 100),
        "score":            score,
        "normalized_score": obs.get("normalized_score", score),
        "step":             obs.get("step", 0),
        "done":             obs.get("done", False),
        "reason":           reason,
        "confidence":       round(float(confidence), 2),
    }
    if error is not None:
        resp["error"] = error
    return resp


# ─── EXCEPTION HANDLERS ───────────────────────────────────────────────────────
_EMPTY_OBS: dict = {
    "visible_threats": [], "hidden_node_count": TOTAL_NODES,
    "scan_coverage": 0.0, "system_health": 100,
    "score": 0.0, "normalized_score": 0.0, "step": 0, "done": False,
}


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if DEBUG:
        log.debug(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=200,
        content=safe_response(_EMPTY_OBS, action="", reward=0.0,
                               reason="Invalid input received. Action rejected.",
                               confidence=0.0, error="invalid action"),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=safe_response(_EMPTY_OBS, action="", reward=0.0,
                               reason="Internal error. State preserved.",
                               confidence=0.0, error="internal error"),
    )


# ─── INPUT MODELS ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task:       str = "easy"
    seed:       int = 0
    session_id: str | None = None   # optional; omit for auto-generated UUID


class StepRequest(BaseModel):
    """Extended step request that adds optional session_id."""
    action:     str
    session_id: str | None = None   # omit to use the most-recently-reset session

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v):
        if not isinstance(v, str):
            v = str(v)
        if len(v) > MAX_ACTION_LEN:
            v = v[:MAX_ACTION_LEN]
        return v


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Adaptive Cyber Defense API v2.0 — OpenEnv compatible"}


@app.get("/tasks")
def get_tasks():
    return TASKS


@app.get("/history")
def get_history(session_id: str | None = None):
    sess = _get_session(session_id)
    if sess is None:
        return {"error": "session_id required. Call /reset first.", "episode_steps": [], "total_steps": 0}
    return {
        "episode_steps": sess.episode_history,
        "total_steps": len(sess.episode_history),
        "total_reward": round(sum(s["reward"] for s in sess.episode_history), 4),
        "final_status": "done" if sess.episode_history and sess.episode_history[-1]["done"] else "in_progress",
        "session_id": session_id,
    }


@app.get("/reset")
@app.get("/reset/")
@app.post("/reset")
@app.post("/reset/")
def reset(req: ResetRequest = None):
    task_name  = (req.task.lower().strip() if req and req.task else "easy")
    if task_name not in TASK_OVERRIDES:
        task_name = "easy"
    task_cfg   = TASK_OVERRIDES[task_name]
    seed       = int(req.seed) if req and req.seed is not None else 0

    # Determine session_id: use caller-provided or generate a new UUID
    sid = (req.session_id.strip() if req and req.session_id else None) or str(uuid.uuid4())

    # Evict old sessions if at capacity, then create or overwrite
    _evict_oldest_sessions()
    sess = Session(task_name=task_name, task_config=task_cfg)
    random.seed(seed)
    _do_reset_session(sess)
    sess.attack_plan = adaptive_attacker.on_episode_start()
    _SESSIONS[sid] = sess

    obs = _obs(sess)
    obs["task"]       = task_name
    obs["session_id"] = sid          # always returned so callers can track it
    # attack_plan kept internally for AdaptiveAttacker but never exposed to agents
    return obs


@app.get("/state", response_model=Observation)
@app.get("/state/", response_model=Observation)
@app.post("/state", response_model=Observation)
@app.post("/state/", response_model=Observation)
def get_state(session_id: str | None = None):
    sess = _get_session(session_id)
    if sess is None:
        return JSONResponse(status_code=200, content={
            "visible_threats": [], "hidden_node_count": TOTAL_NODES,
            "scan_coverage": 0.0, "system_health": 100,
            "score": 0.0, "normalized_score": 0.0, "step": 0, "done": False,
            "error": "session_id required. Call /reset first.",
        })
    try:
        return Observation(**_obs(sess))
    except Exception as e:
        log.warning(f"get_state() snapshot error (transient): {e}")
        return JSONResponse(status_code=200, content={
            "visible_threats": [], "hidden_node_count": 0,
            "scan_coverage": 0.0, "system_health": sess.state.get("system_health", 100),
            "score": 0.0, "step": sess.state.get("step", 0), "done": sess.state.get("done", False),
        })


@app.post("/step")
def step(req: StepRequest):
    # OpenEnv spec requires HTTP 200 always; errors go in the response body
    if not req.session_id:
        return JSONResponse(
            status_code=200,
            content={
                "action": "", "reward": 0.0, "reason": "session_id required. Call /reset first.",
                "confidence": 0.0, "done": False, "error": "session_id required",
                "score": 0.0, "normalized_score": 0.0, "step": 0,
                "visible_threats": [], "hidden_node_count": 5, "scan_coverage": 0.0, "system_health": 100,
            }
        )
    sid = req.session_id.strip()
    if sid not in _SESSIONS:
        return JSONResponse(
            status_code=200,
            content={
                "action": "", "reward": 0.0,
                "reason": f"Session '{sid}' not found. Call /reset to create a new session.",
                "confidence": 0.0, "done": False, "error": "session_not_found",
                "score": 0.0, "normalized_score": 0.0, "step": 0,
                "visible_threats": [], "hidden_node_count": 5, "scan_coverage": 0.0, "system_health": 100,
            }
        )
    sess = _SESSIONS[sid]

    try:
        _validate_session_state(sess)
        s = sess.state  # local alias for brevity

        if s["done"]:
            if DEBUG:
                log.debug("step() called after done=True")
            return safe_response(
                _obs(sess), action="", reward=0.0,
                reason="Episode is over. Call /reset to start a new episode.",
                confidence=1.0, error="Episode over — call /reset",
            )

        raw_action = req.action.strip().lower()

        # Unknown action — safe fallback
        if raw_action not in VALID_ACTIONS:
            if DEBUG:
                log.debug(f"Unknown action: {raw_action!r}")
            reward = _clamp_reward(-0.5)
            s["system_health"] = max(0, s["system_health"] - 5)
            _age_threats(sess)
            _update_visibility(sess)
            _clamp_health(sess)
            _all_r = sess.episode_rewards + [reward]
            s["score"] = round(sum(_all_r) / len(_all_r), 4)
            s["step"] += 1
            if s["system_health"] <= 0 or s["step"] >= sess.task_config.get("max_steps", 50):
                s["done"] = True
            obs = _obs(sess)
            sess.history.append({"step": s["step"], "action": raw_action,
                                  "reward": round(reward, 3), "attack": None})
            return safe_response(
                obs, action=raw_action, reward=reward,
                reason=f"'{raw_action}' is not a recognised action. Valid: block_ip, isolate_machine, patch, ignore, scan_node_1..5.",
                confidence=0.0, error="invalid action",
            )

        reason = ""
        confidence = 0.5
        matched = False
        early_bonus = False
        matched_threat_type = None
        scan_found_nothing = False

        # ── SCAN ──
        if raw_action.startswith("scan"):
            node = raw_action[len("scan_"):] if raw_action.startswith("scan_") else ""
            if node in NODES:
                s["scanned_nodes"].add(node)
                false_neg = sess.task_config.get("false_negative_rate", 0.0)
                revealed = False
                for t in s["threats"]:
                    if t["node"] == node and not t["visible"] and not t.get("contained"):
                        if random.random() > false_neg:
                            t["visible"] = True
                            revealed = True
                reason, confidence = _build_reason(raw_action, False, None, False)
                if revealed:
                    reason = f"Scan of {node} revealed a hidden threat. Partial observability lifted for this node."
                    confidence = 0.90
                else:
                    reason = f"Scan of {node} found no new threats. Coverage improved."
                    confidence = 0.75
                    scan_found_nothing = True
            else:
                reason = f"'{node}' is not a valid node. Valid nodes: node_1 through node_5."
                confidence = 0.10

        # ── DEFENSE ──
        else:
            for t in s["threats"]:
                if t["visible"] and not t.get("contained"):
                    # Use original_type for MITRE lookup — type is preserved even after
                    # stage escalation so agents are never penalized for correct identification.
                    correct = _get_correct_action(t.get("original_type", t["type"]), t.get("severity", 0.5), t.get("stage", "initial"))
                    if raw_action == correct:
                        t["contained"] = True
                        matched_threat_type = t["type"]
                        matched = True
                        early_bonus = t["age"] < 3
                        break

            if not matched:
                for t in s["threats"]:
                    if t["visible"] and not t.get("contained"):
                        matched_threat_type = t["type"]
                        break
                if raw_action == "ignore":
                    s["system_health"] = max(0, s["system_health"] - 10)
                else:
                    s["system_health"] = max(0, s["system_health"] - 5)

            reason, confidence = _build_reason(raw_action, matched, matched_threat_type, early_bonus)

        # Passive health degradation
        _degrade_rate = sess.task_config.get("health_degradation_rate", 0.0)
        if _degrade_rate > 0:
            _active = sum(1 for t in s["threats"] if not t.get("contained"))
            s["system_health"] = max(0, s["system_health"] - _degrade_rate * (_active / TOTAL_NODES) * 100)

        _age_threats(sess)
        _update_visibility(sess)
        _clamp_health(sess)

        # Reward authority: MITRE-aligned lookup table.
        # Normalized via _clamp_reward((r + 2.0) / 4.0) → [0.0, 1.0].
        # correct:      raw 1.0 (+0.1 early bonus if age<3) → 0.750 (0.775)
        # wrong:        raw -0.5                            → 0.375
        # ignore:       raw -1.5                            → 0.125
        # scan reveal:  raw 0.02                            → 0.505
        # scan empty:   raw -0.3                            → 0.425
        if raw_action.startswith("scan"):
            _sn = raw_action[len("scan_"):] if raw_action.startswith("scan_") else ""
            reward = _clamp_reward(0.02) if not scan_found_nothing else _clamp_reward(-0.3)
        elif raw_action == "ignore":
            reward = _clamp_reward(-1.5)
        elif raw_action in ("block_ip", "isolate_machine", "patch"):
            if matched:
                reward = _clamp_reward(1.1 if early_bonus else 1.0)
            else:
                reward = _clamp_reward(-0.5)
        else:
            reward = _clamp_reward(-0.5)

        # Running average score — never saturates (each reward ∈ [0,1], mean ∈ [0,1])
        _all_rewards = sess.episode_rewards + [reward]
        s["score"] = round(sum(_all_rewards) / len(_all_rewards), 4)
        s["step"] += 1

        if s["system_health"] <= 0 or s["step"] >= sess.task_config.get("max_steps", 50):
            s["done"] = True

        obs = _obs(sess)

        sess.history.append({"step": s["step"], "action": raw_action,
                              "reward": round(reward, 3), "attack": matched_threat_type})
        sess.episode_history.append({"step": s["step"], "action": raw_action,
                                     "reward": float(reward), "done": bool(s["done"]),
                                     "reason": reason})

        # Analytics tracking
        sess.episode_actions_taken.append(raw_action)
        sess.episode_rewards.append(reward)
        _MITIGATIONS = {"block_ip", "isolate_machine", "patch"}
        if raw_action in _MITIGATIONS and not matched:
            sess.false_positive_actions += 1
        for threat in obs.get("visible_threats", []):
            tid = threat.get("id", "")
            if tid:
                sess.threats_detected.add(tid)
        for t in s["threats"]:
            if t.get("contained"):
                tid = str(t.get("id", f"{t['type']}_{t['node']}"))
                sess.threats_contained.add(tid)

        # Red team
        translated = translate_action(raw_action)
        threat_ctx = (matched_threat_type or "UNKNOWN").upper()
        adaptive_attacker.observe_defender_action(translated, threat_ctx)

        if s["done"]:
            contained = sum(1 for t in s["threats"] if t.get("contained"))
            total     = max(1, len(s["threats"]))
            adaptive_attacker.on_episode_end(
                defender_won=(contained / total) >= 0.8,
                score=round(s["score"], 4),
            )

        return safe_response(obs, action=raw_action, reward=reward,
                             reason=reason, confidence=confidence)

    except Exception as e:
        log.error(f"/step unhandled exception: {e}", exc_info=True)
        # sess is always defined here because session validation happens before try block
        try:
            _validate_session_state(sess)
            obs = _obs(sess)
        except Exception:
            _do_reset_session(sess)
            obs = _obs(sess)
        # Reward errors indicate engine failure — return explicit error response
        error_msg = f"reward_error: {str(e)[:64]}" if "Reward" in str(type(e).__name__) else "internal_error"
        return safe_response(obs, action="", reward=0.0,
                             reason=f"Step failed: {error_msg}. State preserved.",
                             confidence=0.0, error=error_msg)


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
def threat_intel(session_id: str | None = None):
    """Returns MITRE ATT&CK enriched threat intelligence about active threats."""
    try:
        sess = _get_session(session_id)
        if sess is None:
            return {"error": "session_id required", "active_campaigns": [], "risk_level": "UNKNOWN",
                    "threat_summary": {}, "network_assessment": {}, "recommended_actions": []}
        _validate_session_state(sess)
        visible_threats = _visible_threats(sess)
        scan_coverage   = round(len(sess.state["scanned_nodes"]) / TOTAL_NODES, 3)
        system_health   = sess.state["system_health"]

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
            t["node"] for t in sess.state["threats"]
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
            "timestamp": sess.state["step"],
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
                "unscanned_nodes": max(0, TOTAL_NODES - len(sess.state["scanned_nodes"])),
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
def get_analytics(session_id: str | None = None):
    """Returns real SOC metrics for the current episode."""
    try:
        sess = _get_session(session_id)
        if sess is None:
            return {"error": "session_id required", "soc_metrics": {}, "resources_remaining": 0.0}
        _validate_session_state(sess)
        visible_threats = _visible_threats(sess)
        total_steps   = sess.state["step"]
        system_health = sess.state["system_health"]
        scan_coverage = round(len(sess.state["scanned_nodes"]) / TOTAL_NODES, 3)

        n_detected  = len(sess.threats_detected)
        n_contained = len(sess.threats_contained)

        # Avoid division by zero; return 0.0 rates when nothing detected yet
        mttd = round(total_steps / max(1, n_detected), 2)
        mttr = round(total_steps / max(1, n_contained), 2)

        # Detection rate: detected / (detected + remaining hidden)
        hidden_count = TOTAL_NODES - len(sess.state["scanned_nodes"])
        detection_rate = round(
            n_detected / max(1, n_detected + hidden_count), 3
        )

        containment_rate = round(n_contained / max(1, n_detected), 3) if n_detected > 0 else 0.0

        total_mitigations = sum(
            1 for a in sess.episode_actions_taken
            if a in {"block_ip", "isolate_machine", "patch"}
        )
        false_positive_rate = round(
            sess.false_positive_actions / max(1, total_mitigations), 3
        )

        avg_reward = round(
            sum(sess.episode_rewards) / max(1, len(sess.episode_rewards)), 4
        )

        if len(sess.episode_rewards) >= 10:
            first_5 = sum(sess.episode_rewards[:5]) / 5
            last_5  = sum(sess.episode_rewards[-5:]) / 5
            trend = "IMPROVING" if last_5 > first_5 else "DECLINING"
        else:
            trend = "INSUFFICIENT_DATA"

        action_counts: dict = {}
        for a in sess.episode_actions_taken:
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
            scanned = sess.state["scanned_nodes"]
            unscanned = [f"scan_node_{i}" for i in range(1, TOTAL_NODES + 1)
                         if f"node_{i}" not in scanned]
            recommended = unscanned[0] if unscanned else "ignore"
        else:
            recommended = "ignore"

        compromised = list({
            t["node"] for t in sess.state["threats"]
            if t.get("stage") == "lateral_movement" and not t.get("contained")
        })

        # Resources remaining: ratio of budget not consumed by action costs
        task_budget = sess.task_config.get("resource_per_step", 1.0)
        action_cost = sum(
            0.4 if a == "isolate_machine" else
            0.3 if a in ("block_ip", "patch") else
            0.2 if a.startswith("scan") else 0.0
            for a in sess.episode_actions_taken
        )
        total_budget = task_budget * max(1, total_steps)
        resources_remaining = round(max(0.0, 1.0 - action_cost / max(total_budget, 0.01)), 3)

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
                "threats_detected":      n_detected,
                "threats_contained":     n_contained,
                "threats_active":        len(visible_threats),
                "threats_ids_detected":  list(sess.threats_detected),
            },
            "network_status": {
                "system_health":     system_health,
                "scan_coverage":     scan_coverage,
                "compromised_nodes": compromised,
                "nodes_at_risk":     len(compromised),
            },
            "agent_behavior": {
                "total_actions":    len(sess.episode_actions_taken),
                "action_breakdown": action_counts,
                "most_used_action": most_used,
                "false_positives":  sess.false_positive_actions,
            },
            "recommended_next_action": recommended,
            "resources_remaining":     resources_remaining,
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
    # Port 7860 matches openenv.yaml docker.port and Dockerfile EXPOSE
    port = int(_os.getenv("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
