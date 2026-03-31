import random
import logging
import math
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, field_validator

# ─── DEBUG ────────────────────────────────────────────────────────────────────
DEBUG = True
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.WARNING)
log = logging.getLogger("cyber_defense")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
ATTACKS = ["phishing", "malware", "ddos"]
CORRECT_ACTION = {
    "phishing": "block_ip",
    "malware": "isolate_machine",
    "ddos": "patch",
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

# ─── APP ──────────────────────────────────────────────────────────────────────
app = FastAPI()


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
        }
        for _ in range(3)
    ]


def _fresh_state():
    return {
        "threats": _make_threats(),
        "scanned_nodes": set(),
        "system_health": 100,
        "score": 0.0,
        "step": 0,
        "done": False,
    }


state = _fresh_state()


def _reset_state():
    fresh = _fresh_state()
    state.update(fresh)


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
        return 0.0
    return max(MIN_REWARD, min(MAX_REWARD, r))


def _clamp_score():
    if not math.isfinite(state["score"]):
        state["score"] = 0.0


# ─── LOGIC ────────────────────────────────────────────────────────────────────
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
    return [
        {k: v for k, v in t.items() if k != "contained"}
        for t in state["threats"]
        if t["visible"] and not t.get("contained")
    ]


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


def safe_response(obs, action, reward=0.0, error=None):
    """Always returns a complete response with all required keys."""
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
        content=safe_response(obs, action="", reward=-0.5, error="invalid action"),
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
        content=safe_response(obs, action="", reward=-0.5, error="internal error"),
    )


# ─── INPUT MODEL ──────────────────────────────────────────────────────────────
class StepRequest(BaseModel):
    action: str

    @field_validator("action", mode="before")
    @classmethod
    def coerce_action(cls, v):
        # Coerce to string rather than reject — truncate if too long
        if not isinstance(v, str):
            if DEBUG:
                log.debug(f"Non-string action coerced: {type(v).__name__}={v!r}")
            v = str(v)
        if len(v) > MAX_ACTION_LEN:
            if DEBUG:
                log.debug(f"Oversized action truncated: len={len(v)}")
            v = v[:MAX_ACTION_LEN]
        return v


# ─── ENDPOINTS ────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Cyber Defense API running"}


@app.get("/reset")
@app.get("/reset/")
@app.post("/reset")
@app.post("/reset/")
def reset():
    _reset_state()
    return _obs()


@app.get("/state")
@app.get("/state/")
@app.post("/state")
@app.post("/state/")
def get_state():
    _validate_state()
    return _obs()


@app.post("/step")
def step(req: StepRequest):
    try:
        _validate_state()

        if state["done"]:
            if DEBUG:
                log.debug("step() called after done=True")
            return safe_response(_obs(), action="", reward=0.0, error="Episode over — call /reset")

        raw_action = req.action.strip().lower()

        # Whitelist enforcement — unknown actions get safe penalty, full obs returned
        if raw_action not in VALID_ACTIONS:
            if DEBUG:
                log.debug(f"Invalid action rejected by whitelist: {raw_action!r}")
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
            return safe_response(_obs(), action=raw_action, reward=reward, error="invalid action")

        reward = 0.0

        if raw_action.startswith("scan"):
            parts = raw_action.replace("scan_", "scan ").split()
            node = parts[1] if len(parts) > 1 else ""
            if node in NODES:
                state["scanned_nodes"].add(node)
                revealed = False
                for t in state["threats"]:
                    if t["node"] == node and not t["visible"] and not t.get("contained"):
                        t["visible"] = True
                        revealed = True
                reward = 0.02 if revealed else -0.01
            else:
                reward = -0.01
        else:
            matched = False
            for t in state["threats"]:
                if t["visible"] and not t.get("contained"):
                    correct = CORRECT_ACTION.get(t["type"], "")
                    if raw_action == correct:
                        t["contained"] = True
                        reward += 1.0
                        matched = True
                        break
            if not matched:
                if raw_action == "ignore":
                    reward = -1.0
                    state["system_health"] = max(0, state["system_health"] - 10)
                else:
                    reward = -0.5
                    state["system_health"] = max(0, state["system_health"] - 5)

        reward = _clamp_reward(reward)
        _age_threats()
        _update_visibility()
        _clamp_health()

        state["score"] += reward
        _clamp_score()
        state["step"] += 1

        if state["system_health"] <= 0 or state["step"] >= 50:
            state["done"] = True

        return safe_response(_obs(), action=raw_action, reward=reward)

    except Exception as e:
        log.error(f"/step unhandled exception: {e}", exc_info=True)
        try:
            _validate_state()
            obs = _obs()
        except Exception:
            _reset_state()
            obs = _obs()
        return safe_response(obs, action="", reward=-0.5, error="invalid action")


def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
