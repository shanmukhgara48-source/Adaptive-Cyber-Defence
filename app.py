from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

ATTACKS = ["phishing", "malware", "ddos"]
CORRECT_ACTION = {
    "phishing": "block_ip",
    "malware": "isolate_machine",
    "ddos": "patch",
}
TOTAL_NODES = 5
NODES = [f"node_{i}" for i in range(1, TOTAL_NODES + 1)]


def _make_threats():
    return [
        {
            "type": random.choice(ATTACKS),
            "node": random.choice(NODES),
            "visible": False,
            "age": 0,
            "stage": "initial",
        }
        for _ in range(3)
    ]


state = {
    "threats": _make_threats(),
    "scanned_nodes": set(),
    "system_health": 100,
    "score": 0,
    "step": 0,
    "done": False,
}


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


class StepRequest(BaseModel):
    action: str


@app.get("/")
def root():
    return {"message": "Cyber Defense API running"}


@app.get("/reset")
@app.get("/reset/")
@app.post("/reset")
@app.post("/reset/")
def reset():
    state["threats"] = _make_threats()
    state["scanned_nodes"] = set()
    state["system_health"] = 100
    state["score"] = 0
    state["step"] = 0
    state["done"] = False
    return _obs()


@app.get("/state")
@app.get("/state/")
@app.post("/state")
@app.post("/state/")
def get_state():
    return _obs()


@app.post("/step")
def step(req: StepRequest):
    if state["done"]:
        return {"error": "Episode over — call /reset", **_obs()}

    action = req.action.strip().lower()
    reward = 0.0

    if action.startswith("scan"):
        # Extract node id: "scan_node_1" → "node_1", or "scan node_1" → "node_1"
        parts = action.replace("scan_", "scan ").split()
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
        # Defense action against any visible matching threat
        matched = False
        for t in state["threats"]:
            if t["visible"] and not t.get("contained"):
                correct = CORRECT_ACTION.get(t["type"], "")
                if action == correct:
                    t["contained"] = True
                    reward += 1.0
                    matched = True
                    break
        if not matched:
            if action == "ignore":
                reward = -1.0
                state["system_health"] = max(0, state["system_health"] - 10)
            else:
                reward = -0.5
                state["system_health"] = max(0, state["system_health"] - 5)

    _age_threats()
    _update_visibility()

    state["score"] += reward
    state["step"] += 1

    if state["system_health"] <= 0 or state["step"] >= 50:
        state["done"] = True

    obs = _obs()
    return {"action": action, "reward": round(reward, 3), **obs}
