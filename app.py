from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

ATTACKS = ["phishing", "malware", "ddos"]
CORRECT_ACTION = {
    "phishing": "block_ip",
    "malware":  "isolate_machine",
    "ddos":     "patch",
}

state = {
    "current_attack": random.choice(ATTACKS),
    "system_health":  100,
    "score":          0,
    "step":           0,
    "done":           False,
}


def _new_attack():
    return random.choice(ATTACKS)


class StepRequest(BaseModel):
    action: str


@app.get("/")
def root():
    return {"message": "Cyber Defense API running"}


@app.get("/reset")
def reset():
    state["current_attack"] = _new_attack()
    state["system_health"]  = 100
    state["score"]          = 0
    state["step"]           = 0
    state["done"]           = False
    return {
        "current_attack": state["current_attack"],
        "system_health":  state["system_health"],
        "score":          state["score"],
        "step":           state["step"],
        "done":           state["done"],
    }


@app.get("/state")
def get_state():
    return {
        "current_attack": state["current_attack"],
        "system_health":  state["system_health"],
        "score":          state["score"],
        "step":           state["step"],
        "done":           state["done"],
    }


@app.post("/step")
def step(req: StepRequest):
    if state["done"]:
        return {"error": "Episode over — call /reset to start a new one",
                **get_state()}

    action  = req.action.strip().lower()
    attack  = state["current_attack"]
    correct = CORRECT_ACTION[attack]

    if action == correct:
        reward = 1.0
    elif action == "ignore":
        reward = -1.0
        state["system_health"] = max(0, state["system_health"] - 10)
    else:
        reward = -0.5
        state["system_health"] = max(0, state["system_health"] - 5)

    state["score"] += reward
    state["step"]  += 1

    if state["system_health"] <= 0 or state["step"] >= 50:
        state["done"] = True
    else:
        state["current_attack"] = _new_attack()

    return {
        "action":         action,
        "correct_action": correct,
        "reward":         reward,
        "current_attack": state["current_attack"],
        "system_health":  state["system_health"],
        "score":          round(state["score"], 2),
        "step":           state["step"],
        "done":           state["done"],
    }
