import requests

BASE_URL = "https://shanmukhgara-adaptive-cyber-defense.hf.space"

SYSTEM_PROMPT = (
    "You are a cyber defense agent. Respond with the best action for the given threat. "
    "Some threats may be hidden. Use SCAN actions (e.g., scan_node_1) to discover "
    "threats before they escalate."
)

ATTACK_TO_ACTION = {
    "phishing": "block_ip",
    "malware": "isolate_machine",
    "ddos": "patch",
}


def choose_action(obs):
    visible = obs.get("visible_threats", [])
    scan_coverage = obs.get("scan_coverage", 1.0)

    # If coverage is low, scan an unscanned node
    if scan_coverage < 1.0:
        node_num = int(scan_coverage * 5) + 1
        if node_num <= 5:
            return f"scan_node_{node_num}"

    # Respond to first visible threat
    if visible:
        attack_type = visible[0].get("type", "")
        return ATTACK_TO_ACTION.get(attack_type, "ignore")

    return "ignore"


def run():
    requests.get(f"{BASE_URL}/reset")

    total_reward = 0

    for _ in range(5):
        state_res = requests.get(f"{BASE_URL}/state")
        obs = state_res.json()
        action = choose_action(obs)

        res = requests.post(f"{BASE_URL}/step", json={"action": action})
        data = res.json()
        total_reward += data.get("reward", 0)

    print("Total reward:", round(total_reward, 3))


if __name__ == "__main__":
    run()
