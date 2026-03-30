import requests

BASE_URL = "https://shanmukhgara-adaptive-cyber-defense.hf.space"

def run():
    requests.get(f"{BASE_URL}/reset")

    total_reward = 0

    for _ in range(5):
        res = requests.post(f"{BASE_URL}/step", json={"action": "block_ip"})
        data = res.json()
        total_reward += data.get("reward", 0)

    print("Total reward:", total_reward)

if __name__ == "__main__":
    run()
