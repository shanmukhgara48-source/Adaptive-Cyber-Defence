---
title: Adaptive Cyber Defense Environment
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - cybersecurity
---

# Adaptive Cyber Defense Environment (OpenEnv)

An autonomous SOC simulation environment built on **FastAPI** and packaged as an **OpenEnv**-compatible API. An AI agent defends a corporate network against multi-stage cyber attacks under **partial observability** — threats are hidden until the agent actively scans nodes to discover them.

---

## Overview

The environment simulates a 5-node corporate network under continuous attack. Five threat types (phishing, malware, ransomware, DDoS, lateral_movement) spawn on random nodes and age over time — escalating from `initial` to `lateral_movement` if not contained. The agent must scan to reveal hidden threats, then apply the correct MITRE-aligned mitigation before system health reaches zero.

---

## Features

- **FastAPI backend** — lightweight, production-grade REST API
- **Partial observability** — threats are hidden; `scan_node_X` actions reveal them
- **SCAN-based discovery** — 5 scannable nodes, coverage tracked per episode
- **Deterministic reward system** — clamped to `[-2.0, 2.0]`, no NaN/Inf
- **Threat lifecycle** — threats age, escalate, and cause damage if ignored
- **MITRE ATT&CK mapping** — each threat type maps to a real ATT&CK technique
- **Robust error handling** — never crashes; always returns complete JSON
- **Stress-tested** — 400+ adversarial test cases across 8 categories
- **Auto-healing state** — detects and recovers from state corruption automatically

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET/POST` | `/reset` | Start a new episode |
| `GET/POST` | `/state` | Get current observation |
| `POST` | `/step` | Submit an action |

### Observation Schema

```json
{
  "visible_threats":   [{"type": "malware", "node": "node_2", "stage": "initial", "age": 2}],
  "hidden_node_count": 3,
  "scan_coverage":     0.4,
  "system_health":     85,
  "score":             0.52,
  "step":              7,
  "done":              false
}
```

### Step Response Schema

```json
{
  "action":            "block_ip",
  "reward":            0.75,
  "visible_threats":   [],
  "hidden_node_count": 3,
  "scan_coverage":     0.4,
  "system_health":     85,
  "score":             0.62,
  "step":              8,
  "done":              false
}
```

Reward is normalised to `[0.0, 1.0]` via `(raw + 2.0) / 4.0`. Correct mitigation gives `raw=1.0 → reward=0.750`. Score is the running mean reward for the episode, also in `[0.0, 1.0]`.

---

## Action Space

| Action | Effect |
|--------|--------|
| `block_ip` | Neutralises phishing (T1566) and lateral_movement (T1021) threats |
| `isolate_machine` | Neutralises malware (T1204) and ransomware (T1486) threats |
| `patch` | Neutralises DDoS (T1499) threats |
| `ignore` | −10 health, reward penalty |
| `scan_node_1` … `scan_node_5` | Reveals hidden threats on that node |

---

## MITRE ATT&CK Mapping

| Threat Type | Technique | Tactic | Correct Action |
|-------------|-----------|--------|----------------|
| phishing | T1566 | Initial Access | block_ip |
| malware | T1204 | Execution | isolate_machine |
| ransomware | T1486 | Impact | isolate_machine |
| ddos | T1499 | Impact | patch |
| lateral_movement | T1021 | Lateral Movement | block_ip |

---

## Threat lifecycle

Each threat follows a kill chain progression if not contained:

```
Initial Access → Execution → Lateral Movement → Exfiltration
(T1566)         (T1204)      (T1021)            (critical damage)
↓                ↓            ↓
block_ip    isolate_machine   block_ip
```

### Stage descriptions

| Stage | Description | Agent must act before... |
|-------|-------------|--------------------------|
| `initial` | Threat spawned, may be hidden | Age reaches 5 |
| `reconnaissance` | Threat scanning network | Lateral movement triggers |
| `lateral_movement` | Spreading to adjacent nodes | Exfiltration begins |
| `exfiltration` | Critical asset data being stolen | Health reaches 0 |
| `contained` | Threat neutralised by agent | — |

### Visibility rules
- Threats are **hidden** by default — agent cannot see them
- A threat becomes **visible** when:
  - Agent scans the node it is on (`scan_node_X`)
  - Threat age reaches 5 steps (natural escalation)
  - Threat reaches `lateral_movement` stage
- Hidden threats still cause damage every step

### Early containment bonus
Containing a threat before age 3 gives a `+0.1` speed bonus
on top of the base reward — rewarding proactive defense.

---

## Example Usage

```bash
# Start a new episode
curl -X POST http://localhost:8000/reset

# Scan a node to reveal hidden threats
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "scan_node_1"}'

# Apply mitigation after threat is revealed
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": "block_ip"}'

# Get current state
curl http://localhost:8000/state
```

---

## Deployment

### Local

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t cyber-defense .
docker run -p 7860:7860 cyber-defense
```

### Hugging Face Spaces

Deploy directly — the `Dockerfile` exposes port `7860` as required by Spaces.

---

## Evaluation Readiness

- Passes OpenEnv validator (`openenv.yaml` included)
- Handles all adversarial inputs without crashing (SQL injection, XSS, unicode, null bytes, oversized payloads)
- Always returns complete observation structure on every call
- Stable under concurrent load and 200+ rapid sequential requests

---

## Tech Stack

- **Python 3.10+**
- **FastAPI** + **Pydantic v2**
- **Uvicorn**
- **Docker**
- **OpenEnv**

---

## Baseline Scores

Baseline scores from `inference.py` (deterministic MITRE-lookup agent, no LLM).
Grader: `0.50×containment + 0.20×critical_health + 0.15×avg_resource_left + 0.15×avg_reward`.

| Task       | Max Steps | Threshold | Notes                                         |
|------------|-----------|-----------|-----------------------------------------------|
| easy       | 30        | 0.50      | 3 threats, high detection, generous resources |
| medium     | 50        | 0.60      | 2 threats, FP noise, limited budget           |
| hard       | 30        | 0.45      | 5 threats, APT evasion, scarce resources      |
| nightmare  | 15        | 0.25      | 5 threats, near-zero detection                |
| elite      | 15        | 0.20      | All nodes pre-compromised, insider threat     |
| impossible | 10        | 0.10      | AI attacker, no ceiling                       |

To reproduce:

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export HF_TOKEN=<your_token>
export BASE_URL=http://localhost:8000
python inference.py
```

---

## Grading

Each task is scored on a scale of 0.0–1.0 using this formula:

| Component | Weight | Description |
|-----------|--------|-------------|
| containment_rate | 50% | Fraction of threats contained by end of episode |
| critical_health | 20% | Average health of critical assets (criticality >= 0.7) |
| resource_efficiency | 15% | Average fraction of resources unused per step |
| avg_step_reward | 15% | Average per-step reward quality |

### Worked example (easy task)
- containment_rate = 0.80 (4 of 5 threats contained)
- critical_health  = 0.90 (assets mostly healthy)
- resource_efficiency = 0.70 (good resource use)
- avg_step_reward = 0.50 (mostly correct actions)
- episode_score = 0.50×0.80 + 0.20×0.90 + 0.15×0.70 + 0.15×0.50
- episode_score = 0.40 + 0.18 + 0.105 + 0.075 = 0.76

Passing thresholds (difficulty ladder — higher bar requires better performance):
- easy:       0.50
- medium:     0.60  (higher bar than easy — requires better containment)
- hard:       0.45  (lower bar — genuinely harder to achieve)
- nightmare:  0.25
- elite:      0.20
- impossible: 0.10
