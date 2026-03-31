---
title: Adaptive Cyber Defense Environment
emoji: 🛡️
colorFrom: red
colorTo: purple
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

The environment simulates a 5-node corporate network under continuous attack. Three threat types (phishing, malware, DDoS) spawn on random nodes and age over time — escalating from `initial` to `lateral_movement` if not contained. The agent must scan to reveal hidden threats, then apply the correct mitigation before system health reaches zero.

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
  "reward":            1.0,
  "visible_threats":   [],
  "hidden_node_count": 3,
  "scan_coverage":     0.4,
  "system_health":     85,
  "score":             1.5,
  "step":              8,
  "done":              false
}
```

---

## Action Space

| Action | Effect |
|--------|--------|
| `block_ip` | Neutralises phishing threats |
| `isolate_machine` | Neutralises malware threats |
| `patch` | Neutralises DDoS threats |
| `ignore` | −10 health, −1.0 reward |
| `scan_node_1` … `scan_node_5` | Reveals hidden threats on that node |

---

## MITRE ATT&CK Mapping

| Threat Type | Technique | Tactic |
|-------------|-----------|--------|
| phishing | T1566 | Initial Access |
| malware | T1204 | Execution |
| ddos | T1498 | Impact |

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
