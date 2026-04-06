---
title: Adaptive Cyber Defense Simulator
emoji: 🛡️
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - cybersecurity
  - mitre-attack
---

# 🛡️ Adaptive Cyber Defense Simulator
### Autonomous SOC Assistant — OpenEnv Environment

> An enterprise-grade reinforcement learning environment where AI agents defend a corporate network against evolving, intelligent cyber attacks.  
> Built for the **Meta × Hugging Face OpenEnv Hackathon**.

[![HF Space](https://img.shields.io/badge/🤗_HuggingFace-Space-yellow)](https://huggingface.co/spaces/shanmukhgara/adaptive-cyber-defense)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-775_passing-brightgreen)]()
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)]()

---

## 🎯 What Is This?

A fully autonomous Security Operations Center (SOC) simulator where AI agents must **detect, classify, and respond to live cyber attacks** — without being told what the attacks are.

The agent sees only behavioral indicators (network traffic spikes, authentication failures, data egress anomalies) and must reason about the attack class to choose the correct mitigation. No ground-truth labels. No cheat sheet. Just signals.

**The challenge:** An adaptive red team tracks how the defender responds and shifts its attack strategy each episode to exploit weaknesses. Agents that learn rigid rules get outmaneuvered.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│              Corporate Network (5 nodes)             │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  node_1  │   │  node_3  │   │     node_5       │ │
│  │  (DMZ)   │◄─►│  (hub)   │◄─►│  (DB server)     │ │
│  └──────────┘   └──────────┘   └──────────────────┘ │
│                      ▲                               │
│               ┌──────┴──────┐                        │
│          ┌────┴────┐   ┌────┴────┐                   │
│          │ node_2  │   │ node_4  │                   │
│          │ (WS)    │   │ (SRV)   │                   │
│          └─────────┘   └─────────┘                   │
└─────────────────────────────────────────────────────┘
                    ▲
                    │ Multi-stage attacks
         ┌──────────────────────┐
         │   Adaptive Red Team  │
         │  APT → RANSOMWARE    │
         │  INSIDER → SUPPLY    │
         │  CHAIN → ZERO_DAY    │
         └──────────────────────┘
```

**Partial observability:** Threats are hidden by default. The agent must spend `scan_node_X` actions to reveal them — or wait until they escalate and become visible on their own (by which point damage has already begun).

---

## ⚡ Quick Start

```bash
# Clone
git clone https://github.com/shanmukhgara48-source/Adaptive-Cyber-Defence.git
cd Adaptive-Cyber-Defence

# Install
pip install -r requirements.txt

# Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000

# Run simulation with baseline agent
python run.py --task hard --episodes 3

# Launch interactive dashboard
streamlit run ui.py

# Run LLM inference agent
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
export HF_TOKEN=your_token
python inference.py
```

---

## 🧠 How It Works

### Observation Space

The agent **never sees the threat type directly**. Instead it receives raw behavioral IOC signals:

| Signal | High Value Suggests |
|--------|-------------------|
| `packets_per_second` | Volumetric flood (DDoS) |
| `outbound_data_bytes` | Data exfiltration |
| `lateral_connection_count` | Lateral movement |
| `failed_auth_attempts` | Credential attack / phishing |
| `unusual_process_count` | Malware execution |
| `spread_rate` | Ransomware propagation |

Signals are **noisy** — false positives, cross-signal contamination, and attacker evasion mean the agent must reason from patterns, not memorized thresholds.

### Action Space

| Action | Effect |
|--------|--------|
| `isolate_machine` | Cuts a node from the network (contains malware, ransomware) |
| `block_ip` | Blocks source IP (contains phishing, lateral movement) |
| `patch` | Applies security patch (contains DDoS) |
| `scan_node_1..5` | Reveals hidden threats on a specific node |
| `ignore` | No action — heavy health penalty |

### Reward Function

```
episode_score = 0.50 × containment_rate        # threats neutralised / total
              + 0.20 × critical_health          # critical asset health at end
              + 0.15 × resource_efficiency      # budget utilisation
              + 0.15 × speed_bonus              # early containment (age < 3 = 1.0)

score ∈ [0.0, 1.0]
```

---

## 🎯 Task Difficulty Tiers

| Task | Stars | Passing Score | Max Steps | Description |
|------|-------|---------------|-----------|-------------|
| `easy` | ⭐ | 0.40 | 30 | 3 threats, high detection probability |
| `medium` | ⭐⭐ | 0.55 | 50 | 4 threats, false positives, resource pressure |
| `hard` | ⭐⭐⭐ | 0.70 | 30 | 5 threats, APT evasion, scarce resources |
| `nightmare` | ⭐⭐⭐⭐ | 0.80 | 15 | Nation-state attacker, near-zero detection |
| `elite` | ⭐⭐⭐⭐⭐ | 0.88 | 15 | All nodes pre-compromised, insider threat |
| `impossible` | 💀 | — | 10 | Ceiling benchmark — no passing threshold |

`impossible` exists to measure frontier model capability — no agent is expected to "pass" it.

---

## 🤖 Agents

### Baseline Heuristic Agent
MITRE-lookup rule agent. Maps IOC signals to threat class and selects the correct mitigation deterministically. Scores ~0.84 on hard. Used as the performance floor.

### Q-Learning Agent
Trained for 500 episodes using tabular Q-learning. Learns optimal action selection from the reward signal without explicit rules. Generalises across threat configurations.

### LLM Agent (`inference.py`)
Chain-of-thought reasoning over IOC signals. Receives behavioral indicators, infers attack class, outputs action with natural language justification. Compatible with any OpenAI-API-compatible endpoint.

### Multi-Agent Arena
Two agents compete simultaneously against the same attack scenario. Side-by-side score comparison to benchmark new agents.

---

## 🔴 Adaptive Red Team

The attacker observes defender behavior across episodes and shifts strategy to counter it:

| Defender Pattern | Attacker Response |
|-----------------|-------------------|
| ISOLATE-heavy | Switches to Insider Threat (no external C2) |
| BLOCK_IP-heavy | Switches to Supply Chain (trusted process) |
| SCAN-heavy | Switches to APT (low-and-slow, stays hidden) |
| PATCH-heavy | Switches to Zero-Day (unpatched vector) |
| Balanced defense | Escalates to Ransomware (time pressure) |

---

## 🗺️ MITRE ATT&CK Mapping

| Threat Type | Technique | ID | Correct Mitigation |
|------------|-----------|-----|-------------------|
| Phishing | Spear Phishing | T1566 | `block_ip` |
| Malware | User Execution | T1204 | `isolate_machine` |
| Ransomware | Data Encrypted for Impact | T1486 | `isolate_machine` |
| DDoS | Endpoint Denial of Service | T1499 | `patch` |
| Lateral Movement | Remote Services | T1021 | `block_ip` |

Threats preserve their `original_type` through stage escalation — agents are never penalized for correct identification of an escalated threat.

---

## 📊 Benchmark Results

Scores from `inference.py` deterministic MITRE-lookup baseline:

| Agent | Easy | Medium | Hard | Nightmare |
|-------|------|--------|------|-----------|
| Baseline (MITRE lookup) | 0.82 | 0.76 | 0.84 | 0.71 |
| Random | ~0.21 | ~0.18 | ~0.15 | ~0.12 |
| Ignore-all | 0.10 | 0.08 | 0.06 | 0.04 |

---

## 🏃 Running Tests

```bash
# Full test suite (775 tests)
python -m pytest tests/ -v

# OpenEnv compliance check
python verify_openenv_compliance.py

# Adversarial robustness tests
python -m pytest tests/test_adversarial.py -v

# Stress test (concurrent load)
python stress_test.py
```

---

## 🐳 Docker

```bash
# Build
docker build -t adaptive-cyber-defense .

# Run (port 7860 for HF Spaces compatibility)
docker run -p 7860:7860 adaptive-cyber-defense

# Health check
curl http://localhost:7860/_stcore/health
```

---

## 📁 Project Structure

```
adaptive-cyber-defense/
├── app.py                    # FastAPI server (OpenEnv REST API)
├── inference.py              # LLM agent + evaluation runner
├── run.py                    # CLI simulation runner
├── ui.py                     # Streamlit dashboard
├── grader.py                 # Single-source grader formula + thresholds
├── openenv.yaml              # OpenEnv specification
├── environment.py            # Core environment class
├── verify_openenv_compliance.py
│
├── agents/
│   ├── baseline.py           # MITRE-lookup heuristic agent
│   ├── ql_agent.py           # Q-Learning agent
│   └── ignore.py             # Trivial baseline (always ignore)
│
├── engines/
│   ├── attack.py             # Multi-stage kill-chain attack engine
│   ├── adaptive_attacker.py  # Red team with episode-level strategy learning
│   ├── detection.py          # IOC signal generation + noise
│   ├── decision.py           # Action resolution
│   ├── response.py           # Mitigation effects
│   ├── reward.py             # Per-step reward computation
│   └── scoring.py            # Episode grader
│
├── models/                   # Pydantic data models
├── tasks/                    # Task configurations (easy → impossible)
├── tests/                    # 775 test cases
└── training/                 # Q-Learning training scripts
```

---

## 🔌 API Reference

All endpoints return complete JSON. The server never crashes — malformed inputs return HTTP 200 with an error field.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET/POST` | `/reset` | Start new episode |
| `GET/POST` | `/state` | Current observation |
| `POST` | `/step` | Submit action |
| `GET` | `/analytics` | Episode metrics + grader breakdown |

### Observation Schema
```json
{
  "visible_threats":     [{"type": "unknown", "node": "node_2", "stage": "initial", "age": 2}],
  "hidden_threat_count": 3,
  "scan_coverage":       0.4,
  "system_health":       85,
  "score":               0.52,
  "grader_score":        0.48,
  "step":                7,
  "done":                false
}
```

> Note: `visible_threats[].type` is always `"unknown"` until the agent scans the node. Threat classification must be inferred from IOC signals.

---

## 🏆 Hackathon

Built for the **Meta × Hugging Face OpenEnv Hackathon**  
Round 1: March 25 – April 8, 2026

**Live Environment:**  
https://huggingface.co/spaces/shanmukhgara/adaptive-cyber-defense

**GitHub:**  
https://github.com/shanmukhgara48-source/Adaptive-Cyber-Defence

---

## 📄 License

MIT License — see [LICENSE](LICENSE)
