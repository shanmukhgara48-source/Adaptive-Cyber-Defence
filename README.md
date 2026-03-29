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
---

# Adaptive Cyber Defense Simulator

An OpenEnv-compliant reinforcement learning environment simulating a Security Operations Center (SOC) defending an enterprise network against multi-stage cyber attacks.

## Overview

Threats progress through a kill-chain: **Phishing → Credential Access → Malware Install → Lateral Spread → Exfiltration**. The agent must detect, prioritize, and contain threats using a limited resource budget.

The environment features an **adaptive red-team attacker** that profiles the defender's strategy across episodes and switches attack type to counter it (APT, Ransomware, Insider Threat, Supply Chain, Zero-Day).

## Observation Space

Each `reset()` and `step()` returns a dict observation with:

| Key | Type | Description |
|-----|------|-------------|
| `active_threats` | list | Each threat's stage, severity, MITRE technique, target node |
| `network_state` | dict | Per-node health, compromise status, patch level, criticality |
| `resources` | dict | Remaining scan capacity and response slots |
| `step` | int | Current time step |
| `score` | float | Cumulative reward so far |

## Action Space

11 discrete actions: `BLOCK_IP`, `ISOLATE_NODE`, `PATCH_SYSTEM`, `RUN_DEEP_SCAN`, `SCAN`, `PATCH_VULNERABILITY`, `DECRYPT`, `REVOKE_CREDENTIALS`, `QUARANTINE_SERVICE`, `RESTORE_NODE`, `IGNORE`

## Difficulty Tiers

| Tier   | Threats | Max Steps | Progression Prob | Passing Score |
|--------|---------|-----------|-----------------|---------------|
| Easy   | 1       | 50        | 0.15            | 0.45          |
| Medium | 2       | 40        | 0.25            | 0.45          |
| Hard   | 3       | 30        | 0.40            | 0.45          |

## Quick Start

```bash
pip install -r requirements.txt

# Run a single episode (easy, baseline agent)
python adaptive_cyber_defense/run.py

# Hard task, 5 episodes with adaptive red team
python adaptive_cyber_defense/run.py --task hard --episodes 5 --verbose

# JSON output for evaluation
python adaptive_cyber_defense/run.py --task medium --agent baseline --seed 0 --json

# Streamlit demo UI
streamlit run adaptive_cyber_defense/ui.py
```

## LLM Inference

`inference.py` runs an OpenAI-compatible LLM agent across all three task difficulties:

```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_token_here
python adaptive_cyber_defense/inference.py
```

Falls back to a rule-based heuristic when the LLM is unavailable.

## OpenEnv API

```python
from adaptive_cyber_defense.environment import CyberDefenseEnv

env = CyberDefenseEnv(task="easy", seed=42)
obs = env.reset()           # dict observation

obs, reward, done, info = env.step("SCAN")  # reward ∈ [-1.0, 1.0]
state = env.state()         # full state dict
```

## Baseline Scores

| Task   | Baseline Agent | LLM Agent (fallback) |
|--------|---------------|----------------------|
| Easy   | 0.87          | 0.85                 |
| Medium | 0.83          | 0.81                 |
| Hard   | 0.85          | 0.69                 |

## Architecture

- `environment.py` — OpenEnv-compliant wrapper (`CyberDefenseEnv`)
- `env.py` — Core simulation (`AdaptiveCyberDefenseEnv`)
- `engines/attack.py` — Kill-chain progression engine
- `engines/adaptive_attacker.py` — Red-team attacker with defender profiling
- `engines/detection.py` — Probabilistic SOC detection system
- `engines/decision.py` — AI recommendation engine
- `models/` — State, action, network, MITRE ATT&CK models
- `inference.py` — LLM agent runner (OpenAI-compatible)
- `ui.py` — Streamlit SOC dashboard
- `run.py` — CLI episode runner

## MITRE ATT&CK Integration

Every threat tracks its current MITRE ATT&CK technique:

| Kill-Chain Stage | Technique ID | Tactic |
|-----------------|-------------|--------|
| Phishing | T1566 | Initial Access |
| Credential Access | T1078 | Credential Access |
| Malware Install | T1204 | Execution |
| Lateral Spread | T1021 | Lateral Movement |
| Exfiltration | T1041 | Exfiltration |
