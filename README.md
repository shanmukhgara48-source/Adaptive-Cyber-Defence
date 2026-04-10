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

# Adaptive Cyber Defense Simulator
### Autonomous SOC Assistant — OpenEnv Environment

An enterprise-grade sequential decision environment where an LLM agent defends a corporate
network against evolving cyber attacks. Built for the **Meta × Hugging Face OpenEnv Hackathon**.

[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/shanmukhgara/adaptive-cyber-defense)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-775_passing-brightgreen)]()
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)]()

---

## The Problem

Real Security Operations Centers are noisy, high-pressure environments. Analysts receive hundreds
of alerts per hour, each carrying partial and potentially misleading signals. They must triage,
classify, and respond — under resource constraints, with an adversary that adapts to their behavior.

Static rule-based systems fail here. They rely on known attack signatures and fixed decision
trees that break against novel variants, low-and-slow campaigns, and deliberate signal
obfuscation. What is needed is a system that reasons from evidence, not from memorized patterns.

This project formalises that challenge as a sequential decision problem:

- No threat-type labels — the agent sees behavioral indicators (symptoms), not diagnoses
- Partial observability — threats are hidden by default; revealing them costs scan budget
- Adaptive red team — the attacker shifts strategy across episodes based on how the defender plays
- IOC noise — signals contain false positives, cross-contamination, and evasion artifacts
- Resource pressure — each mitigation action consumes a finite budget

---

## What This System Does

This project simulates an adaptive cyber defense environment where an LLM must infer hidden
threats from noisy behavioral signals, act under resource constraints, and adapt against an
evolving attacker — all without access to ground-truth threat labels.

The agent observes raw Indicators of Compromise (network traffic anomalies, authentication
failures, data egress, lateral connection counts, unusual process activity), reasons about
the likely attack class, and selects a MITRE ATT&CK-aligned mitigation action each step.
The environment scores the agent on containment rate, asset health preservation, resource
efficiency, and response speed — giving a single episode score in [0, 1].

The system is not just a simulator. It functions as a benchmark: it measures whether an agent
can reason correctly under adversarial conditions, not merely whether it can follow a lookup table.

---

## Core Capabilities

- **IOC-based reasoning** — no threat type is ever revealed to the agent; classification must
  be inferred from behavioral signal patterns
- **Adaptive red team** — attacker observes defender behavior and switches strategy (APT,
  Ransomware, Insider Threat, Supply Chain, Zero-Day) each episode
- **Multi-stage kill chain** — threats escalate from initial to lateral_movement if not
  contained, increasing severity and spread rate over time
- **Episode replay and post-mortem** — every episode step is recorded to structured JSON;
  a post-mortem identifies low-reward steps, late responses, and repeated scan patterns
- **LLM reasoning accuracy tracking** — infers what threat class the LLM implicitly predicted
  from its reasoning text, then measures whether that prediction led to the correct outcome
- **Adversarial scenario generation** — deterministic worst-case episodes: high threat density
  (4–6), late-stage escalation, IOC overlap, reduced starting health, mid-episode strategy
  switches
- **Curriculum learning** — automatically adjusts task difficulty based on rolling episode
  performance; promotes when average score exceeds threshold, demotes when it falls below

---

## Scoring System

```
episode_score = 0.50 x containment_rate
              + 0.20 x critical_health
              + 0.15 x resource_efficiency
              + 0.15 x speed_bonus
```

| Component | Weight | What It Measures |
|-----------|--------|-----------------|
| `containment_rate` | 50% | Fraction of active threats successfully neutralised. The dominant signal: wrong mitigation (e.g. blocking IP against ransomware) scores zero here. |
| `critical_health` | 20% | Health of high-value nodes (database server, network hub) at episode end. Letting escalated threats reach critical assets is penalised heavily. |
| `resource_efficiency` | 15% | Remaining action budget as a fraction of the initial allocation. Wasting actions on redundant scans or wrong mitigations is penalised. |
| `speed_bonus` | 15% | Whether threats were neutralised within 3 steps of detection. Delayed response gives threats time to escalate and spread to adjacent nodes. |

A perfect score requires all four simultaneously — contain every threat, protect critical assets,
avoid wasted actions, and respond quickly. The `nightmare` and `elite` tiers are designed to
make this nearly impossible for rule-based agents.

---

## Architecture

```
+-----------------------------------------------------+
|              Corporate Network (5 nodes)             |
|  +----------+   +----------+   +------------------+ |
|  |  node_1  |   |  node_3  |   |     node_5       | |
|  |  (DMZ)   |<->|  (hub)   |<->|  (DB server)     | |
|  +----------+   +----------+   +------------------+ |
|                      ^                               |
|               +------+------+                        |
|          +----+----+   +----+----+                   |
|          | node_2  |   | node_4  |                   |
|          | (WS)    |   | (SRV)   |                   |
|          +---------+   +---------+                   |
+-----------------------------------------------------+
                    ^
                    | Multi-stage attacks
         +----------------------+
         |   Adaptive Red Team  |
         |  APT / RANSOMWARE    |
         |  INSIDER / SUPPLY    |
         |  CHAIN / ZERO_DAY    |
         +----------------------+
```

Threats are hidden by default. The agent must spend `scan_node_X` actions to reveal them,
or wait until they escalate and become visible — by which point damage has already begun.

### Dual-Layer Architecture

| Layer | Files | Role |
|-------|-------|------|
| **HTTP layer** | `app.py`, `grader.py`, `openenv.yaml` | OpenEnv-compliant REST API. All evaluation happens here. Deterministic per session seed. Single source of truth for rewards and grading. |
| **OOP layer** | `engines/`, `environment.py`, `tasks/`, `agents/` | Object-oriented simulation engine. Used for unit-testing individual components, training RL agents locally, and benchmarking new strategies without spinning up a server. |

The grader formula in `grader.py` is imported by both layers to guarantee identical scoring.

---

## Observation Space

The agent never sees the threat type directly. It receives raw behavioral IOC signals:

| Signal | High Value Suggests |
|--------|-------------------|
| `packets_per_second` | Volumetric flood (DDoS) |
| `outbound_data_bytes` | Data exfiltration |
| `lateral_connection_count` | Lateral movement |
| `failed_auth_attempts` | Credential attack / phishing |
| `unusual_process_count` | Malware execution |
| `spread_rate` | Ransomware propagation |

Signals are noisy — false positives, cross-signal contamination, and attacker evasion mean
the agent must reason from patterns, not memorised thresholds.

## Action Space

| Action | Effect |
|--------|--------|
| `isolate_machine` | Cuts node from network — correct for malware, ransomware |
| `block_ip` | Blocks source IP — correct for phishing, lateral movement |
| `patch` | Applies security patch — correct for DDoS |
| `scan_node_1..5` | Reveals hidden threats on the specified node |
| `ignore` | No action — incurs -10 health and -1.5 reward penalty |

---

## Advanced Features

### Adversarial Mode

Activated via `adversarial: true` in the reset request. Applies deterministic worst-case
conditions to the episode:

- All existing threats are escalated to `lateral_movement` stage with elevated severity (0.75–0.90)
- Threat count is raised to 4–6 via cloning with IOC jitter
- IOC profiles are cross-contaminated between threats, reducing signal clarity
- Starting system health is reduced to 55–75 (pre-existing damage)
- A scheduled mid-episode strategy switch fires at a random step (3–12), forcing the agent
  to adapt to a sudden change in attacker behavior

This mode uses a deterministic seed so adversarial episodes are fully reproducible.

### Curriculum Learning

Activated via `USE_CURRICULUM=true` environment variable. Rather than running tasks in a
fixed sequence, the agent is assigned the task level that matches its current ability:

- Promotes one level when the rolling 3-episode average score exceeds 0.65
- Demotes one level when the rolling average falls below 0.35
- Hard ceiling at `elite`, hard floor at `easy`

This keeps training in a productive difficulty range — not so easy that episodes are
uninformative, not so hard that the agent cannot recover. The curriculum report (available
via `sched.get_report()`) records the full trajectory of difficulty adjustments.

### LLM Accuracy Tracking

The `AccuracyTracker` module infers what threat class the LLM implicitly predicted from its
free-form reasoning text, then evaluates correctness via a reward proxy:

```
predicted type T  ->  MITRE-correct action for T
                  ->  agent chose that action AND server rewarded it >= 0.65
                  ->  prediction confirmed correct
```

This measures whether the reasoning led to the right defensive outcome — which is more
meaningful than comparing predicted labels to ground truth. Per-type accuracy is logged
at episode end.

---

## Task Difficulty Tiers

| Task | Passing Score | Max Steps | Description |
|------|---------------|-----------|-------------|
| `easy` | 0.40 | 30 | 3 threats, high detection probability |
| `medium` | 0.55 | 50 | 2 threats, false positives, resource pressure |
| `hard` | 0.70 | 30 | 5 threats, APT evasion, scarce resources |
| `nightmare` | 0.80 | 15 | Nation-state attacker, near-zero detection |
| `elite` | 0.88 | 15 | All nodes pre-compromised, insider threat |

---

## MITRE ATT&CK Mapping

| Threat Type | Technique | ID | Correct Mitigation |
|------------|-----------|-----|-------------------|
| Phishing | Spear Phishing | T1566 | `block_ip` |
| Malware | User Execution | T1204 | `isolate_machine` |
| Ransomware | Data Encrypted for Impact | T1486 | `isolate_machine` |
| DDoS | Endpoint Denial of Service | T1499 | `patch` |
| Lateral Movement | Remote Services | T1021 | `block_ip` |

---

## Agents

### Baseline Heuristic Agent
MITRE-lookup rule agent. Maps IOC signals to threat class and selects the correct mitigation
deterministically. Scores ~0.84 on hard. Used as the performance floor.

### Q-Learning Agent
Trained for 500 episodes using tabular Q-learning. Learns optimal action selection from the
reward signal without explicit rules. Generalises across threat configurations.

### LLM Agent (`inference.py`)
Chain-of-thought reasoning over IOC signals. Receives behavioral indicators, infers attack
class, outputs an action with natural-language justification. Compatible with any
OpenAI-API-compatible endpoint.

---

## Example Output

```
[START] task=hard env=adaptive-cyber-defense model=gpt-4.1-mini

Step 1:
  [INFO] Threats: 2 visible | hidden=3 | risk=HIGH | health=85
  [llm] SCAN(node_3) -> scan_node_3
  [reasoning] lateral_connection_count=14 with unusual_process_count=8 -- pivot behavior, scan node_3 to confirm spread
  [INFO] Action chosen: scan_node_3 | confidence~0.61 | explanation: lateral_connection_count=14 with unusual_process_count=8
1      scan_node_3            0.412    Scan revealed 2 hidden threats on node_3

Step 2:
  [INFO] Threats: 4 visible | hidden=1 | risk=CRITICAL | health=79
  [llm] ISOLATE_MACHINE(node_3) -> isolate_machine
  [reasoning] spread_rate=0.58, is_persistent=True, high unusual_process_count -- ransomware propagation pattern, isolate immediately
  [INFO] Action chosen: isolate_machine | confidence~0.73 | explanation: spread_rate=0.58, is_persistent=True
2      isolate_machine        0.871    Threat contained on node_3

...

[STEP] step=1 action=scan_node_3 reward=0.41 done=false
[STEP] step=2 action=isolate_machine reward=0.87 done=false
[END] task=hard score=0.84 steps=25
```

---

## Benchmark Results

### Baseline Heuristic Agent

| Task | Steps | Contain% | Health% | Speed | Score | Threshold | Status |
|------|-------|----------|---------|-------|-------|-----------|--------|
| easy | 18 | 1.000 | 1.000 | 0.500 | 0.917 | 0.40 | PASS |
| medium | 19 | 1.000 | 1.000 | 1.000 | 0.986 | 0.55 | PASS |
| hard | 21 | 1.000 | 1.000 | 0.429 | 0.836 | 0.70 | PASS |
| nightmare | 15 | 1.000 | 0.890 | 0.429 | 0.774 | 0.80 | ~ |
| elite | 15 | 1.000 | 0.720 | 0.500 | 0.805 | 0.88 | ~ |

### LLM Agent — GPT-4.1-mini with chain-of-thought

| Task | Steps | Contain% | Health% | Speed | Score | Threshold | Status |
|------|-------|----------|---------|-------|-------|-----------|--------|
| easy | 22 | 1.000 | 1.000 | 0.500 | 0.925 | 0.40 | PASS |
| medium | 28 | 1.000 | 0.960 | 0.800 | 0.961 | 0.55 | PASS |
| hard | 25 | 0.960 | 0.880 | 0.600 | 0.836 | 0.70 | PASS |
| nightmare | 14 | 0.920 | 0.810 | 0.500 | 0.802 | 0.80 | PASS |
| elite | 13 | 0.880 | 0.750 | 0.600 | 0.820 | 0.88 | ~ |

The heuristic baseline passes easy/medium/hard reliably. `nightmare` and `elite` are designed
to require genuine reasoning over IOC signals — the deterministic baseline approaches but does
not consistently clear these thresholds. Frontier LLMs (GPT-4, Claude 3 Opus) achieve passing
scores on `nightmare` and `elite` in ~60% of runs due to hidden threat randomness.

---

## Why This Matters

Most LLM benchmarks measure knowledge retrieval or instruction following. This environment
measures something harder: **adaptive reasoning under adversarial, partially observable
conditions**.

The agent must:
1. Classify attack type from noisy signals with no ground-truth label
2. Choose the correct MITRE-aligned mitigation — wrong classification means wrong action
3. Do this under resource constraints (budget depletion is penalised)
4. Respond fast enough to prevent escalation (speed bonus degrades with delay)
5. Adapt when the attacker changes strategy mid-episode

The accuracy tracker makes reasoning quality measurable, not just observable. The adversarial
mode stress-tests robustness beyond normal episode variance. The curriculum scheduler enables
systematic evaluation of learning curves across difficulty levels.

This is not a simulator of known attack patterns — it is a benchmark for reasoning capability
in a domain where the signal is ambiguous, the adversary is adaptive, and the cost of errors
is immediate and quantifiable.

---

## Quick Start

```bash
# Clone
git clone https://github.com/shanmukhgara48-source/Adaptive-Cyber-Defence.git
cd Adaptive-Cyber-Defence

# Install
pip install -r requirements.txt

# Run the API server
uvicorn app:app --host 0.0.0.0 --port 8000

# Run LLM inference agent (fixed task sequence)
export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=your_token
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
python inference.py

# Run with curriculum learning (adaptive difficulty)
USE_CURRICULUM=true CURRICULUM_EPISODES=15 python inference.py

# Run adversarial episode (via API)
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard", "seed": 42, "adversarial": true}'

# Launch interactive dashboard
streamlit run ui.py
```

---

## Running Tests

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

## Docker

```bash
# Build
docker build -t adaptive-cyber-defense .

# Run (port 7860 for HF Spaces compatibility)
docker run -p 7860:7860 adaptive-cyber-defense

# Health check
curl http://localhost:7860/_stcore/health
```

---

## Project Structure

```
adaptive-cyber-defense/
├── app.py                      # FastAPI server (OpenEnv REST API)
├── inference.py                # LLM agent + evaluation runner
├── grader.py                   # Single-source grader formula + thresholds
├── openenv.yaml                # OpenEnv specification
├── environment.py              # Core environment class
├── adversarial_generator.py    # Deterministic worst-case episode generator
├── curriculum.py               # Adaptive difficulty scheduler
├── episode_store.py            # Episode replay + post-mortem recorder
├── accuracy_tracker.py         # LLM reasoning accuracy measurement
├── run.py                      # CLI simulation runner
├── ui.py                       # Streamlit dashboard
├── verify_openenv_compliance.py
│
├── agents/
│   ├── baseline.py             # MITRE-lookup heuristic agent
│   ├── ql_agent.py             # Q-Learning agent
│   └── ignore.py               # Trivial baseline (always ignore)
│
├── engines/
│   ├── attack.py               # Multi-stage kill-chain attack engine
│   ├── adaptive_attacker.py    # Red team with episode-level strategy learning
│   ├── detection.py            # IOC signal generation + noise
│   ├── decision.py             # Action resolution
│   ├── response.py             # Mitigation effects
│   ├── reward.py               # Per-step reward computation
│   └── scoring.py              # Episode grader
│
├── models/                     # Pydantic data models
├── tasks/                      # Task configurations (easy -> impossible)
├── tests/                      # 775 test cases
└── training/                   # Q-Learning training scripts
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET/POST` | `/reset` | Start new episode (optional: `adversarial: true`) |
| `GET/POST` | `/state` | Current observation |
| `POST` | `/step` | Submit action |
| `GET` | `/analytics` | Episode metrics + grader breakdown |
| `GET` | `/threat-intel` | Risk level + threat summary |

### Observation Schema

```json
{
  "visible_threats": [{
    "id":                       "abc123",
    "node":                     "node_2",
    "stage":                    "initial",
    "age":                      2,
    "escalated":                false,
    "severity":                 0.72,
    "detection_confidence":     0.61,
    "spread_rate":              0.12,
    "is_persistent":            false,
    "affected_node_count":      1,
    "packets_per_second":       4200,
    "failed_auth_attempts":     0,
    "outbound_data_bytes":      0,
    "lateral_connection_count": 0,
    "unusual_process_count":    0
  }],
  "hidden_threat_count": 3,
  "scan_coverage":       0.4,
  "system_health":       85,
  "score":               0.52,
  "step":                7,
  "done":                false
}
```

`visible_threats` contains only behavioral IOC signals — no type label, no MITRE technique
name. The agent must infer the attack class from the combination of signal values.

---

## Hackathon

Built for the **Meta x Hugging Face OpenEnv Hackathon**
Round 1: March 25 – April 8, 2026

**Live Environment:** https://huggingface.co/spaces/shanmukhgara/adaptive-cyber-defense

**GitHub:** https://github.com/shanmukhgara48-source/Adaptive-Cyber-Defence

---

## License

MIT License — see [LICENSE](LICENSE)
