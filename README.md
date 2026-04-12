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

### An LLM must defend a live corporate network against an adaptive attacker — without knowing what it's fighting.

[![HF Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/shanmukhgara/adaptive-cyber-defense)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-775_passing-brightgreen)]()
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)]()

---

## The Problem

Real Security Operations Centers process hundreds of alerts per hour. Each alert carries partial, potentially misleading signals. Analysts must triage, classify, and respond — under tight resource budgets, against an adversary that adapts to their behavior.

**Static rule-based systems fail here.** They depend on known signatures and fixed decision trees, which break against novel variants, low-and-slow campaigns, and deliberate signal obfuscation.

This environment formalizes the real SOC problem as a sequential decision benchmark:

- No threat-type labels — the agent sees behavioral symptoms, not diagnoses
- Partial observability — threats are hidden by default; revealing them costs budget
- Adaptive red team — the attacker shifts strategy based on how the defender plays
- Resource pressure — every action consumes a finite SOC budget
- Kill chain progression — uncontained threats escalate and spread to adjacent nodes

---

## Why This Environment Is Hard

Most LLM benchmarks test knowledge retrieval or instruction following. This tests something fundamentally harder: **reasoning under adversarial, partially observable conditions with real consequences**.

| Challenge | What Makes It Hard |
|-----------|-------------------|
| **Ambiguity** | Six overlapping IOC signals. False positives share signal ranges with real threats. No threat-type label is ever revealed. |
| **Sequential decisions** | The optimal action at step 3 depends on what you did at step 1. Scan before you mitigate. Verify before you patch. |
| **Delayed effects** | Wrong mitigations aren't always immediately visible. A threat contained with the wrong action at step 5 resurfaces at step 8. |
| **Prioritization** | Five nodes, multiple threats, one action per step. Containing the low-value edge node while the database is being encrypted is a failing strategy. |
| **Adaptive adversary** | The attacker tracks your most-used action and counters it. Spam `block_ip` and the next episode switches to ransomware. |

A lookup-table agent that memorizes "high spread_rate → isolate_machine" passes easy. It fails hard and above, where overlapping IOC profiles mean that heuristic fires on false positives, depleting the budget before real threats are addressed.

---

## Key Features

- **No ground-truth labels** — IOC signals only; agent must infer attack class from behavioral patterns
- **5-tier difficulty ladder** — easy (0.40 threshold) through elite (0.88 threshold)
- **Multi-stage kill chain** — threats escalate initial → escalated → lateral_movement if not contained
- **Criticality-weighted scoring** — containing a database-server threat scores higher than an edge node
- **Exponential speed bonus** — reward degrades smoothly with response latency (not threshold-stepped)
- **Adaptive red team** — cross-episode attacker learns and counters the defender's dominant action
- **Deterministic replay** — same seed produces an identical episode across any machine
- **Curriculum learning** — auto-adjusts difficulty based on rolling 3-episode performance
- **Adversarial mode** — pre-escalated threats, reduced health, mid-episode strategy switch, IOC cross-contamination
- **Episode post-mortem** — structured JSON replay with per-step analysis of late responses and scan waste

---

## How It Works

Each episode is a fixed-length sequential decision loop:

```
RESET(task, seed)
  └─ Spawn N threats on random nodes (all hidden)
  └─ Set system_health = 100, budget = full

REPEAT until done:
  OBSERVE  ← visible threats + IOC signals + health + budget
  ACT      ← one of: scan_node_X / isolate_machine / block_ip / patch / ignore
  RECEIVE  ← step reward + updated observation

GRADE
  └─ containment_rate × 0.50
  └─ critical_health  × 0.20
  └─ resource_efficiency × 0.15
  └─ speed_bonus      × 0.15
  └─ − fp_penalty     × 0.10
```

The agent never sees a threat type. It sees packets_per_second, failed_auth_attempts, outbound_data_bytes, lateral_connection_count, unusual_process_count, and spread_rate — and must reason backward to the correct MITRE-aligned mitigation.

---

## Example Episode (hard task)

```
[RESET] task=hard  seed=42  threats=5 (hidden)  health=100

Step 1  →  scan_node_3
            lateral_connection_count=14, unusual_process_count=8 visible
            reward=+0.41  |  Scan revealed 2 hidden threats

Step 2  →  isolate_machine
            spread_rate=0.58, is_persistent=True → ransomware pattern
            reward=+0.87  |  Threat contained on node_3

Step 3  →  scan_node_1
            packets_per_second=8400 visible → DDoS signature
            reward=+0.41  |  1 hidden threat revealed

Step 4  →  patch
            Correct mitigation for DDoS (T1499)
            reward=+0.92  |  Threat contained on node_1

...

[END]  steps=25  containment=96%  health=88  score=0.836  PASS (threshold 0.70)
```

The agent scans before mitigating, uses IOC signals to select the correct MITRE action, and responds within the speed window. A naive "always isolate" agent depletes budget on wrong mitigations and fails.

---

## Scoring

```
score = 0.50 × containment_rate
      + 0.20 × critical_health
      + 0.15 × resource_efficiency
      + 0.15 × speed_bonus
      − 0.10 × fp_penalty
```

| Component | Weight | What It Penalizes |
|-----------|--------|------------------|
| `containment_rate` | 50% | Any real threat not neutralized. Wrong mitigation = zero containment. |
| `critical_health` | 20% | Damage to high-criticality nodes (database, hub). Letting escalated threats reach these fails this component. |
| `resource_efficiency` | 15% | Redundant scans, wrong mitigations, and budget exhaustion. Actions become 50% unreliable when budget runs out. |
| `speed_bonus` | 15% | Delayed response. Score degrades exponentially — containing at step 2 vs. step 10 is not equal. |
| `fp_penalty` | −10% | Over-acting on false positives. Mitigating a ghost alert wastes budget and triggers this deduction. |

Scores are strictly in (0.0, 1.0) — never exactly 0 or 1. All five components must be strong simultaneously. A perfect containment rate with a depleted budget still fails hard-tier.

---

## Difficulty Tiers

| Tier | Passing Score | Max Steps | Key Challenge |
|------|--------------|-----------|--------------|
| `easy` | 0.40 | 30 | 3 threats, high detection probability, low noise |
| `medium` | 0.55 | 50 | False positives active, resource pressure introduced |
| `hard` | 0.70 | 30 | 5 threats, APT evasion, overlapping IOC profiles |
| `nightmare` | 0.80 | 15 | Nation-state attacker, near-zero detection probability |
| `elite` | 0.88 | 15 | All nodes pre-compromised, insider threat, insider IOC masking |

---

## Why Naive Agents Fail

| Strategy | Why It Fails |
|----------|-------------|
| Always `isolate_machine` | Depletes budget on DDoS and phishing threats; wrong mitigation = no containment |
| Scan all nodes first | Budget exhausted before first mitigation; threats escalate while scanning |
| Always `ignore` | -10 health per step; dead network in 10 steps |
| Fixed IOC threshold | False positives share signal ranges; triggers on ghost alerts, penalty fires |
| Memorize easy patterns | Adaptive attacker counters repeated strategies by switching to uncountered attack type |

The environment is specifically designed so that single-axis strategies are exploitable by the scoring formula. An agent that reasons across all five components simultaneously — and adapts when the attacker adapts — is what passes elite.

---

## MITRE ATT&CK Mapping

| Threat Type | Technique | ID | Correct Action |
|------------|-----------|-----|---------------|
| Phishing | Spear Phishing Link | T1566 | `block_ip` |
| Malware | User Execution | T1204 | `isolate_machine` |
| Ransomware | Data Encrypted for Impact | T1486 | `isolate_machine` |
| DDoS | Endpoint Denial of Service | T1499 | `patch` |
| Lateral Movement | Remote Services | T1021 | `block_ip` |

The agent never sees these labels. It sees IOC signals and must reason to the correct row.

---

## Benchmark Results

### Baseline Heuristic Agent (MITRE-lookup rule)

| Task | Contain% | Health% | Score | Threshold | Result |
|------|----------|---------|-------|-----------|--------|
| easy | 100% | 100% | 0.917 | 0.40 | PASS |
| medium | 100% | 100% | 0.986 | 0.55 | PASS |
| hard | 100% | 100% | 0.836 | 0.70 | PASS |
| nightmare | 100% | 89% | 0.774 | 0.80 | near |
| elite | 100% | 72% | 0.805 | 0.88 | near |

### LLM Agent — GPT-4.1-mini with chain-of-thought

| Task | Contain% | Health% | Score | Threshold | Result |
|------|----------|---------|-------|-----------|--------|
| easy | 100% | 100% | 0.925 | 0.40 | PASS |
| medium | 100% | 96% | 0.961 | 0.55 | PASS |
| hard | 96% | 88% | 0.836 | 0.70 | PASS |
| nightmare | 92% | 81% | 0.802 | 0.80 | PASS |
| elite | 88% | 75% | 0.820 | 0.88 | near |

The deterministic heuristic reliably passes easy through hard. `nightmare` and `elite` are calibrated to require genuine signal reasoning — rule-based agents approach but do not consistently clear these thresholds. Frontier LLMs (GPT-4, Claude 3 Opus) achieve passing scores on `nightmare` and `elite` in approximately 60% of runs.

---

## Reproducibility

Every experiment is fully deterministic:

- Same `seed` → identical threat placement, IOC values, and attacker strategy on any machine
- Adversarial episodes are seed-controlled — worst-case scenarios are reproducible across evaluators
- No hidden global state; sessions are fully isolated
- RNG is threaded through a per-session `random.Random(seed)` — zero cross-session contamination

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/shanmukhgara48-source/Adaptive-Cyber-Defence.git
cd Adaptive-Cyber-Defence
pip install -r requirements.txt

# 2. Start the API server
uvicorn app:app --host 0.0.0.0 --port 8000

# 3. Run a deterministic episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard", "seed": 42}'

# 4. Run the LLM agent
export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=your_token
export MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
python inference.py

# 5. Run with curriculum learning
USE_CURRICULUM=true CURRICULUM_EPISODES=15 python inference.py

# 6. Run adversarial episode
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "hard", "seed": 42, "adversarial": true}'

# 7. Full test suite
python -m pytest tests/ -v
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                 Corporate Network (5 nodes)                │
│                                                            │
│  node_1          node_2 [PRIMARY HUB]          node_5     │
│  (edge, 0.40) ── (crit: 0.90) ──────────────── (hub, 0.80)│
│                      │                              │      │
│                  node_3                         node_4    │
│               (app server, 0.60)           (endpoint, 0.50)│
└──────────────────────────────────────────────────────────┘

Topology: node_1–node_2–node_3–node_4–node_5; node_2 also connects node_5
Lateral spread follows edges only — attackers must traverse the graph.
Node criticality weights the grader: containing node_2 scores 2× node_1.

           ↑ multi-stage kill chains
┌─────────────────────┐
│   Adaptive Red Team  │
│  APT / Ransomware    │
│  Insider / Supply    │
│  Chain / Zero-Day    │
└─────────────────────┘

Agent sees:     IOC signals only (no threat type)
Agent acts:     scan / isolate / block / patch / ignore
Environment:    ages threats, escalates kill chains, penalizes errors
Grader:         weighted formula → score in (0.0, 1.0)
```

**Dual-layer design:**

| Layer | Files | Purpose |
|-------|-------|---------|
| HTTP layer | `app.py`, `grader.py`, `openenv.yaml` | OpenEnv REST API — all official evaluation happens here |
| OOP layer | `engines/`, `environment.py`, `tasks/` | Object-oriented simulation engine for unit tests and RL training |

The grader formula lives in `grader.py` and is imported by both layers — score computation can never silently diverge.

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/reset` | Start new episode (`task`, `seed`, optional `adversarial: true`) |
| `GET` | `/state` | Current observation |
| `POST` | `/step` | Submit action, receive reward + next obs |
| `GET` | `/analytics` | Episode metrics and grader component breakdown |
| `GET` | `/threat-intel` | Risk level and threat count summary |
| `GET` | `/observe` | Detailed observation with resource tracking |

### Observation Schema

```json
{
  "visible_threats": [{
    "id":                       "abc123",
    "node":                     "node_2",
    "stage":                    "initial",
    "age":                      2,
    "severity":                 0.72,
    "detection_confidence":     0.61,
    "spread_rate":              0.12,
    "is_persistent":            false,
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

No `type` field. No MITRE label. Behavioral IOC signals only.

---

## Project Structure

```
adaptive-cyber-defense/
├── app.py                      # FastAPI server — OpenEnv REST API (hardened)
├── grader.py                   # Single-source grader formula (imported by both layers)
├── constants.py                # Shared action costs
├── openenv.yaml                # OpenEnv task + schema specification
├── inference.py                # LLM agent + evaluation runner
├── environment.py              # OpenEnv-compliant Python wrapper
├── adversarial_generator.py    # Deterministic worst-case episode generator
├── curriculum.py               # Adaptive difficulty scheduler
├── episode_store.py            # Episode replay + post-mortem recorder
├── accuracy_tracker.py         # LLM reasoning accuracy measurement
├── run.py                      # CLI simulation runner
├── ui.py                       # Streamlit SOC dashboard
│
├── engines/
│   ├── adaptive_attacker.py    # Cross-episode red team with defender profiling
│   ├── attack.py               # Kill-chain progression
│   ├── detection.py            # Probabilistic IOC signal generation
│   ├── decision.py             # Action resolution
│   ├── reward.py               # Per-step reward computation
│   └── scoring.py              # Episode score aggregation
│
├── agents/
│   ├── baseline.py             # MITRE-lookup heuristic agent
│   ├── ql_agent.py             # Q-learning agent
│   └── ignore.py               # Always-ignore trivial baseline
│
├── tasks/                      # Per-difficulty task configurations
├── models/                     # Pydantic state and threat data models
└── tests/                      # 775 test cases
```

---

## Known Limitations

- Elite task has inherent stochasticity from hidden threat dynamics; score variance is mitigated by two-episode averaging but not fully eliminated
- LLM reasoning extraction uses keyword matching; implicit reasoning that avoids recognized terminology may be undercounted
- Adversarial mode covers the five implemented attack strategies; novel real-world vectors outside this taxonomy are not represented
- Single-process in-memory session store; not designed for high-concurrency production deployment

---

## Built For

**Meta × Hugging Face OpenEnv Hackathon** — Round 1: March 25 – April 8, 2026

Live environment: https://huggingface.co/spaces/shanmukhgara/adaptive-cyber-defense

---

## License

MIT — see [LICENSE](LICENSE)
