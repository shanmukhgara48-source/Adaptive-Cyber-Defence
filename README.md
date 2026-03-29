# Adaptive Cyber Defense Simulator

An OpenEnv-compliant reinforcement learning environment simulating a Security Operations Center (SOC) defending an enterprise network against multi-stage cyber attacks.

## Overview

Threats progress through a kill-chain: **Phishing → Credential Access → Malware Install → Lateral Spread → Exfiltration**. The agent must detect, prioritize, and contain threats using a limited resource budget.

## Difficulty Tiers

| Tier   | Threats | Max Steps | Progression Prob |
|--------|---------|-----------|-----------------|
| Easy   | 1       | 50        | 0.15            |
| Medium | 2       | 40        | 0.25            |
| Hard   | 3       | 30        | 0.40            |

## Quick Start

```bash
# Run a single episode (easy, baseline agent)
python adaptive_cyber_defense/run.py

# Hard task, 5 episodes, JSON output
python adaptive_cyber_defense/run.py --task hard --episodes 5 --json

# Streamlit demo UI
streamlit run adaptive_cyber_defense/ui.py
```

## Actions

`BLOCK_IP`, `ISOLATE_NODE`, `PATCH_SYSTEM`, `RUN_DEEP_SCAN`, `IGNORE`, `DECRYPT`, `REVOKE_CREDENTIALS`, `QUARANTINE_SERVICE`, `RESTORE_NODE`, `SCAN`, `PATCH_VULNERABILITY`

## API

```python
from adaptive_cyber_defense import AdaptiveCyberDefenseEnv
from adaptive_cyber_defense.models.action import Action, ActionInput

env = AdaptiveCyberDefenseEnv()
state = env.reset(seed=42)

done = False
while not done:
    action = ActionInput(action=Action.BLOCK_IP)
    state, reward, done, info = env.step(action)
```
