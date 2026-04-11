"""
Upload DQN weights to Hugging Face Hub.

Uploads agents/dqn_weights.json to a HF model repository and generates
a model card (README.md) with the benchmark scores table embedded.

Usage
-----
    # Requires HF_TOKEN in environment
    HF_TOKEN=hf_xxx python scripts/upload_to_hf.py

    # Override repo and weights path
    HF_TOKEN=hf_xxx python scripts/upload_to_hf.py \
        --repo Glory-royyuru/adaptive-cyber-defense-dqn \
        --weights agents/dqn_weights.json

Flags
-----
    --repo      HF repo id (default: Glory-royyuru/adaptive-cyber-defense-dqn)
    --weights   Local path to weights file (default: agents/dqn_weights.json)
    --private   Create/keep repo private (default: public)

Output
------
    HF repo with:
      - dqn_weights.json   (the trained network weights)
      - README.md          (model card with benchmark table)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent     # adaptive_cyber_defense/scripts/
_PKG  = _HERE.parent                        # adaptive_cyber_defense/
_ROOT = _PKG.parent                         # /Users/…/Documents/

for _p in (_ROOT, _PKG):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Benchmark scores (from last 3-seed run: seeds 42/43/44, DQN trained 300 eps on hard)
# ---------------------------------------------------------------------------

_BENCHMARK_RESULTS = {
    "heuristic": {
        "easy":   {"mean": 0.8550, "std": 0.0009, "threshold": 0.40, "result": "PASS"},
        "medium": {"mean": 0.7924, "std": 0.0111, "threshold": 0.55, "result": "PASS"},
        "hard":   {"mean": 0.7716, "std": 0.0557, "threshold": 0.70, "result": "PASS"},
    },
    "dqn": {
        "easy":   {"mean": 0.7567, "std": 0.1447, "threshold": 0.40, "result": "PASS"},
        "medium": {"mean": 0.8230, "std": 0.0204, "threshold": 0.55, "result": "PASS"},
        "hard":   {"mean": 0.5911, "std": 0.1626, "threshold": 0.70, "result": "FAIL"},
    },
}

_DEFAULT_REPO    = "Glory-royyuru/adaptive-cyber-defense-dqn"
_DEFAULT_WEIGHTS = _PKG / "agents" / "dqn_weights.json"


# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------

def _build_model_card(repo_id: str, weights_path: Path) -> str:
    weights_kb = weights_path.stat().st_size // 1024

    rows = []
    for agent, tasks in _BENCHMARK_RESULTS.items():
        for task, stats in tasks.items():
            rows.append(
                f"| {agent:<10} | {task:<8} | {stats['threshold']:.2f}      "
                f"| {stats['mean']:.4f} ± {stats['std']:.4f} | {stats['result']} |"
            )

    table = "\n".join(rows)

    return f"""\
---
license: mit
tags:
  - reinforcement-learning
  - cybersecurity
  - dqn
  - numpy
language: []
---

# Adaptive Cyber Defense — DQN Agent

A pure-NumPy Deep Q-Network (DQN) agent trained on the Adaptive Cyber Defense
Simulator. The agent learns to contain network intrusions across a simulated
8-node enterprise topology using 5 discrete defense actions.

## Architecture

- **State space**: 54-dimensional float32 vector (8 nodes × 6 features + 6 globals)
- **Action space**: 5 discrete actions (BLOCK_IP, ISOLATE_NODE, PATCH_SYSTEM, RUN_DEEP_SCAN, IGNORE)
- **Network**: 3-layer MLP — 54 → 128 → 128 → 5, ReLU activations, He init
- **Optimizer**: Manual Adam (β₁=0.9, β₂=0.999, ε=1e-8), implemented in NumPy
- **Training**: 300 episodes on `hard` task, ε-greedy decay 1.0→0.05 over 150 episodes
- **Weights file**: `dqn_weights.json` ({weights_kb} KB)

## Benchmark Results

Evaluation: 3 seeds (42, 43, 44), greedy policy (ε=0), `BaseTask.run()` scoring.

| Agent      | Task     | Threshold | Mean ± Std          | Result |
|------------|----------|-----------|---------------------|--------|
{table}

DQN beats the heuristic baseline on medium (+0.031 delta) after training on the hard task.
The hard-task gap (−0.181) reflects the inherent challenge of a tabula-rasa agent vs.
a MITRE-knowledgeable rule-based system with limited training episodes.

## Weights Format

The weights file is a JSON dict with keys `layers` (list of layer weight/bias arrays),
`epsilon`, `state_dim`, and `n_actions`. Load via:

```python
from adaptive_cyber_defense.agents.dqn_agent import DQNAgent
agent = DQNAgent.load("dqn_weights.json")
agent.epsilon = 0.0  # greedy eval mode
```

## Grader Formula

```
score = 0.50 × containment_rate
      + 0.20 × critical_health
      + 0.15 × resource_efficiency
      + 0.15 × speed_bonus
```

Scores are strictly in (0.0, 1.0) via `safe_score()` from `grader.py`.

## Repository

[Glory-royyuru/adaptive-cyber-defense-dqn](https://huggingface.co/{repo_id})
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload DQN weights to Hugging Face Hub")
    p.add_argument("--repo",    type=str, default=_DEFAULT_REPO,
                   help=f"HF repo id (default: {_DEFAULT_REPO})")
    p.add_argument("--weights", type=str, default=str(_DEFAULT_WEIGHTS),
                   help="Local path to weights JSON file")
    p.add_argument("--private", action="store_true",
                   help="Create/keep repo private")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def upload(args: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN environment variable is not set.")
        print("  Export it:  export HF_TOKEN=hf_...")
        sys.exit(1)

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights file not found at {weights_path}")
        sys.exit(1)

    from huggingface_hub import HfApi, create_repo

    api    = HfApi(token=token)
    sep    = "=" * 68
    print(f"\n{sep}")
    print(f"  HF Upload — repo={args.repo}  weights={weights_path}")
    print(sep)

    # ── Ensure repo exists ────────────────────────────────────────────────────
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="model",
            private=args.private,
            exist_ok=True,
            token=token,
        )
        print(f"  [repo] Ready: https://huggingface.co/{args.repo}")
    except Exception as e:
        print(f"  [repo] create_repo error: {e}")
        sys.exit(1)

    # ── Upload weights ────────────────────────────────────────────────────────
    print(f"  [upload] Uploading {weights_path.name} ({weights_path.stat().st_size // 1024} KB) ...")
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo="dqn_weights.json",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Upload DQN weights (pure-NumPy, 300 episodes hard task)",
    )
    print(f"  [upload] dqn_weights.json uploaded.")

    # ── Upload model card ─────────────────────────────────────────────────────
    card_content = _build_model_card(args.repo, weights_path)
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(card_content)
        card_tmp = f.name

    api.upload_file(
        path_or_fileobj=card_tmp,
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type="model",
        commit_message="Add model card with benchmark scores table",
    )
    Path(card_tmp).unlink(missing_ok=True)
    print(f"  [upload] README.md (model card) uploaded.")

    print(f"\n  Done. View at: https://huggingface.co/{args.repo}")
    print(sep + "\n")


if __name__ == "__main__":
    upload(_parse_args())
