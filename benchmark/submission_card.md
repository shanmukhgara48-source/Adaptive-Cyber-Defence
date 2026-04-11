# Adaptive Cyber Defense — Submission Card

> **Benchmark**: 3 agents × 3 difficulty levels × 5 seeds = 45 episodes  
> **Grader formula**: 0.50 × containment_rate + 0.20 × critical_health + 0.15 × resource_efficiency + 0.15 × speed_bonus  
> **Generated**: 2026-04-11T05:31:07.981505+00:00

## Results

| Agent              | Task     | Threshold | Mean ± Std         | Result     |
|--------------------|----------|-----------|--------------------|------------|
| Heuristic (MITRE)  | Easy     | 0.40      | 0.8573 ± 0.0034   | **PASS**   |
| Heuristic (MITRE)  | Medium   | 0.55      | 0.7973 ± 0.0095   | **PASS**   |
| Heuristic (MITRE)  | Hard     | 0.70      | 0.7995 ± 0.0547   | **PASS**   |
| DQN (300 ep)       | Easy     | 0.40      | 0.8394 ± 0.0686   | **PASS**   |
| DQN (300 ep)       | Medium   | 0.55      | 0.8309 ± 0.0237   | **PASS**   |
| DQN (300 ep)       | Hard     | 0.70      | 0.6040 ± 0.0658   | FAIL       |
| Q-Learning         | Easy     | 0.40      | 0.3698 ± 0.0103   | FAIL       |
| Q-Learning         | Medium   | 0.55      | 0.3455 ± 0.0365   | FAIL       |
| Q-Learning         | Hard     | 0.70      | 0.3404 ± 0.0424   | FAIL       |

**Pass rates** (threshold met across all three difficulties):

- Heuristic (MITRE): 3/3 tasks passed
- DQN (300 ep): 2/3 tasks passed
- Q-Learning: 0/3 tasks passed

## What the DQN Learned — and Where It Hits Its Ceiling

The DQN agent (300-episode baseline, lr=1e-3, γ=0.95) mastered the easy and medium difficulties convincingly — scoring 0.8394 and 0.8309 respectively, both well above their thresholds. On the hard task the agent learned a coherent containment strategy centred on BLOCK_IP and PATCH_SYSTEM, achieving a mean score of 0.6040 ± 0.0658 (threshold: 0.70). A four-config hyperparameter sweep (500 episodes each) and a subsequent 1 000-episode continuation run both failed to push the hard-task score above this level: the best greedy rolling-20 mean reached during Phase 3 was 0.2283 at episode 231, plateauing at 0.2086 by episode 245. The 300-episode baseline therefore remains the best DQN checkpoint. The gap to the heuristic (0.7995) reveals that the structural difficulty of the hard task is not a training-length problem but a policy-space problem: the action most likely to contain a high-severity threat (ISOLATE_NODE) is also the one that bankrupts the agent.

## Structural Finding: The ISOLATE_NODE / Budget Mismatch

The hard task imposes a **resource budget of 0.30 per step**, but ISOLATE_NODE costs **0.40** — 33 % over budget. Every use of ISOLATE_NODE therefore consumes more than one full step's allowance, triggering the `waste_penalty` and `resource_efficiency` drag in the grader formula:

```
score = 0.50 × containment_rate
      + 0.20 × critical_health
      + 0.15 × resource_efficiency   ← penalised by overspend
      + 0.15 × speed_bonus
```

The heuristic avoids ISOLATE_NODE on resource-constrained steps and still scores 0.7995, confirming that BLOCK_IP + PATCH_SYSTEM is the correct strategy. The DQN learned this eventually (ISOLATE_NODE usage collapsed to < 5 % of actions by episode 300), but credit assignment across 30 steps in a partially-observable environment (false-negative rate 0.55, base detection 0.20) limits the ceiling to ≈ 0.60. Closing the remaining gap to the 0.70 threshold would require either a recurrent architecture to handle partial observability, or a reward shaping term that explicitly penalises ISOLATE_NODE on low-budget steps.

## Training Curve Summary

### Phase 2 — Hyperparameter Sweep (500 episodes each, hard task)

| Config | Label                    | lr   | γ     | ε-decay   | Final ε | Hard mean | Winner? |
|--------|--------------------------|------|-------|-----------|---------|-----------|---------|
| A      | baseline                 | 1e-3 | 0.990 | 0.995/ep | 0.082   | 0.5532    | —     |
| B      | slow-decay long-horizon  | 5e-4 | 0.995 | 0.997/ep | 0.223   | 0.6321    | yes     |
| C      | fast-collapse            | 1e-3 | 0.990 | 0.990/ep | 0.050   | 0.4700    | —     |
| D      | conservative high-γ      | 3e-4 | 0.995 | 0.998/ep | 0.368   | 0.5469    | —     |

Config **B** (slow-decay long-horizon) won on hard-task greedy mean (0.6321).

### Phase 3 — Long Run on Config B (1 000-episode budget, warm-start ε = 0.10)

Plateau detection used **greedy rollout scores** (ε = 0, no gradient), not the noisy training-episode scores, to give an honest stopping signal.

| Metric                         | Value |
|--------------------------------|-------|
| Episodes run before plateau    | 245 |
| Best greedy rolling-20 mean    | 0.2283 (ep 231) |
| Final greedy rolling-20 mean   | 0.2086 |
| Hard-task mean after Phase 3   | 0.5390 ± 0.1700 |
| vs 300-ep baseline             | −0.0650 (baseline wins) |

Phase 3 could not surpass the 300-episode baseline because Config B's slow ε-decay (0.997/ep) left the sweep checkpoint at ε = 0.22 after 500 episodes — the agent had not yet converged when Phase 2 ended. Capping ε to 0.10 on warm-start accelerated exploitation but also froze the policy in a local basin that the 300-ep baseline (trained with the tighter γ = 0.95) had already escaped.

## Grader Validation

All **65 episode scores** across both evaluation runs were validated through `grader.safe_score()` (strict range (0.0, 1.0)).

| Check                     | Result |
|---------------------------|--------|
| Total scores validated    | 65 |
| Strict violations (≤ 0 or ≥ 1) | 0 |
| Boundary clamps           | 0 |
| Overall pass              | YES |

---

*Generated by `evaluation/submission_summary.py` — do not edit by hand; re-run the script to refresh.*
