# Developer Reference — Adaptive Cyber Defense

Architecture, design decisions, and extension guide for contributors and maintainers.

---

## 1. System Architecture

The system has two parallel layers. Both share a single grader formula. Only the HTTP layer is used for official OpenEnv evaluation.

```
┌─────────────────────────────────────────────────────────┐
│                     HTTP Layer                           │
│                                                          │
│   app.py  ──  grader.py  ──  constants.py               │
│     │               │                                    │
│  FastAPI         Single-source              openenv.yaml │
│  REST API        scoring formula            Task spec    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     OOP Layer                            │
│                                                          │
│  environment.py  ──  env.py  ──  engines/               │
│       │                              │                   │
│  OpenEnv wrapper             attack, detection,         │
│  (Python API)                decision, reward           │
│                                                          │
│  tasks/          agents/       models/                   │
│  easy/medium/    baseline      Pydantic state +          │
│  hard/nightmare  ql_agent      threat dataclasses        │
└─────────────────────────────────────────────────────────┘
```

**Key invariant:** `grader.py` is imported by both layers. The score formula is never duplicated. Any divergence between layers is a bug.

### Component Responsibilities

| File | Responsibility |
|------|---------------|
| `app.py` | FastAPI server. Session management, action dispatch, state mutation, reward computation, grader invocation. The only file an evaluator needs to run. |
| `grader.py` | Authoritative episode score formula. `safe_score()`, `TASK_PASSING_SCORES`, `compute_grader_score()`. Read-only in production. |
| `constants.py` | Shared action costs (`COST_RAW`, `get_scaled_costs()`). Imported by app.py and inference.py. |
| `openenv.yaml` | Machine-readable task spec. Observation schema, action space, difficulty tiers, passing thresholds. |
| `engines/adaptive_attacker.py` | Cross-episode red team. Tracks defender behavior profile, selects counter-strategy. |
| `engines/attack.py` | Kill-chain state machine. Threat aging, stage escalation, spread mechanics. |
| `inference.py` | LLM agent runner. Calls the HTTP API, formats observations for the LLM, parses action from response. |

---

## 2. Environment Flow

### Session Lifecycle

```
POST /reset
  │
  ├── Allocate Session (or reuse existing sid)
  ├── Seed session RNG:  session.rng = random.Random(seed)
  ├── Seed attacker RNG: AdaptiveAttacker(seed = ATTACKER_SEED ^ (seed & 0xFFFFFFFF))
  ├── Load task config (easy / medium / hard / nightmare / elite)
  ├── Spawn N threats on random nodes  ← all hidden, all visible=False
  ├── Set system_health = 100, budget = full
  └── Return initial observation

POST /step
  │
  ├── Validate session, check done flag
  ├── Coerce + validate action string
  ├── _update_visibility()    ← may reveal new threats
  ├── _age_threats()          ← advance age, trigger escalations
  ├── Match action to visible threats
  ├── Apply action effects (containment, health, budget)
  ├── Compute step reward
  ├── Append action to episode_actions_taken  ← BEFORE calling _obs()
  ├── _obs()                  ← builds observation (sees current action)
  ├── Check termination (health ≤ 0 or step ≥ max_steps)
  └── Return safe_response(obs, action, reward)

GET /analytics
  │
  ├── Compute total budget spent (difficulty-scaled scan costs)
  ├── Invoke compute_grader_score()
  └── Return component breakdown + current episode score

GET /state, /observe, /threat-intel
  └── Read-only views of current session state
```

### Session Store

Sessions are stored in an `OrderedDict[str, Session]` with LRU eviction at 256 sessions. The most-recently-used session is tracked in `_LATEST_SID` for stateless clients.

```python
# Session object (simplified)
@dataclass
class Session:
    sid:                  str
    rng:                  random.Random      # seeded per session
    attacker:             AdaptiveAttacker   # seeded with XOR of global + session seed
    state:                dict               # threats, health, step, done
    task_name:            str
    task_config:          dict
    effective_config:     dict | None        # overrides for adversarial mode
    episode_actions_taken: list[str]         # append BEFORE _obs()
    episode_rewards:      list[float]
    action_counts:        dict[str, int]
    episode_history:      list[dict]
```

---

## 3. State and Data Model

### Core State Dict

```python
state = {
    "threats":       list[dict],   # all threats — visible and hidden
    "scanned_nodes": set[str],     # nodes the agent has scanned this episode
    "system_health": int,          # 0–100; episode ends at 0
    "score":         float,        # cumulative reward (not grader score)
    "step":          int,
    "done":          bool,
}
```

### Threat Lifecycle

```
SPAWN
  visible = False
  stage   = "initial"
  age     = 0

EACH STEP:
  age += 1
  if age >= escalation_threshold (8):
      stage = "escalated"
  if stage == "lateral_movement" or node in scanned_nodes:
      visible = True
  if age >= age_visibility_threshold (5) and detect_roll passes:
      visible = True

CONTAINS:
  contained = True
  visible   = True    ← always visible after containment

RESURFACE (persistent threats only):
  contained threats with resurface_risk > 0 may reactivate
  monitor action reduces resurface_risk
```

### Threat Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | UUID for this threat instance |
| `node` | str | Which network node it occupies |
| `type` | str | Internal type — never sent to agent |
| `stage` | str | `initial` / `escalated` / `lateral_movement` |
| `age` | int | Steps since spawn |
| `severity` | float | 0–1; affects health damage per step |
| `spread_rate` | float | IOC signal — also affects spread mechanics |
| `is_persistent` | bool | Whether threat can resurface after containment |
| `is_false_positive` | bool | Ghost alert — correct mitigation still costs budget |
| `contained` | bool | True after successful mitigation |
| `visible` | bool | Whether agent can observe this threat |
| `pending_action` | str \| None | Delayed mitigation in progress (e.g. `patch` takes 2 steps) |

### Node Structure

```python
NODES = ["node_1", "node_2", "node_3", "node_4", "node_5"]

# Topology: node_1 — node_2 — node_3 — node_4 — node_5
#                        └───────────────── node_5
# node_2 is the primary hub (3 edges); node_5 is the secondary hub (2 edges).
# Lateral spread is topology-constrained: threats can only move along these edges.
NODE_CRITICALITY = {
    "node_1": 0.40,  # edge leaf — low-value ingress point
    "node_2": 0.90,  # primary hub — connects node_1, node_3, node_5
    "node_3": 0.60,  # mid-chain application server
    "node_4": 0.50,  # secondary endpoint
    "node_5": 0.80,  # secondary hub — connects node_2, node_4
}
```

Node criticality feeds into `compute_criticality_weighted_containment()` in `grader.py`. Containing node_2 (0.90) contributes 2.25× more to the effective containment term than containing node_1 (0.40). Agents that contain the easiest-to-reach threat rather than the most critical one receive a lower criticality-weighted score.

---

## 4. Reward System

### Step Reward

Step rewards are computed per-action per matched threat. They are clamped to `[-2.0, 2.0]`.

| Condition | Reward |
|-----------|--------|
| Correct MITRE mitigation | +1.0 |
| Correct mitigation on unverified threat (50% success) | +0.5 (expected) |
| Scan reveals hidden threat | +0.02 per threat revealed |
| Scan reveals nothing new | −0.01 |
| Wrong mitigation | −0.5 |
| `ignore` with visible threat | −1.0 + −10 health |
| `ignore` with no visible threats | 0.0 (no penalty) |
| Invalid action | −0.5 + −5 health |
| Budget exhausted (actions unreliable) | rewards reduced 50% |

**Design rationale for `ignore` penalty:** The penalty fires only when a visible threat exists. Ignoring an empty observation is neutral — penalizing the agent for a step where there was nothing to do would create a degenerate incentive to scan constantly.

### Grader Score (Episode)

```python
# grader.py — authoritative formula
score = 0.50 * effective_containment
      + 0.20 * critical_health
      + 0.15 * resource_efficiency
      + 0.15 * speed_bonus
      − 0.10 * fp_penalty
```

Where:
- `effective_containment = 0.60 * raw_containment + 0.40 * criticality_weighted_containment`
- `critical_health = system_health / 100.0` at episode end
- `resource_efficiency = 1.0 - (budget_spent / budget_total)`
- `speed_bonus = mean(exp(-0.35 × age_at_containment))` across contained threats
- `fp_penalty = min(1.0, fp_actions / max(1, real_threats_seen))`

All scores pass through `safe_score(x) = min(0.9999, max(0.0001, x))` to enforce strict open-interval membership. This is a hard requirement of the OpenEnv validator.

### Formula Design Choices

**Why 50% weight on containment?** Containment is the primary objective. An agent that preserves health but fails to neutralize threats has failed its core mission.

**Why criticality-weighted containment?** Raw containment treats a contained edge node equally with a contained database server. The 60/40 blend forces agents to prioritize high-value nodes, which is the correct real-world behavior.

**Why exponential speed bonus instead of step thresholds?** Threshold functions create discontinuities (contained at step 4 = 0.5× reward of step 2, despite identical dwell time). Exponential decay (`exp(-0.35 × age)`) is smooth and differentiable, providing a meaningful gradient for learning-based agents at every containment time, not just at threshold crossings.

**Why FP penalty instead of no-action penalty?** Penalizing false-positive actions specifically (rather than all wrong actions) targets the exploit: an agent that mitigation-spams every visible threat without reasoning receives a proportional deduction. Maximum deduction is 0.10 — enough to matter at hard/elite decision boundaries without making a single FP action episode-ending.

---

## 5. Kill Chain Design

Threats follow a four-stage progression:

```
initial
  │  age >= escalation_threshold (default: 8 steps)
  ▼
escalated
  │  lateral spread attempted (if spread_rate > threshold)
  ▼
lateral_movement
  │  agent is alerted (threat becomes visible automatically)
  ▼
[contained by agent] or [episode ends with damage]
```

**Why four stages?** The chain creates a meaningful time window for the agent. A threat at `initial` is still hidden and low-urgency. At `escalated`, damage is increasing. At `lateral_movement`, it is visible and spreading — the agent is alerted but the threat has already gained ground. This mirrors real incident response, where early detection is rewarded and late response is penalized, but not catastrophically.

**Stage-specific IOC amplifiers:** Each stage applies multipliers to IOC signal values. A `lateral_movement`-stage threat has higher `lateral_connection_count` than the same threat at `initial`. This creates a natural signal: the threat's behavioral signature becomes louder as it escalates, providing a gradient for the agent to learn early detection.

**Spread mechanics:** When a threat reaches `lateral_movement`, it may spawn a child threat on an adjacent node. Spread is gated by `spread_rate` and `spread_attempted` flag (one spread attempt per threat per episode). The `monitor` action reduces `resurface_risk`, which also gates post-containment reactivation.

---

## 6. Ambiguity and Observability

### IOC Signal Design

The six IOC signals are intentionally designed with overlapping value ranges across threat types:

| Signal | High → suggests | But also elevated in |
|--------|-----------------|---------------------|
| `packets_per_second` | DDoS | Ransomware spread, worm propagation |
| `outbound_data_bytes` | Exfiltration | Legitimate backup traffic (FP) |
| `lateral_connection_count` | Lateral movement | Ransomware spread |
| `failed_auth_attempts` | Phishing, credential stuffing | System misconfiguration (FP) |
| `unusual_process_count` | Malware | Ransomware, insider threat |
| `spread_rate` | Ransomware | Worm, lateral movement |

No single signal uniquely identifies a threat type. An agent that keys on `spread_rate > 0.5 → isolate_machine` will be correct for ransomware but wrong for worm-style lateral movement (correct action: `block_ip`).

### False Positive Generation

False positives are spawned probabilistically alongside real threats. 40% of false positives have an elevated IOC value on one axis to mimic noisy alerts that "look real." Correct classification (do nothing, or `ignore`) costs nothing. Acting on an FP wastes budget and triggers the `fp_penalty`.

### Partial Observability

A threat is hidden by default. It becomes visible through one of three paths:

1. **Agent scans the node** (`scan_node_X`) — immediate reveal, costs scan budget
2. **Age threshold** — at `age >= age_visibility_threshold` (default: 5), probabilistic reveal gated by `base_detection_prob` and `false_negative_rate`
3. **Lateral movement stage** — automatic reveal when threat reaches this stage, but escalation damage has already occurred

On harder difficulties, `base_detection_prob` is reduced (0.4 on elite vs. 1.0 on easy), making natural discovery unreliable. This forces the agent to actively scan — but scanning costs budget. The tension between exploration and exploitation is the core challenge.

---

## 7. Determinism and RNG

### Seed Architecture

```
/reset seed=42
  │
  ├── session.rng = random.Random(42)
  │     └── used for: threat placement, IOC values, FP generation,
  │                   visibility rolls, spread events, resurface rolls
  │
  └── session.attacker = AdaptiveAttacker(seed = ATTACKER_SEED ^ (42 & 0xFFFFFFFF))
        └── used for: attacker strategy selection, exploration rolls
```

The attacker seed is XOR'd with the session seed so that different sessions with different seeds produce independent attacker behavior. Same seed → same attacker. Different seeds → different attackers.

### RNG Call Discipline

Every code path that may consume an RNG call must consume **exactly the same number** of RNG calls regardless of branching. This is enforced in `_update_visibility()`:

```python
def _update_visibility(sess):
    for t in sess.state["threats"]:
        if t.get("contained") or t.get("visible"):
            continue
        # Always consume exactly 2 RNG calls — even if the result is unused
        _roll1 = sess.rng.random()
        _roll2 = sess.rng.random()
        if t["stage"] == "lateral_movement":
            if _roll1 < detect_prob and _roll2 > fn_rate:
                t["visible"] = True
        elif t["age"] >= age_thresh:
            if _roll1 < detect_prob and _roll2 > fn_rate:
                t["visible"] = True
```

Violating this discipline breaks reproducibility for all subsequent steps. When adding new stochastic branches, always consume a fixed number of RNG calls before the branch, not inside it.

### What Determinism Guarantees

Same `(task, seed)` pair → identical:
- Threat types, nodes, IOC values
- Visibility reveal timing
- Attacker strategy for that episode
- False positive placement
- Spread events and resurface timing

It does **not** guarantee identical scores across different agent implementations — the agent's actions are external inputs and vary.

---

## 8. Testing Strategy

### Test Layers

| Suite | File | Coverage |
|-------|------|----------|
| HTTP API compliance | `tests/test_phase1.py` – `test_phase5.py` | All endpoints, all valid/invalid inputs |
| Integration (OOP layer) | `tests/test_phase6.py`, `test_phase7.py` | Task variants, baseline vs. ignore |
| Extended integration | `tests/test_tc106_200.py`, `test_tc201_300.py` | Episode lifecycle, determinism, memory |
| Regression (audit fixes) | `tests/test_audit.py` | 18 targeted tests for the 6 bugs fixed in audit |
| Attacker engine | `tests/test_adaptive_attacker.py` | Profile updates, counter-strategy selection |

### Regression Test Design (test_audit.py)

Each of the 6 audit bugs has at least one direct regression test:

| Bug | Test | What It Verifies |
|-----|------|-----------------|
| pending_action `break` → `continue` | `TestPendingActionContinue` | Second threat is targeted when first has pending_action |
| `ignore` penalty fires without visible threats | `TestIgnoreNoThreats` | No health penalty when no visible threats; penalty still fires with visible threat |
| `_update_visibility` consumed wrong RNG calls | `TestLateralMovementVisibility` | Elite empirical reveal rate ≤ 0.30; mock test confirms detect_prob applied |
| Attacker seed shared across sessions | `TestAttackerSeedPerSession` | Different session seeds → different attacker RNG; same seed → same attacker |
| `episode_actions_taken` appended after `_obs()` | `TestEpisodeActionsTakenOrder` | Resource reflects current action; actions list contains current step |
| `/analytics` scan cost wrong for hard task | `TestAnalyticsScanCost` | resources_remaining matches expected after 3 scans on hard; analytics and observe agree |

### Determinism Tests

```bash
# Verify same seed produces identical episodes
python -m pytest tests/test_audit.py::TestAttackerSeedPerSession -v
python -m pytest tests/test_tc106_200.py::TestPhase10Integration::test_tc154_determinism_across_10_episodes -v
```

### Running the Full Suite

```bash
# All tests
python -m pytest tests/ -v

# HTTP layer only (fast, no OOP layer)
python -m pytest tests/test_phase1.py tests/test_phase2.py tests/test_phase3.py \
                 tests/test_phase4.py tests/test_phase5.py tests/test_audit.py -v

# Audit regression tests only
python -m pytest tests/test_audit.py -v
```

---

## 9. Known Limitations

**OOP layer / HTTP layer divergence:** The OOP layer (`env.py`, `engines/`) has a pre-existing bug: `'Threat' object has no attribute 'age'` in `tasks/base.py:195`. This causes `test_phase6.py`, `test_phase7.py`, `test_tc106_200.py`, `test_tc201_300.py` to fail. The HTTP layer (`app.py`) is unaffected — all 293 HTTP-layer tests pass. The OOP bug is pre-existing and was not introduced by the audit.

**Single-process session store:** The `OrderedDict` session store has no mutex. Concurrent resets from multiple clients targeting the same `sid` can produce race conditions. Acceptable for evaluation use; not suitable for multi-tenant production.

**LRU eviction at 256 sessions:** Long-running evaluations with many concurrent sessions may evict active sessions. If a session is evicted mid-episode, the next `/step` will return a 404. Design for session recreation on 404.

**Score variance on elite:** The elite task has near-zero detection probability and pre-compromised nodes. Hidden threat dynamics introduce irreducible stochasticity. Two-episode averaging reduces but does not eliminate score variance. Expect ±0.05 variance on repeated runs with the same seed.

**Keyword-based reasoning extraction:** `AccuracyTracker` infers LLM threat classification from regex matches on reasoning text. Reasoning that uses synonyms, paraphrase, or indirect language may be missed.

---

## 10. Extension Guide

### Adding a New Attack Type

1. Add the new type to `THREAT_TYPES` in `app.py` and the corresponding MITRE mapping to `MITRE_MAP`.
2. Define IOC value ranges in the threat generation function. Ensure at least two IOC signals overlap with an existing type to maintain ambiguity.
3. Add the correct mitigation action to `CORRECT_ACTIONS`.
4. Update `openenv.yaml` to include the new type in the threat taxonomy.
5. Add test cases to `tests/test_audit.py` verifying correct action reward and wrong action penalty.

### Modifying the Reward Formula

The formula is in `grader.py:compute_grader_score()`. It is imported by both `app.py` and `inference.py`. Edit only `grader.py` — do not redefine the formula elsewhere.

After modifying:
1. Run `python grader.py` to verify all boundary cases still pass.
2. Update `TASK_PASSING_SCORES` if you change the score range to preserve tier calibration.
3. Run the full test suite — score-range tests will catch threshold violations.

### Adding a New Difficulty Tier

1. Create `tasks/new_tier.py` with a config dict following the existing pattern.
2. Register it in `TASK_CONFIGS` in `app.py`.
3. Add a passing score to `TASK_PASSING_SCORES` in `grader.py`. It must be strictly greater than the tier below it.
4. Add the tier to `openenv.yaml` under `tasks:`.
5. Add difficulty-scaled scan costs to `_SCAN_COST_BY_DIFFICULTY` and `_VERIFY_COST_BY_DIFFICULTY` in `app.py`.

### Adding a New Endpoint

All endpoints must:
- Return the full 9-key observation on every path (including errors), using `safe_response()`
- Never raise HTTP 5xx — wrap all logic in try/except
- Use the session store and respect LRU eviction
- Not modify `grader.py` or `constants.py`

### Modifying RNG Behavior

Any new stochastic branch must:
1. Consume a **fixed** number of calls from `sess.rng` regardless of the branch outcome
2. Consume those calls **before** the conditional logic, not inside it
3. Be documented in a comment with the exact call count

Violation breaks episode reproducibility. Test with `TestAttackerSeedPerSession` pattern.

---

## Appendix: Audit Summary

Six bugs were identified and fixed in the hardening audit (April 2026):

| ID | Severity | Description | Fix |
|----|----------|-------------|-----|
| BUG-1 | P0 | Defense loop `break` skipped all subsequent threats when one had `pending_action` | Changed `break` to `continue` |
| BUG-2 | P0 | `ignore` applied -10 health even with no visible threats | Gated penalty on `matched_threat_type is not None` |
| BUG-3 | P0 | `_update_visibility` consumed variable RNG calls, breaking determinism on lateral_movement | Unified to exactly 2 RNG calls per threat per step |
| BUG-4 | P1 | All sessions shared identical attacker seed, enabling cross-session attacker prediction | Attacker seed XOR'd with session seed |
| BUG-5 | P1 | `episode_actions_taken` appended after `_obs()`, so grader saw stale action history | Moved append before `_obs()` call |
| BUG-6 | P2 | `/analytics` used flat scan cost instead of difficulty-scaled cost | Added `_SCAN_COST_BY_DIFFICULTY` lookup to `/analytics` and `/observe` |

18 regression tests cover all six bugs. All 293 HTTP-layer tests pass post-fix.
