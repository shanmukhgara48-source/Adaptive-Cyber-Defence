# Developer Reference — Adaptive Cyber Defense

Internal architecture and design notes for contributors.

---

## Repository Layout

```
adaptive_cyber_defense/
├── app.py                  # Primary hardened API 
├── server/app.py           # Legacy simple API 
├── Dockerfile              # Docker build (port 7860)
├── requirements.txt        # Runtime dependencies
├── pyproject.toml          # OpenEnv metadata + entrypoint declaration
├── openenv.yaml            # OpenEnv task definitions + observation/action schema
├── test_runner.py          # Adversarial stress test suite (400+ cases)
├── environment.py          # OpenEnv-compliant Python wrapper 
├── env.py                  # Core simulation engine 
├── engines/
│   ├── attack.py           # Kill-chain progression
│   ├── adaptive_attacker.py # Red-team profiling
│   ├── detection.py        # Probabilistic detection
│   ├── decision.py         # AI recommendation engine
│   ├── reward.py           # Reward computation
│   └── scoring.py          # Score aggregation
├── models/
│   ├── state.py            # State dataclass
│   ├── threat.py           # Threat dataclass + MITRE mapping
│   ├── action.py           # Action enum
│   └── network.py          # Network node model
├── agents/
│   ├── baseline.py         # Rule-based heuristic agent
│   ├── ignore.py           # Always-ignore baseline
│   └── ql_agent.py         # Q-learning agent
├── tasks/
│   ├── easy.py / medium.py / hard.py  # Task configs
│   └── base.py             # Base task class
├── training/               # Training scripts + result plots
├── inference.py            # LLM agent runner 
├── run.py                  # CLI episode runner
└── ui.py                   # Streamlit SOC dashboard
```

---

## Core API — `app.py`

### State

```python
state = {
    "threats":        list[dict],   # All threats (visible + hidden)
    "scanned_nodes":  set[str],     # Nodes the agent has scanned
    "system_health":  int,          # 0–100
    "score":          float,        # Cumulative reward
    "step":           int,          # Steps taken this episode
    "done":           bool,         # Episode terminal flag
}
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `safe_response(obs, action, reward, error)` | Builds the complete 9-key response dict. **All `/step` returns go through this.** |
| `_validate_state()` | Checks state integrity; calls `_reset_state()` if corrupted |
| `_clamp_reward(r)` | Pins reward to `[-2.0, 2.0]`, converts NaN/Inf to 0.0 |
| `_clamp_score()` | Resets score to 0.0 if non-finite |
| `_clamp_health()` | Pins health to `[0, 100]` |
| `_obs()` | Builds observation dict from current state |
| `_age_threats()` | Advances threat age each step; triggers stage escalation at age ≥ 8 |
| `_update_visibility()` | Makes threats visible if node scanned, age ≥ 5, or in lateral_movement |

### Action Whitelist

```python
VALID_ACTIONS = {"block_ip", "isolate_machine", "patch", "ignore",
                 "scan_node_1", "scan_node_2", "scan_node_3", "scan_node_4", "scan_node_5"}
```

Any action not in this set receives a `−0.5` reward penalty and `−5` health penalty. The full observation is still returned.

### Reward Table

| Condition | Reward |
|-----------|--------|
| Correct mitigation (matched threat) | `+1.0` |
| Scan reveals hidden threat | `+0.02` |
| Scan finds nothing new | `−0.01` |
| Wrong mitigation (unmatched) | `−0.5` |
| `ignore` action | `−1.0` |
| Invalid / unknown action | `−0.5` |

All rewards clamped to `[-2.0, 2.0]`.

### Episode Termination

Episode ends (`done = True`) when:
- `system_health <= 0`, or
- `step >= 50`

After `done = True`, `/step` returns the current obs with `error: "Episode over — call /reset"` rather than advancing state.

---

## Partial Observability

- Each episode spawns 3 threats on random nodes, all `visible = False`
- A threat becomes visible when:
  - Its node has been scanned (`scan_node_X`)
  - Its `age >= 5` (natural discovery)
  - Its `stage == "lateral_movement"` (age ≥ 8)
- `hidden_node_count` = `TOTAL_NODES − len(scanned_nodes)`
- `scan_coverage` = `len(scanned_nodes) / TOTAL_NODES`

---

## Input Validation

### Pydantic coercion (not rejection)

The `StepRequest` validator **coerces** non-string types to `str` and truncates strings over 64 chars. This ensures the request always reaches the endpoint body — so `safe_response` with full obs can always be returned.

```python
@field_validator("action", mode="before")
@classmethod
def coerce_action(cls, v):
    if not isinstance(v, str):
        v = str(v)
    if len(v) > MAX_ACTION_LEN:
        v = v[:MAX_ACTION_LEN]
    return v
```

### Exception handlers

Both `RequestValidationError` and the generic `Exception` handler return `safe_response` — never a bare `{"error": ...}` dict.

---

## Safety Layer

```
Request arrives
    │
    ▼
Pydantic coerces action to string, truncates if > 64 chars
    │
    ▼
/step body: try/except wraps ALL logic
    │
    ├─ done=True → safe_response(..., error="Episode over")
    │
    ├─ action not in VALID_ACTIONS → penalty + safe_response(..., error="invalid action")
    │
    ├─ valid action → execute game logic → safe_response(...)
    │
    └─ any exception → _validate_state() / auto-reset → safe_response(..., error="invalid action")
```

Every path returns all 9 keys: `action`, `reward`, `visible_threats`, `hidden_node_count`, `scan_coverage`, `system_health`, `score`, `step`, `done`.

---

## Self-Healing

`_validate_state()` is called at the start of `/state` and `/step`. It checks:
- All keys present and correct types
- `system_health` and `score` are finite floats

If any assertion fails, `_reset_state()` is called immediately before continuing.

---

## Testing

### Run the stress suite

```bash
# Start server first
uvicorn app:app --host 127.0.0.1 --port 8000 &

# Run all 400+ tests
python test_runner.py
```

### Test categories

| Category | Cases | What it breaks |
|----------|-------|----------------|
| A — Invalid Actions | ~100 | Empty, null, unicode, injections, long strings, fuzz |
| B — API Misuse | ~30 | Wrong HTTP methods, bad JSON, wrong types, large payloads |
| C — State Breakers | ~20 | Step after done, repeated reset, health underflow |
| D — Partial Observability | ~15 | Scan coverage consistency, hidden count math |
| E — Reward Edge Cases | ~15 | NaN/Inf reward, reward presence on every call |
| F — Multi-Threat Chaos | ~15 | 120 rapid sequential actions, alternating valid/invalid |
| G — Performance Stress | ~15 | 200 rapid steps, 50 resets, 10 concurrent resets |
| H — Security | ~50 | SQLi, XSS, path traversal, shell injection, unicode attacks |
| Z — Endpoint Health | ~15 | All required keys on all endpoints |

### Pass criteria

- No HTTP 5xx responses
- All responses contain the required observation keys
- `system_health` always in `[0, 100]`
- `reward` and `score` always finite

---

## Design Decisions

**Never crash** — the API is designed to absorb any input without a 500. Unknown actions receive a penalty instead of a rejection so the agent always gets a training signal.

**Coerce, don't reject** — Pydantic coerces bad types to strings rather than returning 422, which would omit the observation keys expected by test runners.

**Deterministic clamping** — all numerical outputs are explicitly clamped/rounded so no NaN or Inf values propagate into agent training.

**Whitelist over blacklist** — valid actions are an explicit frozenset. Anything outside it is handled uniformly rather than with pattern-matching heuristics.

---

## Known Limits

- Single-process in-memory state — concurrent resets can produce race conditions (no mutex; acceptable for evaluation use)
- Simple 3-threat simulation — not a full SOC; kill-chain depth is limited
- No real network — all nodes and threats are simulated
- No persistent storage — state resets on server restart
