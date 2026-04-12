"""
Authoritative grader formula for the Adaptive Cyber Defense environment.

Single source of truth for:
  - safe_score            — enforces strict open interval (0.0, 1.0)
  - TASK_PASSING_SCORES   — monotonically increasing per-difficulty thresholds
  - compute_speed_bonus   — early-containment speed component
  - compute_grader_score  — weighted episode quality formula

Imported by both app.py (server) and inference.py (client) so the
formula can never silently diverge between evaluation paths.

Formula
-------
    score = 0.50 × containment_rate
          + 0.20 × critical_health
          + 0.15 × resource_efficiency
          + 0.15 × speed_bonus
          - 0.10 × fp_penalty           (optional; default 0.0)
    score = safe_score(score)   # strictly in (0.0, 1.0)

fp_penalty
----------
    fp_penalty = min(1.0, false_positive_actions / max(1, real_threats_seen))

    Rationale: penalises over-action on ghost alerts without catastrophically
    punishing early exploration.  Maximum deduction is 0.10 — enough to matter
    at the decision boundary (e.g. hard threshold 0.70) without making a single
    FP action a session-killer.  An agent that acts on zero false positives
    receives the full score from the four positive components.
"""

import math as _math

# ---------------------------------------------------------------------------
# Single source of truth: safe_score
# Imported by inference.py and app.py — do NOT redefine elsewhere.
# ---------------------------------------------------------------------------

def safe_score(x: float) -> float:
    """Clamp x to the strict open interval (0.0, 1.0).

    Never returns exactly 0.0 or 1.0 — required by Phase 2 validator.
    This is the ONLY place this function is defined; all other modules
    import it from here.
    """
    return min(0.9999, max(0.0001, float(x)))


# ---------------------------------------------------------------------------
# Passing thresholds — strictly monotonically increasing with difficulty
# ---------------------------------------------------------------------------

TASK_PASSING_SCORES: dict[str, float] = {
    "easy":       0.40,
    "medium":     0.55,
    "hard":       0.70,
    "nightmare":  0.80,
    "elite":      0.88,
}


# ---------------------------------------------------------------------------
# Canonical per-step reward specification
# Single source of truth for per-step signal values.
# Imported by app.py (HTTP path) so the reward scale is never silently
# diverged between the OOP training environment and the HTTP eval path.
#
# Scan/verify/monitor values are direct (not clamped).
# Mitigation values are raw pre-clamp inputs — pass through _clamp_reward()
# before returning: _clamp_reward(r) = clamp((r + 2.0) / 4.0, 0.001, 0.999)
# ---------------------------------------------------------------------------
STEP_REWARDS: dict[str, float] = {
    # ── Scan actions ─────────────────────────────────────────────────────
    "scan_real_threat":      +0.25,   # scan revealed a hidden real threat
    "scan_fp_revealed":      -0.05,   # scan revealed only a false positive
    "scan_clean":            -0.10,   # scan found nothing (clean node)
    "scan_contained":        -0.20,   # scan on a node with only contained threats
    "scan_exhausted":        -0.30,   # scan while resource budget is exhausted
    # ── Verify actions ───────────────────────────────────────────────────
    "verify_new_threat":     +0.15,   # verify confirmed an unverified visible threat
    "verify_wasted":         -0.08,   # verify on node with nothing new to confirm
    # ── Monitor actions ──────────────────────────────────────────────────
    "monitor_reduced_risk":  +0.10,   # monitor reduced resurface risk for a threat
    "monitor_wasted":        -0.08,   # monitor on node with nothing resurfaceable
    # ── Mitigation raw pre-clamp values ──────────────────────────────────
    "correct_early":         +1.10,   # correct action, threat age < 3
    "correct_instant":       +1.00,   # correct instant mitigation (block_ip/isolate)
    "correct_patch_queued":  +0.80,   # correct delayed patch action (still active)
    "wrong_action":          -0.50,   # wrong MITRE action for threat type
    "ignore_visible_threat": -1.50,   # ignore while at least one visible threat exists
}


# ---------------------------------------------------------------------------
# Speed bonus component
# ---------------------------------------------------------------------------

def compute_speed_bonus(containment_events: list, lambda_decay: float = 0.35) -> float:
    """Continuous exponential-decay speed bonus: score = exp(-λ × age_at_containment).

    Replaces the step-threshold formula (age<3→1.0, age<5→0.5, else→0.0) with a
    smooth exponential that rewards any early containment — even age=10 scores 0.03
    instead of 0.0.  λ=0.35 calibrated so age=2 ≈ 0.50 (same as old mid-threshold),
    age=0 = 1.0, age=5 ≈ 0.17.

    Research motivation: threshold functions create artificial discontinuities where
    an agent contained at step 4 receives half the reward of one contained at step 2
    despite a nearly identical dwell time.  Exponential decay is smooth and differentiable,
    enabling gradient-based policy optimisers to receive a meaningful learning signal
    for every improvement in containment speed — not just for crossing a threshold.
    """
    if not containment_events:
        return 0.0
    scores = [
        # Default 5: median expected containment age; avoids silently zeroing the
        # component (old default 99 → exp(-34.6) ≈ 0) when the field is missing.
        # age=0 (contained the step it was detected) maps to exp(0)=1.0 — intentional.
        round(_math.exp(-lambda_decay * ev.get("age_at_containment", 5)), 4)
        for ev in containment_events
    ]
    return round(sum(scores) / len(scores), 4)


def compute_criticality_weighted_containment(
    threats: list,
    node_criticality: dict,
) -> float:
    """Criticality-weighted containment rate.

    Standard containment_rate treats a contained edge-node threat equally with a
    contained hub-node threat.  This function weights each threat by the criticality
    of the node it occupies, so containing high-value nodes contributes more to the
    final score than containing low-value edge nodes.

    Args:
        threats:          List of raw threat dicts (internal format, not agent-facing).
                          Must have keys: 'is_false_positive', 'contained', 'node'.
        node_criticality: Dict mapping node_id → criticality ∈ [0, 1].

    Returns:
        Weighted containment rate ∈ [0, 1].  Returns 1.0 if no real threats exist.
    """
    real = [t for t in threats if not t.get("is_false_positive")]
    if not real:
        return 1.0
    total_w    = sum(node_criticality.get(t["node"], 0.6) for t in real)
    contained_w = sum(
        node_criticality.get(t["node"], 0.6) for t in real if t.get("contained")
    )
    return contained_w / total_w if total_w > 0 else 1.0


# ---------------------------------------------------------------------------
# Weighted episode quality formula
# ---------------------------------------------------------------------------

def compute_grader_score(
    containment_rate: float,
    critical_health: float,
    resource_efficiency: float,
    speed_bonus: float,
    fp_penalty: float = 0.0,
    criticality_weighted_containment: float | None = None,
) -> float:
    """Return the authoritative episode quality score strictly in (0.0, 1.0).

    Scores are clamped to (epsilon, 1-epsilon) so they are never exactly 0.0
    or 1.0, satisfying evaluators that require strict open-interval membership.
    A final double-clamp after rounding catches any float rounding edge case.

    Args:
        containment_rate:              Fraction of real threats eventually contained.
        critical_health:               system_health / 100.0 at episode end.
        resource_efficiency:           Fraction of SOC budget not spent (1 - used/total).
                                       Computed with get_scaled_costs() so that no
                                       MITRE-correct action is systematically penalised.
        speed_bonus:                   compute_speed_bonus() result (exponential decay).
        fp_penalty:                    False-positive action rate in [0.0, 1.0].
                                       = min(1, fp_actions / max(1, real_threats)).
                                       Default 0.0 for backward compatibility.
        criticality_weighted_containment:
                                       Optional node-criticality-weighted containment rate
                                       from compute_criticality_weighted_containment().
                                       When provided, the effective containment term is
                                       60 % raw + 40 % criticality-weighted — forcing agents
                                       to prioritise high-value nodes over easy targets.
                                       Default None → uses raw containment_rate only.
    """
    # 1e-4 = 0.0001, consistent with safe_score()'s boundary.
    # The prior value of 1e-6 was below round(..., 4) precision:
    #   round(1e-6, 4) = 0.0  →  clamp had no effect before safe_score rescued it.
    _EPSILON = 1e-4
    fp = min(1.0, max(0.0, float(fp_penalty)))
    # Criticality blending: prioritise hub/high-value node containment in scoring.
    if criticality_weighted_containment is not None:
        effective_containment = (
            0.60 * containment_rate
            + 0.40 * float(criticality_weighted_containment)
        )
    else:
        effective_containment = containment_rate
    raw = (
        0.50 * effective_containment
        + 0.20 * critical_health
        + 0.15 * resource_efficiency
        + 0.15 * speed_bonus
        - 0.10 * fp
    )
    # Clamp BEFORE rounding so float arithmetic never pushes raw to boundary
    clamped = max(_EPSILON, min(1.0 - _EPSILON, raw))
    rounded = round(clamped, 4)
    # safe_score is the final gate — catches any residual rounding edge case
    return safe_score(rounded)


# ---------------------------------------------------------------------------
# Test harness — run directly: python grader.py
# ---------------------------------------------------------------------------

def test_grader() -> list:
    """Verify all scores are strictly in (0, 1) across boundary and typical cases.

    Each tuple: (name, containment, health, resource, speed, fp_penalty)
    fp_penalty=0.0 cases verify backward compatibility; fp_penalty>0 cases
    verify the FP deduction is bounded and scores remain in (0, 1).
    """
    test_cases = [
        # (name, containment, health, resource, speed, fp_penalty)
        ("worst_case",      0.0, 0.0, 0.0, 0.0, 0.0),
        ("perfect_no_fp",   1.0, 1.0, 1.0, 1.0, 0.0),
        ("perfect_max_fp",  1.0, 1.0, 1.0, 1.0, 1.0),   # deducts 0.10
        ("mid_no_fp",       0.5, 0.5, 0.5, 0.5, 0.0),
        ("mid_half_fp",     0.5, 0.5, 0.5, 0.5, 0.5),   # deducts 0.05
        ("high_no_fp",      0.9, 0.9, 0.9, 0.9, 0.0),
        ("low_no_fp",       0.1, 0.1, 0.1, 0.1, 0.0),
        ("hard_no_fp",      0.8, 0.8, 0.0, 0.5, 0.0),   # resource_efficiency=0
        ("hard_with_fp",    0.8, 0.8, 0.0, 0.5, 0.2),   # resource=0 + some FPs
    ]
    results = []
    for name, c, h, r, s, fp in test_cases:
        score = compute_grader_score(c, h, r, s, fp)
        results.append((name, c, h, r, s, fp, score))
    return results


def _simulate_tasks() -> list:
    """Simulate realistic task scores for easy → elite."""
    tasks = [
        ("easy",      0.95, 0.95, 0.90, 0.80),
        ("medium",    0.90, 0.90, 0.85, 0.75),
        ("hard",      0.80, 0.80, 0.75, 0.60),
        ("nightmare", 0.50, 0.55, 0.60, 0.30),
        ("elite",     0.85, 0.80, 0.70, 0.65),
    ]
    return [(name, compute_grader_score(c, h, r, s)) for name, c, h, r, s in tasks]


if __name__ == "__main__":
    # ── Test case results ────────────────────────────────────────────────────
    print("\nTEST CASE RESULTS")
    print("-" * 80)
    print(f"{'Name':<18} {'Contain':>8} {'Health':>8} {'Resource':>10} {'Speed':>7} {'FP':>6} {'Score':>8}")
    print("-" * 80)
    for name, c, h, r, s, fp, score in test_grader():
        print(f"{name:<18} {c:>8.3f} {h:>8.3f} {r:>10.3f} {s:>7.3f} {fp:>6.2f} {score:>8.4f}")

    # ── Task simulation results ──────────────────────────────────────────────
    print("\nTASK SIMULATION RESULTS")
    print("-" * 40)
    print(f"{'Task':<12} {'Score':>8}")
    print("-" * 40)
    task_results = _simulate_tasks()
    for name, score in task_results:
        print(f"{name:<12} {score:>8.4f}")

    # ── Validation ──────────────────────────────────────────────────────────
    print("\nVALIDATION")
    all_scores = [s for _, _, _, _, _, _, s in test_grader()] + [s for _, s in task_results]
    errors = [s for s in all_scores if s <= 0.0 or s >= 1.0]
    if errors:
        for s in errors:
            print(f"  ERROR: score {s} is out of strict (0, 1) range")
        print("RESULT: FAIL")
    else:
        print(f"  All {len(all_scores)} scores strictly in (0, 1) — no boundary violations")
        print("RESULT: PASS")
