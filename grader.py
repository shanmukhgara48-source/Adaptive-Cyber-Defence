"""
Authoritative grader formula for the Adaptive Cyber Defense environment.

Single source of truth for:
  - TASK_PASSING_SCORES  — monotonically increasing per-difficulty thresholds
  - compute_speed_bonus  — early-containment speed component
  - compute_grader_score — weighted episode quality formula

Imported by both app.py (server) and inference.py (client) so the
formula can never silently diverge between evaluation paths.

Formula
-------
    score = 0.50 × containment_rate
          + 0.20 × critical_health
          + 0.15 × resource_efficiency
          + 0.15 × speed_bonus
    score = clamp(score, 0.0, 1.0)
"""

# ---------------------------------------------------------------------------
# Passing thresholds — strictly monotonically increasing with difficulty
# ---------------------------------------------------------------------------

TASK_PASSING_SCORES: dict[str, float] = {
    "easy":       0.40,
    "medium":     0.55,
    "hard":       0.70,
    "nightmare":  0.80,
    "elite":      0.88,
    "impossible": 0.0,   # ceiling benchmark — no passing threshold
}


# ---------------------------------------------------------------------------
# Speed bonus component
# ---------------------------------------------------------------------------

def compute_speed_bonus(containment_events: list) -> float:
    """Mean early-containment score across all contained threats.

    Per-threat score:
      age_at_containment < 3  → 1.0  (neutralised before escalation)
      age_at_containment < 5  → 0.5  (contained while still fresh)
      otherwise               → 0.0  (dwell too long)
    """
    if not containment_events:
        return 0.0
    scores = []
    for ev in containment_events:
        age = ev.get("age_at_containment", 99)
        if age < 3:
            scores.append(1.0)
        elif age < 5:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return round(sum(scores) / len(scores), 4)


# ---------------------------------------------------------------------------
# Weighted episode quality formula
# ---------------------------------------------------------------------------

def compute_grader_score(
    containment_rate: float,
    critical_health: float,
    resource_efficiency: float,
    speed_bonus: float,
) -> float:
    """Return the authoritative episode quality score strictly in (0.0, 1.0).

    Scores are clamped to (epsilon, 1-epsilon) so they are never exactly 0.0
    or 1.0, satisfying evaluators that require strict open-interval membership.
    A final double-clamp after rounding catches any float rounding edge case.

    Args:
        containment_rate:    Fraction of real threats eventually contained.
        critical_health:     system_health / 100.0 at episode end.
        resource_efficiency: Fraction of SOC budget not spent (1 - used/total).
        speed_bonus:         compute_speed_bonus() result.
    """
    _EPSILON = 1e-6
    raw = (
        0.50 * containment_rate
        + 0.20 * critical_health
        + 0.15 * resource_efficiency
        + 0.15 * speed_bonus
    )
    # Clamp BEFORE rounding to avoid rounding pushing to boundary
    clamped = max(_EPSILON, min(1.0 - _EPSILON, raw))
    rounded = round(clamped, 4)
    # Final safety clamp: catches any float rounding edge case after round()
    return min(0.9999, max(0.0001, rounded))


# ---------------------------------------------------------------------------
# Test harness — run directly: python grader.py
# ---------------------------------------------------------------------------

def test_grader() -> list:
    """Verify all scores are strictly in (0, 1) across boundary and typical cases."""
    test_cases = [
        ("worst_case",   0.0, 0.0, 0.0, 0.0),
        ("perfect_case", 1.0, 1.0, 1.0, 1.0),
        ("mid_case",     0.5, 0.5, 0.5, 0.5),
        ("high_case",    0.9, 0.9, 0.9, 0.9),
        ("low_case",     0.1, 0.1, 0.1, 0.1),
    ]
    results = []
    for name, c, h, r, s in test_cases:
        score = compute_grader_score(c, h, r, s)
        results.append((name, c, h, r, s, score))
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
    print("-" * 68)
    print(f"{'Name':<14} {'Contain':>8} {'Health':>8} {'Resource':>10} {'Speed':>7} {'Score':>8}")
    print("-" * 68)
    for name, c, h, r, s, score in test_grader():
        print(f"{name:<14} {c:>8.3f} {h:>8.3f} {r:>10.3f} {s:>7.3f} {score:>8.4f}")

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
    all_scores = [s for _, _, _, _, _, s in test_grader()] + [s for _, s in task_results]
    errors = [s for s in all_scores if s <= 0.0 or s >= 1.0]
    if errors:
        for s in errors:
            print(f"  ERROR: score {s} is out of strict (0, 1) range")
        print("RESULT: FAIL")
    else:
        print(f"  All {len(all_scores)} scores strictly in (0, 1) — no boundary violations")
        print("RESULT: PASS")
