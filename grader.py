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
    """Return the authoritative episode quality score in [0.0, 1.0].

    Args:
        containment_rate:    Fraction of real threats eventually contained.
        critical_health:     system_health / 100.0 at episode end.
        resource_efficiency: Fraction of SOC budget not spent (1 - used/total).
        speed_bonus:         compute_speed_bonus() result.
    """
    return round(
        max(0.0, min(1.0,
            0.50 * containment_rate
            + 0.20 * critical_health
            + 0.15 * resource_efficiency
            + 0.15 * speed_bonus
        )),
        4,
    )
