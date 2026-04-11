# ---------------------------------------------------------------------------
# Shared constants — single source of truth for action costs.
# Import this dict wherever action-resource costs are needed so that any
# future change only needs to happen in one place.
# ---------------------------------------------------------------------------

# Raw resource cost of each mitigation action.
# Scan actions default to 0.2 via the fallback in callers; ignore costs 0.0.
COST_RAW: dict[str, float] = {
    "isolate_machine": 0.4,
    "block_ip":        0.3,
    "patch":           0.3,
}


def get_scaled_costs(resource_per_step: float) -> dict[str, float]:
    """Return action costs scaled so no single action exceeds resource_per_step.

    On tasks where resource_per_step >= max(COST_RAW) (e.g. easy: 1.5), the
    raw costs are returned unchanged.

    On budget-constrained tasks (e.g. hard: resource_per_step=0.30), the most
    expensive action (isolate_machine=0.40) would exceed the per-step allowance,
    systematically penalising the MITRE-correct mitigation for malware/ransomware.
    Scaling all costs by (resource_per_step / max_cost) removes that bias while
    preserving relative cost ratios between actions.

    Used by the grader and observation layer.  The step-level _resource_exhausted
    check deliberately retains COST_RAW so gameplay mechanics are unchanged.
    """
    max_cost = max(COST_RAW.values())   # 0.40 (isolate_machine)
    if max_cost <= resource_per_step:
        return dict(COST_RAW)
    scale = resource_per_step / max_cost
    return {k: round(v * scale, 4) for k, v in COST_RAW.items()}
