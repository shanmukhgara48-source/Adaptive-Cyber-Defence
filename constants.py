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
