"""
Adversarial Scenario Generator
================================
Generates extremely difficult, unpredictable episodes for robustness testing.

What it does
------------
1. Escalates all threats to late stage (lateral_movement + escalated=True)
2. Raises severity and forces immediate visibility
3. Clones and mutates threats until density reaches 4–6
4. Cross-contaminates IOC profiles to reduce signal clarity
5. Starts the network with reduced health (pre-existing damage)
6. Prepares a random mid-episode attacker strategy switch schedule
7. Forces a hard initial attacker strategy

What it does NOT do
-------------------
- Does NOT touch grader.py or the grader formula
- Does NOT change task thresholds or passing scores
- Does NOT modify any API response schema
- Does NOT affect normal (adversarial=False) sessions in any way

Usage
-----
    # In app.py reset() — only when req.adversarial is True
    from adversarial_generator import AdversarialScenarioGenerator
    gen = AdversarialScenarioGenerator()
    config = gen.generate(seed)
    gen.apply_to_session(sess, config)      # call after _do_reset_session()

    # In app.py step() — no-op for normal sessions
    gen.maybe_switch_strategy(sess)         # fires only if schedule exists

    # Standalone / test usage
    config = gen.generate(seed=42)
    # config is a plain dict — safe to log or serialise
"""

from __future__ import annotations

import random
from typing import Any


# ---------------------------------------------------------------------------
# Constants — must match values in app.py but imported separately
# to avoid circular imports.
# ---------------------------------------------------------------------------

ATTACKER_STRATEGIES: list[str] = [
    "APT",
    "RANSOMWARE",
    "INSIDER_THREAT",
    "SUPPLY_CHAIN",
    "ZERO_DAY",
]

# IOC fields that can be cross-contaminated between threat profiles
_IOC_KEYS: list[str] = [
    "packets_per_second",
    "failed_auth_attempts",
    "outbound_data_bytes",
    "lateral_connection_count",
    "unusual_process_count",
]

# Network nodes — must match NODES in app.py
_NODES: list[str] = [f"node_{i}" for i in range(1, 6)]


# ---------------------------------------------------------------------------
# AdversarialScenarioGenerator
# ---------------------------------------------------------------------------

class AdversarialScenarioGenerator:
    """
    Deterministic adversarial episode generator.

    Seed → config dict → session mutation.
    Two calls with the same seed produce identical adversarial episodes.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, seed: int) -> dict:
        """
        Produce a deterministic adversarial configuration for *seed*.

        Returns a plain dict — safe to log, serialise, and pass to
        apply_to_session().

        Fields
        ------
        seed                  : the input seed (echoed back for traceability)
        target_threat_count   : 4–6 — how many threats the episode will start with
        min_severity          : 0.75–0.90 — floor severity for all threats
        min_age               : 5–9 — minimum age (steps) of each threat at start
        ioc_noise_factor      : 0.30–0.55 — cross-contamination strength
        health_start          : 55–75 — initial system health (pre-damaged network)
        initial_strategy      : one of ATTACKER_STRATEGIES — hard starting strategy
        strategy_schedule     : list of {step, strategy, triggered} dicts
        """
        rng = random.Random(seed)
        return {
            "seed":                seed,
            "target_threat_count": rng.randint(4, 6),
            "min_severity":        round(rng.uniform(0.75, 0.90), 3),
            "min_age":             rng.randint(5, 9),
            "ioc_noise_factor":    round(rng.uniform(0.30, 0.55), 3),
            "health_start":        rng.randint(55, 75),
            "initial_strategy":    rng.choice(ATTACKER_STRATEGIES),
            "strategy_schedule":   self._build_strategy_schedule(rng),
        }

    def apply_to_session(self, sess: Any, config: dict) -> None:
        """
        Mutate *sess* in place to create an adversarial starting state.

        Must be called AFTER _do_reset_session() so that sess.state["threats"]
        and sess.attacker are already initialised.  Never touches sess.task_config
        or grader.py — the scoring formula is unchanged.

        Parameters
        ----------
        sess   : a live Session object
        config : dict returned by generate()
        """
        rng    = sess.rng
        state  = sess.state
        threats: list[dict] = state["threats"]

        min_sev   = config["min_severity"]
        min_age   = config["min_age"]
        noise     = config["ioc_noise_factor"]

        # ── 1. Escalate all existing threats to late stage ────────────────────
        for t in threats:
            t["stage"]              = "lateral_movement"
            t["escalated"]          = True
            t["age"]                = min_age + rng.randint(0, 3)
            t["severity"]           = min(0.99, round(min_sev + rng.uniform(0.0, 0.12), 3))
            t["visible"]            = True          # skip detection phase — immediately visible
            t["detection_confidence"] = round(rng.uniform(0.35, 0.65), 3)  # reduced certainty
            t["spread_rate"]        = round(rng.uniform(0.35, 0.65), 3)    # fast spread
            t["is_persistent"]      = True

        # ── 2. Clone threats up to target density (4–6) ──────────────────────
        needed = config["target_threat_count"] - len(threats)
        for i in range(max(0, needed)):
            if not threats:
                break
            base  = rng.choice(threats)
            clone = dict(base)   # shallow copy — safe for flat threat dicts
            clone["id"]               = f"adv_{base['id']}_{i}"
            clone["node"]             = rng.choice(_NODES)
            clone["contained"]        = False
            clone["spread_attempted"] = False
            clone["age"]              = min_age + rng.randint(0, 2)
            clone["severity"]         = min(0.99, round(min_sev + rng.uniform(0.0, 0.10), 3))
            # Jitter IOC values so clone is distinct from parent
            for key in _IOC_KEYS:
                val = clone.get(key, 0)
                clone[key] = max(0, int(val * (1.0 + rng.uniform(-noise, noise))))
            threats.append(clone)

        # ── 3. Cross-contaminate IOC profiles (reduce signal clarity) ─────────
        self._add_ioc_overlap(threats, rng, noise)

        # ── 4. Start network with pre-existing damage ─────────────────────────
        state["system_health"] = config["health_start"]

        # ── 5. Store strategy schedule for mid-episode attacker switches ──────
        #      Stored as a new Session attribute — never read by existing code.
        sess.adversarial_strategy_schedule = config["strategy_schedule"]

        # ── 6. Force a hard initial attacker strategy ─────────────────────────
        sess.attacker.current_strategy = config["initial_strategy"]

    def maybe_switch_strategy(self, sess: Any) -> bool:
        """
        Check the adversarial strategy schedule and switch the attacker strategy
        if the current step has crossed a scheduled threshold.

        Returns True if at least one switch occurred, False otherwise.
        This is a **no-op** for sessions that have no adversarial_strategy_schedule
        (i.e. all normal sessions — zero performance cost).

        Designed to be called once per step() after s["step"] is incremented.
        """
        schedule = getattr(sess, "adversarial_strategy_schedule", None)
        if not schedule:
            return False

        current_step = sess.state.get("step", 0)
        switched = False
        for entry in schedule:
            if not entry.get("triggered") and current_step >= entry["step"]:
                old_strategy               = sess.attacker.current_strategy
                sess.attacker.current_strategy = entry["strategy"]
                entry["triggered"]         = True
                switched                   = True
                print(
                    f"  [adversarial] step={current_step} "
                    f"strategy {old_strategy} → {entry['strategy']}"
                )
        return switched

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_strategy_schedule(self, rng: random.Random) -> list[dict]:
        """
        Create 1–2 strategy switches at distinct random steps in [3, 12].

        Returns a list of {step, strategy, triggered=False} dicts — one per switch.
        Triggered flag is set to True in-place by maybe_switch_strategy().
        """
        strategies = ATTACKER_STRATEGIES.copy()
        rng.shuffle(strategies)
        n_switches = rng.randint(1, 2)
        steps      = sorted(rng.sample(range(3, 13), n_switches))
        return [
            {"step": step, "strategy": strategies[i], "triggered": False}
            for i, step in enumerate(steps)
        ]

    def _add_ioc_overlap(
        self,
        threats: list[dict],
        rng:     random.Random,
        factor:  float,
    ) -> None:
        """
        Contaminate each threat's IOC profile with a fraction of a random
        donor threat's signals.  Preserves the dominant IOC pattern while
        adding noise that makes classification harder.

        Each field receives between 10% and *factor* × donor's value added
        on top of its own value.
        """
        if len(threats) < 2:
            return
        for t in threats:
            others = [x for x in threats if x is not t]
            if not others:
                continue
            donor = rng.choice(others)
            for key in _IOC_KEYS:
                t_val        = t.get(key, 0)
                d_val        = donor.get(key, 0)
                contamination = rng.uniform(0.10, factor)
                t[key]       = max(0, int(t_val + d_val * contamination))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

# Import and use directly:  from adversarial_generator import generator
generator = AdversarialScenarioGenerator()
