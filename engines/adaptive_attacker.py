"""
engines/adaptive_attacker.py — Adaptive red-team attacker.

Observes what the defender does across episodes AND within a single episode,
then switches attack strategy to exploit the defender's most predictable weakness.

Two adaptation layers:
  1. Cross-episode (UCB1 bandit)   — selects the strategy with the best
     estimated reward across all historical episodes, with exploration bonus.
  2. Mid-episode (sliding window)  — reacts within 10 steps if the defender
     settles into a detectable micro-pattern (e.g. "always scans after seeing
     a lateral movement event").

Counter-strategy table:
  ISOLATOR  defender  → INSIDER_THREAT   (bypasses isolation via valid creds)
  BLOCKER   defender  → SUPPLY_CHAIN     (no external IP to block)
  SCANNER   defender  → APT              (low-and-slow, stays under scan threshold)
  PATCHER   defender  → ZERO_DAY         (exploit is unpatched by definition)
  BALANCED  defender  → RANSOMWARE       (time pressure breaks balanced play)
"""

from __future__ import annotations

import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WINDOW_SIZE   = 10      # recent-action sliding window length
# _MIN_EPISODES = 0: UCB1 already handles warm-up via the unpulled-arm branch,
# which pulls each arm exactly once before exploitation begins.  The old value
# of 2 forced PHISHING for 2 episodes, creating a biased prior that contaminated
# the bandit's first exploitation phase.
_MIN_EPISODES  = 0
_UCB_C         = 1.4142  # UCB1 exploration constant (√2)


# ---------------------------------------------------------------------------
# Action name normalisation
# Maps caller-side variants → canonical HTTP-API key used by _recalculate_rates().
# Handles both OOP enum names (BLOCK_IP) and HTTP API names (block_ip) so the
# attacker profiler works regardless of which layer calls observe_defender_action().
# ---------------------------------------------------------------------------

_ACTION_NORM: Dict[str, str] = {
    # OOP enum names (Action.X.name) → HTTP API keys
    "block_ip":            "block_ip",
    "BLOCK_IP":            "block_ip",
    "isolate_node":        "isolate_machine",
    "ISOLATE_NODE":        "isolate_machine",
    "isolate_machine":     "isolate_machine",
    "ISOLATE_MACHINE":     "isolate_machine",
    "patch_system":        "patch",
    "PATCH_SYSTEM":        "patch",
    "patch":               "patch",
    "PATCH":               "patch",
    "patch_vulnerability": "patch",
    "PATCH_VULNERABILITY": "patch",
    "run_deep_scan":       "scan_node",
    "RUN_DEEP_SCAN":       "scan_node",
    "scan_node":           "scan_node",
    "SCAN_NODE":           "scan_node",
    "scan":                "scan_node",
    "SCAN":                "scan_node",
    "ignore":              "ignore",
    "IGNORE":              "ignore",
}


# ---------------------------------------------------------------------------
# Defender behaviour profile
# ---------------------------------------------------------------------------

@dataclass
class DefenderBehaviorProfile:
    """Tracks what the defender does — attacker exploits the dominant pattern."""

    # Cumulative action counts across all observed steps
    action_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    # Per-threat-type action breakdown (for future fine-grained countering)
    actions_per_threat_type: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )

    # Sliding window of the last _WINDOW_SIZE actions (most recent = right)
    recent_window: deque = field(
        default_factory=lambda: deque(maxlen=_WINDOW_SIZE)
    )

    # Aggregate rate fields (lazily recomputed)
    isolation_rate: float = 0.0
    block_rate:     float = 0.0
    scan_rate:      float = 0.0
    patch_rate:     float = 0.0
    steps_observed: int   = 0
    episodes_observed: int = 0
    rates_dirty: bool = field(default=True, repr=False)

    def _recalculate_rates(self) -> None:
        if not self.rates_dirty:
            return
        total = max(1, self.steps_observed)
        self.isolation_rate = self.action_counts.get("isolate_machine", 0) / total
        self.block_rate     = self.action_counts.get("block_ip",        0) / total
        self.scan_rate      = sum(
            v for k, v in self.action_counts.items() if k.startswith("scan_")
        ) / total
        self.patch_rate     = self.action_counts.get("patch",           0) / total
        self.rates_dirty   = False

    def record_action(self, action_name: str, threat_type: str = "UNKNOWN") -> None:
        # Normalise to HTTP-API canonical key before recording
        canonical = _ACTION_NORM.get(action_name, action_name.lower())
        self.action_counts[canonical] += 1
        self.actions_per_threat_type[threat_type][canonical] += 1
        self.recent_window.append(canonical)
        self.steps_observed += 1
        self.rates_dirty = True

    def get_most_used_action(self) -> str:
        if not self.action_counts:
            return "UNKNOWN"
        return max(self.action_counts, key=self.action_counts.get)

    def get_window_dominant_action(self) -> Optional[str]:
        """
        Return the action that fills >50% of the recent window.
        Returns None when the window is too mixed to exploit.
        """
        if len(self.recent_window) < _WINDOW_SIZE // 2:
            return None
        counts: Dict[str, int] = defaultdict(int)
        for a in self.recent_window:
            counts[a] += 1
        top_action, top_count = max(counts.items(), key=lambda x: x[1])
        if top_count / len(self.recent_window) > 0.50:
            return top_action
        return None

    def get_defender_strategy_label(self) -> str:
        self._recalculate_rates()
        if self.isolation_rate > 0.4:
            return "ISOLATOR"
        elif self.block_rate > 0.4:
            return "BLOCKER"
        elif self.scan_rate > 0.4:
            return "SCANNER"
        elif self.patch_rate > 0.3:
            return "PATCHER"
        else:
            return "BALANCED"

    def get_window_strategy_label(self) -> Optional[str]:
        """
        Same logic as get_defender_strategy_label() but applied only to the
        recent window — enables mid-episode adaptation.
        Returns None when the window contains too few actions.
        """
        if len(self.recent_window) < _WINDOW_SIZE // 2:
            return None
        counts: Dict[str, int] = defaultdict(int)
        for a in self.recent_window:
            counts[a] += 1
        total = len(self.recent_window)
        iso  = counts.get("isolate_machine", 0) / total
        blk  = counts.get("block_ip",        0) / total
        scn  = sum(v for k, v in counts.items() if k.startswith("scan_")) / total
        pat  = counts.get("patch",            0) / total
        if iso > 0.4:  return "ISOLATOR"
        if blk > 0.4:  return "BLOCKER"
        if scn > 0.4:  return "SCANNER"
        if pat > 0.3:  return "PATCHER"
        return None   # window is too mixed → no mid-episode switch


# ---------------------------------------------------------------------------
# UCB1 bandit — strategy selector
# ---------------------------------------------------------------------------

class UCB1Bandit:
    """
    Upper-Confidence-Bound bandit over the attacker's available strategies.

    Each strategy is an arm.  Reward is the negative of the defender's grader
    score (attacker wins when defender score is low).

    UCB1 index: Q(arm) + C × sqrt(ln(total_pulls) / pulls(arm))
    """

    def __init__(self, arms: List[str], c: float = _UCB_C) -> None:
        self.arms     = arms
        self.c        = c
        self.counts   = {a: 0   for a in arms}   # pulls per arm
        self.values   = {a: 0.0 for a in arms}   # mean reward per arm
        self.total    = 0

    def select(self, rng: random.Random) -> str:
        """Return arm with highest UCB1 index; break ties randomly."""
        # Pull each arm once before exploitation
        unpulled = [a for a in self.arms if self.counts[a] == 0]
        if unpulled:
            return rng.choice(unpulled)

        ln_total = math.log(self.total)
        scores = {
            a: self.values[a] + self.c * math.sqrt(ln_total / self.counts[a])
            for a in self.arms
        }
        best_score = max(scores.values())
        best_arms  = [a for a, s in scores.items() if s == best_score]
        return rng.choice(best_arms)

    def update(self, arm: str, reward: float) -> None:
        """Incremental mean update: Q(arm) ← Q(arm) + (r - Q(arm)) / n."""
        self.counts[arm] += 1
        self.total       += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

    def summary(self) -> Dict[str, dict]:
        return {
            a: {"pulls": self.counts[a], "mean_reward": round(self.values[a], 4)}
            for a in self.arms
        }


# ---------------------------------------------------------------------------
# Adaptive attacker
# ---------------------------------------------------------------------------

class AdaptiveAttacker:
    """
    Red-team attacker that learns the defender's strategy and counters it.

    Two adaptation layers
    ---------------------
    1. Cross-episode UCB1 bandit — after _MIN_EPISODES warm-up, selects the
       strategy arm with the highest estimated attacker reward (= 1 - defender
       score), balancing exploitation with exploration.

    2. Mid-episode sliding window — every step, inspect the last _WINDOW_SIZE
       actions.  If the defender has settled into a dominant pattern not
       countered by the current strategy, fire a mid-episode switch.

    Lifecycle::

        attacker = AdaptiveAttacker(seed=42)

        for episode in range(N):
            plan = attacker.on_episode_start()
            # ... run episode, each step:
            attacker.observe_defender_action(action_name)
            mid = attacker.check_mid_episode_adaptation()
            if mid["switched"]:
                # apply mid["new_strategy"] override to the AttackEngine
            # ... end of episode:
            attacker.on_episode_end(defender_won, score)

        print(attacker.get_full_adaptation_report())
    """

    COUNTER_STRATEGY: Dict[str, str] = {
        "ISOLATOR": "INSIDER_THREAT",
        "BLOCKER":  "SUPPLY_CHAIN",
        "SCANNER":  "APT",
        "PATCHER":  "ZERO_DAY",
        "BALANCED": "RANSOMWARE",
    }

    COUNTER_REASONING: Dict[str, str] = {
        "ISOLATOR": (
            "Defender relies on isolate_machine. Switching to INSIDER THREAT — "
            "insider uses valid credentials, bypassing node isolation."
        ),
        "BLOCKER": (
            "Defender relies on block_ip. Switching to SUPPLY CHAIN — "
            "attack originates from a trusted internal service; no external IP to block."
        ),
        "SCANNER": (
            "Defender relies on scanning. Switching to APT low-and-slow — "
            "staying below scan detection threshold, advancing only every 5+ steps."
        ),
        "PATCHER": (
            "Defender relies on patch. Switching to ZERO DAY — "
            "exploits an unknown vulnerability; patch is not yet available."
        ),
        "BALANCED": (
            "Defender uses a balanced strategy. Switching to RANSOMWARE — "
            "time pressure forces defender to abandon balance and react urgently."
        ),
    }

    # All available strategies (bandit arms)
    ALL_STRATEGIES: List[str] = [
        "PHISHING", "APT", "RANSOMWARE",
        "INSIDER_THREAT", "SUPPLY_CHAIN", "ZERO_DAY",
    ]

    # Attack engine config overrides per strategy
    ATTACK_CONFIGS: Dict[str, Dict] = {
        "PHISHING": {
            "initial_attack_type":    "PHISHING",
            "dwell_time_multiplier":  1.0,
            "detection_evasion":      0.0,
            "spread_rate":            1.0,
        },
        "APT": {
            "initial_attack_type":    "PHISHING",
            "dwell_time_multiplier":  3.0,
            "detection_evasion":      0.6,
            "spread_rate":            0.3,
        },
        "RANSOMWARE": {
            "initial_attack_type":    "MALWARE",
            "dwell_time_multiplier":  0.3,
            "detection_evasion":      0.0,
            "spread_rate":            3.0,
        },
        "INSIDER_THREAT": {
            "initial_attack_type":    "LATERAL_MOVEMENT",   # was "lateral_movement" — engine expects uppercase
            "dwell_time_multiplier":  1.5,
            "detection_evasion":      0.7,
            "spread_rate":            0.8,
        },
        "SUPPLY_CHAIN": {
            "initial_attack_type":    "MALWARE",
            "dwell_time_multiplier":  2.0,
            "detection_evasion":      0.5,
            "spread_rate":            2.0,
        },
        "ZERO_DAY": {
            "initial_attack_type":    "MALWARE",
            "dwell_time_multiplier":  1.0,
            "detection_evasion":      0.8,
            "spread_rate":            1.5,
        },
    }

    def __init__(self, seed: int = 42, learning_rate: float = 0.3) -> None:
        self.rng              = random.Random(seed)
        self.learning_rate    = learning_rate   # kept for backward compat
        self.defender_profile = DefenderBehaviorProfile()
        self.current_strategy = "PHISHING"
        self.strategy_history: List[dict] = []
        self.episode_count:    int         = 0
        self.adaptation_log:   List[str]  = []

        # UCB1 bandit — learns which strategy reliably hurts the defender
        self._bandit = UCB1Bandit(arms=self.ALL_STRATEGIES)

        # Mid-episode state
        self._mid_episode_switches:   int           = 0         # guard: max 1 switch per episode
        self._episode_start_strategy: str           = "PHISHING"
        self._mid_episode_strategy:   Optional[str] = None      # tracks which strategy a mid-ep switch chose

    # -----------------------------------------------------------------------
    # Per-step observation
    # -----------------------------------------------------------------------

    def observe_defender_action(
        self,
        action_name: str,
        active_threat_type: str = "UNKNOWN",
    ) -> None:
        """Call every step with the action the defender just took."""
        self.defender_profile.record_action(action_name, active_threat_type)

    # -----------------------------------------------------------------------
    # Mid-episode adaptation
    # -----------------------------------------------------------------------

    def check_mid_episode_adaptation(self) -> dict:
        """
        Inspect the sliding window.  If the defender has shifted into a
        dominant pattern that is not already countered by the current
        strategy, switch strategies mid-episode (max once per episode).

        Returns a dict:
            {"switched": bool, "new_strategy": str | None, "reason": str}

        The caller should apply `get_attack_config_override(new_strategy)` to
        the AttackEngine when switched=True.
        """
        if self._mid_episode_switches > 0:
            return {"switched": False, "new_strategy": None, "reason": "already switched this episode"}

        window_label = self.defender_profile.get_window_strategy_label()
        if window_label is None:
            return {"switched": False, "new_strategy": None, "reason": "window too mixed"}

        ideal_counter = self.COUNTER_STRATEGY[window_label]
        if ideal_counter == self.current_strategy:
            return {"switched": False, "new_strategy": None, "reason": "already optimal"}

        # Switch!
        old = self.current_strategy
        self.current_strategy       = ideal_counter
        self._mid_episode_strategy  = ideal_counter   # record for UCB1 credit split in on_episode_end
        self._mid_episode_switches += 1
        reason = (
            f"Mid-episode switch: window shows {window_label} pattern → "
            f"countering with {ideal_counter} (was {old})"
        )
        self.adaptation_log.append(
            f"Episode {self.episode_count + 1} [MID]: {reason}"
        )
        return {"switched": True, "new_strategy": ideal_counter, "reason": reason}

    # -----------------------------------------------------------------------
    # Episode lifecycle
    # -----------------------------------------------------------------------

    def choose_attack_strategy(self) -> Tuple[str, str]:
        """
        Decide which attack type to use at episode start.

        Warm-up (< _MIN_EPISODES): probe with PHISHING.
        After warm-up: UCB1 bandit selects the arm; 10% random exploration
        is already baked into UCB1's exploration bonus, so we add a further
        5% forced-random to keep the defender from pattern-matching us.
        """
        if self.episode_count < _MIN_EPISODES:
            reasoning = (
                f"Episode {self.episode_count + 1}: warm-up probe with PHISHING "
                "to collect defender behaviour baseline."
            )
            return "PHISHING", reasoning

        # UCB1 selection
        bandit_pick = self._bandit.select(self.rng)

        # 5% forced random on top of UCB1's own exploration
        if self.rng.random() < 0.05:
            forced = self.rng.choice(self.ALL_STRATEGIES)
            reasoning = (
                f"Forced random exploration (5%): using {forced}. "
                f"UCB1 had suggested {bandit_pick}."
            )
            return forced, reasoning

        # Map bandit pick → heuristic counter if defender profile is strong
        label   = self.defender_profile.get_defender_strategy_label()
        counter = self.COUNTER_STRATEGY.get(label, bandit_pick)

        # Blend: if UCB1 and heuristic agree, use it; otherwise prefer UCB1
        # (bandit has seen the actual rewards; heuristic is one-shot inference)
        if bandit_pick == counter:
            chosen    = bandit_pick
            reasoning = (
                f"UCB1 and heuristic agree: {chosen} (defender={label})."
            )
        else:
            chosen    = bandit_pick
            reasoning = (
                f"UCB1 selects {bandit_pick} (heuristic suggested {counter} "
                f"for {label} defender). Trusting bandit — it has observed rewards."
            )

        self.adaptation_log.append(
            f"Episode {self.episode_count + 1}: "
            f"defender_profile={label} | bandit={bandit_pick} | chosen={chosen}"
        )
        return chosen, reasoning

    def on_episode_start(self) -> dict:
        """Call at the start of each episode. Returns the episode attack plan."""
        strategy, reasoning = self.choose_attack_strategy()
        self.current_strategy         = strategy
        self._episode_start_strategy  = strategy
        self._mid_episode_switches    = 0             # reset per-episode guard
        self._mid_episode_strategy    = None          # clear stale mid-ep record
        self.defender_profile.recent_window.clear()   # clear stale window from prior episode

        self.defender_profile._recalculate_rates()
        plan = {
            "episode":          self.episode_count + 1,
            "attack_strategy":  strategy,
            "reasoning":        reasoning,
            "defender_profile": {
                "strategy_label":   self.defender_profile.get_defender_strategy_label(),
                "isolation_rate":   round(self.defender_profile.isolation_rate, 3),
                "block_rate":       round(self.defender_profile.block_rate,     3),
                "scan_rate":        round(self.defender_profile.scan_rate,      3),
                "patch_rate":       round(self.defender_profile.patch_rate,     3),
                "most_used_action": self.defender_profile.get_most_used_action(),
                "steps_observed":   self.defender_profile.steps_observed,
            },
            "bandit_state": self._bandit.summary(),
        }
        self.strategy_history.append(plan)
        return plan

    def on_episode_end(self, defender_won: bool, score: float) -> None:
        """
        Call at end of each episode.

        Updates the UCB1 bandit with attacker reward = 1 - defender_score.
        High defender score → low attacker reward → this strategy is bad for us.

        Credit split: if a mid-episode switch fired, the episode outcome reflects
        both strategies.  60% of credit goes to the mid-episode strategy (it ran
        longer and dominated the outcome) and 40% to the start strategy.
        """
        attacker_reward = 1.0 - max(0.0, min(1.0, score))
        if self._mid_episode_switches > 0 and self._mid_episode_strategy:
            self._bandit.update(self._mid_episode_strategy,   attacker_reward * 0.60)
            self._bandit.update(self._episode_start_strategy, attacker_reward * 0.40)
        else:
            self._bandit.update(self._episode_start_strategy, attacker_reward)

        self.episode_count += 1
        if defender_won and score > 0.8:
            self.adaptation_log.append(
                f"Episode {self.episode_count}: defender dominated "
                f"(score={score:.2f}). Bandit penalised {self._episode_start_strategy}."
            )

    # -----------------------------------------------------------------------
    # Config overrides
    # -----------------------------------------------------------------------

    def get_attack_config_override(self, strategy: Optional[str] = None) -> dict:
        """Return AttackEngine override dict for the given (or current) strategy."""
        key = strategy or self.current_strategy
        return dict(self.ATTACK_CONFIGS.get(key, self.ATTACK_CONFIGS["PHISHING"]))

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def get_full_adaptation_report(self) -> str:
        """Human-readable summary of attacker adaptation over all episodes."""
        p = self.defender_profile
        p._recalculate_rates()
        lines = [
            "=== RED TEAM ADAPTATION REPORT ===",
            f"Episodes observed : {self.episode_count}",
            f"Defender profile  : {p.get_defender_strategy_label()}",
            "Defender action breakdown:",
            f"  isolation_rate  : {p.isolation_rate:.1%}",
            f"  block_rate      : {p.block_rate:.1%}",
            f"  scan_rate       : {p.scan_rate:.1%}",
            f"  patch_rate      : {p.patch_rate:.1%}",
            "",
            "UCB1 bandit — strategy performance (attacker reward = 1 - defender_score):",
        ]
        for arm, stats in sorted(
            self._bandit.summary().items(),
            key=lambda x: -x[1]["mean_reward"],
        ):
            bar = "#" * int(stats["mean_reward"] * 20)
            lines.append(
                f"  {arm:<16} pulls={stats['pulls']:>3}  "
                f"mean_reward={stats['mean_reward']:.4f}  {bar}"
            )

        lines += ["", "Strategy evolution (last 5 episodes):"]
        for entry in self.strategy_history[-5:]:
            truncated = entry["reasoning"][:70]
            lines.append(
                f"  Ep {entry['episode']:02d}: "
                f"{entry['attack_strategy']:16s} | {truncated}..."
            )

        if self.adaptation_log:
            lines += ["", "Adaptation events:"]
            for msg in self.adaptation_log[-8:]:
                lines.append(f"  {msg}")

        return "\n".join(lines)
