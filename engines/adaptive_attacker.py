"""
engines/adaptive_attacker.py — Adaptive red-team attacker.

Observes what the defender does across episodes and switches attack strategy
to exploit the defender's most predictable weakness.

Core idea:
  ISOLATOR  defender  → INSIDER_THREAT   (bypasses isolation via valid creds)
  BLOCKER   defender  → SUPPLY_CHAIN     (no external IP to block)
  SCANNER   defender  → APT              (low-and-slow, stays under scan threshold)
  PATCHER   defender  → ZERO_DAY         (exploit is unpatched by definition)
  BALANCED  defender  → RANSOMWARE       (time pressure breaks balanced play)
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Defender behaviour profile
# ---------------------------------------------------------------------------

@dataclass
class DefenderBehaviorProfile:
    """Tracks what the defender does most — attacker exploits this."""
    action_counts: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    actions_per_threat_type: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(int))
    )
    isolation_rate: float = 0.0
    block_rate:     float = 0.0
    scan_rate:      float = 0.0
    patch_rate:     float = 0.0
    steps_observed: int   = 0
    episodes_observed: int = 0

    def record_action(self, action_name: str, threat_type: str = "UNKNOWN") -> None:
        self.action_counts[action_name] += 1
        self.actions_per_threat_type[threat_type][action_name] += 1
        self.steps_observed += 1
        total = max(1, self.steps_observed)
        self.isolation_rate = self.action_counts.get("ISOLATE_NODE", 0) / total
        self.block_rate     = self.action_counts.get("BLOCK_IP",     0) / total
        self.scan_rate      = self.action_counts.get("SCAN",         0) / total
        self.patch_rate     = (
            self.action_counts.get("PATCH_VULNERABILITY", 0) / total
        )

    def get_most_used_action(self) -> str:
        if not self.action_counts:
            return "UNKNOWN"
        return max(self.action_counts, key=self.action_counts.get)

    def get_defender_strategy_label(self) -> str:
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


# ---------------------------------------------------------------------------
# Adaptive attacker
# ---------------------------------------------------------------------------

class AdaptiveAttacker:
    """
    Red-team attacker that learns the defender's strategy and counters it.

    Lifecycle::

        attacker = AdaptiveAttacker(seed=42)

        for episode in range(N):
            plan = attacker.on_episode_start()
            # ... run episode, each step:
            attacker.observe_defender_action(action_name)
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
            "Defender relies on ISOLATE_NODE. Switching to INSIDER THREAT — "
            "insider uses valid credentials and internal access, bypassing isolation."
        ),
        "BLOCKER": (
            "Defender relies on BLOCK_IP. Switching to SUPPLY CHAIN — "
            "attack originates from trusted internal service, no external IP to block."
        ),
        "SCANNER": (
            "Defender relies on SCAN. Switching to APT low-and-slow — "
            "staying below scan detection threshold, advancing only every 5+ steps."
        ),
        "PATCHER": (
            "Defender relies on PATCH_VULNERABILITY. Switching to ZERO DAY — "
            "exploits unknown vulnerability, patch is not yet available."
        ),
        "BALANCED": (
            "Defender uses balanced strategy. Switching to RANSOMWARE — "
            "time pressure forces defender to abandon balance and react urgently."
        ),
    }

    # Attack engine overrides per strategy
    ATTACK_CONFIGS: Dict[str, Dict] = {
        "PHISHING": {
            "initial_attack_type":    "PHISHING",
            "dwell_time_multiplier":  1.0,
            "detection_evasion":      0.0,
            "spread_rate":            1.0,
        },
        "APT": {
            "initial_attack_type":    "PHISHING",
            "dwell_time_multiplier":  3.0,   # stays in each stage 3× longer
            "detection_evasion":      0.6,   # 60% harder to detect
            "spread_rate":            0.3,   # spreads slowly
        },
        "RANSOMWARE": {
            "initial_attack_type":    "MALWARE",
            "dwell_time_multiplier":  0.3,   # moves very fast
            "detection_evasion":      0.0,
            "spread_rate":            3.0,   # spreads aggressively
        },
        "INSIDER_THREAT": {
            "initial_attack_type":    "ACCESS",   # starts already inside
            "dwell_time_multiplier":  1.5,
            "detection_evasion":      0.7,         # hard to detect (legit creds)
            "spread_rate":            0.8,
        },
        "SUPPLY_CHAIN": {
            "initial_attack_type":    "MALWARE",
            "dwell_time_multiplier":  2.0,
            "detection_evasion":      0.5,
            "spread_rate":            2.0,   # affects many nodes at once
        },
        "ZERO_DAY": {
            "initial_attack_type":    "MALWARE",
            "dwell_time_multiplier":  1.0,
            "detection_evasion":      0.8,   # almost undetectable at first
            "spread_rate":            1.5,
        },
    }

    def __init__(self, seed: int = 42, learning_rate: float = 0.3) -> None:
        self.rng              = random.Random(seed)
        self.learning_rate    = learning_rate
        self.defender_profile = DefenderBehaviorProfile()
        self.current_strategy = "PHISHING"
        self.strategy_history: List[dict]  = []
        self.episode_count:    int          = 0
        self.adaptation_log:   List[str]   = []

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
    # Episode lifecycle
    # -----------------------------------------------------------------------

    def choose_attack_strategy(self) -> tuple[str, str]:
        """
        Decide which attack type to use this episode.
        Returns (strategy_name, reasoning).
        """
        if self.episode_count < 2:
            reasoning = (
                f"Episode {self.episode_count + 1}: Probing defender with "
                "standard PHISHING attack to observe response patterns."
            )
            return "PHISHING", reasoning

        label   = self.defender_profile.get_defender_strategy_label()
        counter = self.COUNTER_STRATEGY[label]
        reasoning = self.COUNTER_REASONING[label]

        # 15% random exploration so defender can't perfectly predict us
        if self.rng.random() < 0.15:
            counter = self.rng.choice(list(self.COUNTER_STRATEGY.values()))
            reasoning = (
                f"Randomizing strategy (15% chance) to stay unpredictable. "
                f"Using {counter} this episode."
            )

        self.adaptation_log.append(
            f"Episode {self.episode_count + 1}: "
            f"Defender profile={label} → Counter strategy={counter}"
        )
        return counter, reasoning

    def on_episode_start(self) -> dict:
        """
        Call at the start of each episode.
        Returns the episode attack plan dict.
        """
        strategy, reasoning = self.choose_attack_strategy()
        self.current_strategy = strategy

        plan = {
            "episode":          self.episode_count + 1,
            "attack_strategy":  strategy,
            "reasoning":        reasoning,
            "defender_profile": {
                "strategy_label":  self.defender_profile.get_defender_strategy_label(),
                "isolation_rate":  round(self.defender_profile.isolation_rate, 3),
                "block_rate":      round(self.defender_profile.block_rate,     3),
                "scan_rate":       round(self.defender_profile.scan_rate,      3),
                "patch_rate":      round(self.defender_profile.patch_rate,     3),
                "most_used_action":self.defender_profile.get_most_used_action(),
                "steps_observed":  self.defender_profile.steps_observed,
            },
        }
        self.strategy_history.append(plan)
        return plan

    def on_episode_end(self, defender_won: bool, score: float) -> None:
        """Call at the end of each episode to update learning state."""
        self.episode_count += 1
        if defender_won and score > 0.8:
            self.adaptation_log.append(
                f"Episode {self.episode_count}: Defender dominated "
                f"(score={score:.2f}). Will escalate intensity next episode."
            )

    # -----------------------------------------------------------------------
    # Config overrides
    # -----------------------------------------------------------------------

    def get_attack_config_override(self, strategy: str) -> dict:
        """Return AttackEngine override dict for the chosen strategy."""
        return dict(self.ATTACK_CONFIGS.get(strategy, self.ATTACK_CONFIGS["PHISHING"]))

    # -----------------------------------------------------------------------
    # Reporting
    # -----------------------------------------------------------------------

    def get_full_adaptation_report(self) -> str:
        """Human-readable summary of attacker adaptation over all episodes."""
        p = self.defender_profile
        lines = [
            "=== RED TEAM ADAPTATION REPORT ===",
            f"Episodes observed: {self.episode_count}",
            f"Defender strategy detected: {p.get_defender_strategy_label()}",
            "Defender action breakdown:",
            f"  Isolation rate: {p.isolation_rate:.1%}",
            f"  Block rate:     {p.block_rate:.1%}",
            f"  Scan rate:      {p.scan_rate:.1%}",
            f"  Patch rate:     {p.patch_rate:.1%}",
            "",
            "Strategy evolution:",
        ]
        for entry in self.strategy_history[-5:]:
            truncated = entry["reasoning"][:60]
            lines.append(
                f"  Ep {entry['episode']:02d}: "
                f"{entry['attack_strategy']:15s} | "
                f"{truncated}..."
            )
        return "\n".join(lines)
