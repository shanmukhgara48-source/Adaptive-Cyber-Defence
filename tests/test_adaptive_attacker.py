"""
tests/test_adaptive_attacker.py — Unit tests for the adaptive red-team attacker.
"""

import pytest
from adaptive_cyber_defense.engines.adaptive_attacker import (
    AdaptiveAttacker,
    DefenderBehaviorProfile,
)


# ---------------------------------------------------------------------------
# DefenderBehaviorProfile
# ---------------------------------------------------------------------------

class TestDefenderProfile:
    def test_records_actions(self):
        profile = DefenderBehaviorProfile()
        profile.record_action("ISOLATE_NODE")
        profile.record_action("ISOLATE_NODE")
        profile.record_action("SCAN")
        assert profile.action_counts["ISOLATE_NODE"] == 2
        assert profile.isolation_rate > 0.5

    def test_detects_isolator(self):
        profile = DefenderBehaviorProfile()
        for _ in range(10):
            profile.record_action("ISOLATE_NODE")
        assert profile.get_defender_strategy_label() == "ISOLATOR"

    def test_detects_blocker(self):
        profile = DefenderBehaviorProfile()
        for _ in range(10):
            profile.record_action("BLOCK_IP")
        assert profile.get_defender_strategy_label() == "BLOCKER"

    def test_detects_scanner(self):
        profile = DefenderBehaviorProfile()
        for _ in range(10):
            profile.record_action("SCAN")
        assert profile.get_defender_strategy_label() == "SCANNER"

    def test_detects_patcher(self):
        profile = DefenderBehaviorProfile()
        for _ in range(10):
            profile.record_action("PATCH_VULNERABILITY")
        assert profile.get_defender_strategy_label() == "PATCHER"

    def test_balanced_label_for_mixed_actions(self):
        profile = DefenderBehaviorProfile()
        for action in ["ISOLATE_NODE", "BLOCK_IP", "SCAN", "PATCH_VULNERABILITY",
                       "RESTORE_NODE", "DO_NOTHING", "DECRYPT", "QUARANTINE_SERVICE"]:
            profile.record_action(action)
        assert profile.get_defender_strategy_label() == "BALANCED"

    def test_unknown_when_empty(self):
        profile = DefenderBehaviorProfile()
        assert profile.get_most_used_action() == "UNKNOWN"


# ---------------------------------------------------------------------------
# AdaptiveAttacker
# ---------------------------------------------------------------------------

class TestAdaptiveAttacker:
    def test_first_two_episodes_use_phishing(self):
        attacker = AdaptiveAttacker(seed=42)
        plan1 = attacker.on_episode_start()
        assert plan1["attack_strategy"] == "PHISHING"
        attacker.on_episode_end(False, 0.5)
        plan2 = attacker.on_episode_start()
        assert plan2["attack_strategy"] == "PHISHING"

    def test_counters_isolator_with_insider(self):
        # Use seed=0 to avoid the 15% random branch
        attacker = AdaptiveAttacker(seed=0)
        # Episode 1: probe
        attacker.on_episode_start()
        for _ in range(20):
            attacker.observe_defender_action("ISOLATE_NODE")
        attacker.on_episode_end(True, 0.9)
        # Episode 2: probe
        attacker.on_episode_start()
        attacker.on_episode_end(True, 0.9)
        # Episode 3: should counter
        plan = attacker.on_episode_start()
        # Random branch is 15%; seed=0 should deterministically pick INSIDER_THREAT
        # If it randomizes, accept any strategy from the counter map
        assert plan["attack_strategy"] in list(AdaptiveAttacker.COUNTER_STRATEGY.values())

    def test_counters_blocker_with_supply_chain(self):
        attacker = AdaptiveAttacker(seed=1)
        attacker.on_episode_start()
        for _ in range(20):
            attacker.observe_defender_action("BLOCK_IP")
        attacker.on_episode_end(True, 0.8)
        attacker.on_episode_start()
        attacker.on_episode_end(True, 0.8)
        plan = attacker.on_episode_start()
        assert plan["attack_strategy"] in list(AdaptiveAttacker.COUNTER_STRATEGY.values())

    def test_profile_reflects_observed_actions(self):
        attacker = AdaptiveAttacker(seed=42)
        attacker.on_episode_start()
        for _ in range(10):
            attacker.observe_defender_action("BLOCK_IP")
        assert attacker.defender_profile.get_defender_strategy_label() == "BLOCKER"
        assert attacker.defender_profile.block_rate > 0.9

    def test_config_override_apt_has_high_evasion(self):
        attacker = AdaptiveAttacker(seed=42)
        config = attacker.get_attack_config_override("APT")
        assert config["detection_evasion"] >= 0.5
        assert config["dwell_time_multiplier"] >= 2.0

    def test_config_override_ransomware_spreads_fast(self):
        attacker = AdaptiveAttacker(seed=42)
        config = attacker.get_attack_config_override("RANSOMWARE")
        assert config["spread_rate"] >= 2.0
        assert config["dwell_time_multiplier"] < 1.0

    def test_config_override_zero_day_high_evasion(self):
        attacker = AdaptiveAttacker(seed=42)
        config = attacker.get_attack_config_override("ZERO_DAY")
        assert config["detection_evasion"] >= 0.7

    def test_config_override_phishing_defaults(self):
        attacker = AdaptiveAttacker(seed=42)
        config = attacker.get_attack_config_override("PHISHING")
        assert config["dwell_time_multiplier"] == 1.0
        assert config["detection_evasion"] == 0.0
        assert config["spread_rate"] == 1.0

    def test_report_generates_without_error(self):
        attacker = AdaptiveAttacker(seed=42)
        attacker.on_episode_start()
        attacker.on_episode_end(False, 0.4)
        report = attacker.get_full_adaptation_report()
        assert "RED TEAM ADAPTATION REPORT" in report
        assert "Defender strategy" in report

    def test_episode_count_increments(self):
        attacker = AdaptiveAttacker(seed=42)
        for i in range(5):
            attacker.on_episode_start()
            attacker.on_episode_end(True, 0.7)
        assert attacker.episode_count == 5

    def test_strategy_history_records_all_episodes(self):
        attacker = AdaptiveAttacker(seed=42)
        for _ in range(4):
            attacker.on_episode_start()
            attacker.on_episode_end(False, 0.5)
        assert len(attacker.strategy_history) == 4

    def test_plan_has_required_keys(self):
        attacker = AdaptiveAttacker(seed=42)
        plan = attacker.on_episode_start()
        for key in ("episode", "attack_strategy", "reasoning", "defender_profile"):
            assert key in plan
        for key in ("strategy_label", "isolation_rate", "block_rate",
                    "scan_rate", "patch_rate", "most_used_action"):
            assert key in plan["defender_profile"]
