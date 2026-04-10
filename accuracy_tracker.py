"""
LLM Reasoning Accuracy Tracker
================================
Measures how accurately the LLM classifies threats from IOC signals alone.

Method
------
Since ground-truth threat types are intentionally withheld from all API
responses (by design — agents must infer from behavioral signals), accuracy
is estimated via a **reward-proxy**:

    predicted type  →  MITRE-correct action for that type
                    →  action chosen by LLM
                    →  reward returned by server

If the LLM predicted type X, chose the MITRE-correct action for X,
AND the server returned a high reward (≥ 0.65), the prediction is
treated as *confirmed correct*.

This is strictly better than comparing to a ground-truth label for this
use case: it measures whether the LLM's classification led to the RIGHT
defensive outcome — which is what actually matters in the SOC scenario.

The tracker also accepts an optional ``actual_type`` parameter in
``record()``. If a caller provides it (e.g. a server-side harness with
access to internal state), ground-truth comparison is used instead.

No ground-truth type is ever exposed through any API endpoint.

Usage
-----
    from accuracy_tracker import AccuracyTracker

    tracker = AccuracyTracker()
    tracker.record(reasoning="high outbound bytes and spread — ransomware",
                   action="isolate_machine",
                   reward=0.88)
    metrics = tracker.summary()
    # {"accuracy": 1.0, "total_predictions": 1, "per_type_accuracy": {"ransomware": 1.0}, ...}
"""

from __future__ import annotations

from typing import Optional


# ---------------------------------------------------------------------------
# MITRE keyword map  — keywords that appear in LLM reasoning → threat class
# ---------------------------------------------------------------------------

# Each entry: threat_type → list of substrings to search for (case-insensitive)
THREAT_KEYWORDS: dict[str, list[str]] = {
    "ransomware": [
        "ransomware", "encrypt", "spread_rate", "propagat", "data encrypted",
        "t1486", "spread rate",
    ],
    "phishing": [
        "phish", "spear", "credential", "auth fail", "failed auth",
        "t1566", "spear phishing",
    ],
    "lateral": [
        "lateral", "movement", "pivot", "remote serv", "lateral movement",
        "t1021", "lateral connection",
    ],
    "ddos": [
        "ddos", "denial of service", "flood", "volumetric", "packet rate",
        "packets per second", "t1499",
    ],
    "malware": [
        "malware", "unusual process", "payload", "user execution",
        "t1204", "process anomal",
    ],
}

# MITRE-correct action for each threat type — single source of truth
# Must match the mapping in agents/baseline.py and app.py's _get_correct_action.
TYPE_TO_ACTION: dict[str, str] = {
    "ransomware": "isolate_machine",
    "phishing":   "block_ip",
    "lateral":    "block_ip",
    "ddos":       "patch",
    "malware":    "isolate_machine",
}

# Reward threshold above which a mitigation is treated as "server-confirmed correct"
CONFIRMATION_REWARD_THRESHOLD = 0.65

# Mitigation actions — scans and ignore are not confirmation candidates
_MITIGATION_ACTIONS = {"block_ip", "isolate_machine", "patch"}


# ---------------------------------------------------------------------------
# AccuracyTracker
# ---------------------------------------------------------------------------

class AccuracyTracker:
    """
    Per-episode LLM threat-classification accuracy tracker.

    Instantiate once per episode; call reset() or create a new instance
    between episodes.
    """

    def __init__(self) -> None:
        self.total_predictions: int = 0
        self.confirmed_correct: int = 0
        self._prediction_dist: dict[str, int] = {}
        self._per_type: dict[str, dict[str, int]] = {}
        self._step_records: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_prediction(self, reasoning: str) -> Optional[str]:
        """
        Parse free-form LLM reasoning text and return the implicit predicted
        threat class, or None if no recognisable keyword is found.

        The first matching class wins — ordered by specificity so that more
        distinctive terms (e.g. "ransomware") are checked before generic ones
        (e.g. "malware").
        """
        if not reasoning:
            return None

        text = reasoning.lower()
        for threat_type, keywords in THREAT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return threat_type

        return None

    def record(
        self,
        reasoning: str,
        action: str,
        reward: float,
        actual_type: Optional[str] = None,
    ) -> None:
        """
        Record one step's prediction and evaluate correctness.

        Parameters
        ----------
        reasoning   : LLM reasoning string from choose_action()
        action      : environment action that was submitted (e.g. "isolate_machine")
        reward      : reward returned by the server after the action
        actual_type : ground-truth threat type if available from internal state.
                      When None (default), reward-proxy mode is used.
        """
        predicted = self.extract_prediction(reasoning)

        step_record: dict = {
            "predicted": predicted,
            "actual":    actual_type,
            "action":    action,
            "reward":    round(float(reward), 3),
            "confirmed": False,
        }

        if predicted is None:
            # Reasoning contained no recognisable threat keyword — skip accuracy update
            self._step_records.append(step_record)
            return

        self.total_predictions += 1
        self._prediction_dist[predicted] = self._prediction_dist.get(predicted, 0) + 1

        if predicted not in self._per_type:
            self._per_type[predicted] = {"predicted": 0, "confirmed": 0}
        self._per_type[predicted]["predicted"] += 1

        # ── Determine correctness ──────────────────────────────────────────────
        if actual_type is not None:
            # Ground-truth mode — direct type comparison
            correct = (predicted == actual_type.lower().strip())
        else:
            # Reward-proxy mode:
            # LLM predicted type T → expected action = TYPE_TO_ACTION[T]
            # If the LLM chose that expected action AND the server rewarded it
            # above threshold → treat as a confirmed correct classification.
            expected_action = TYPE_TO_ACTION.get(predicted)
            correct = (
                action in _MITIGATION_ACTIONS
                and action == expected_action
                and float(reward) >= CONFIRMATION_REWARD_THRESHOLD
            )

        if correct:
            self.confirmed_correct += 1
            self._per_type[predicted]["confirmed"] += 1

        step_record["confirmed"] = correct
        self._step_records.append(step_record)

    def summary(self) -> dict:
        """
        Return accuracy metrics for the current episode.

        Keys
        ----
        accuracy             : confirmed_correct / total_predictions  (0.0–1.0)
        total_predictions    : steps where reasoning contained a threat keyword
        confirmed_correct    : predictions confirmed by action + reward
        per_type_accuracy    : per-class confirmation rate
        prediction_dist      : count of each predicted class
        mode                 : "ground_truth" or "reward_proxy"
        """
        if self.total_predictions == 0:
            return {
                "accuracy":          0.0,
                "total_predictions": 0,
                "confirmed_correct": 0,
                "per_type_accuracy": {},
                "prediction_dist":   {},
                "mode":              "reward_proxy",
            }

        per_type_acc = {
            t: round(v["confirmed"] / max(1, v["predicted"]), 4)
            for t, v in self._per_type.items()
        }

        # Detect which mode was actually used
        used_gt = any(r["actual"] is not None for r in self._step_records)
        mode = "ground_truth" if used_gt else "reward_proxy"

        return {
            "accuracy":          round(self.confirmed_correct / self.total_predictions, 4),
            "total_predictions": self.total_predictions,
            "confirmed_correct": self.confirmed_correct,
            "per_type_accuracy": per_type_acc,
            "prediction_dist":   dict(self._prediction_dist),
            "mode":              mode,
        }

    def log(self, prefix: str = "") -> None:
        """Print [INFO] accuracy lines — safe to call at any time."""
        s = self.summary()
        tag = f"[{prefix}] " if prefix else ""
        total = s["total_predictions"]
        if total == 0:
            print(f"  {tag}[INFO] LLM Accuracy: no predictions yet")
            return

        correct = s["confirmed_correct"]
        acc     = s["accuracy"]
        print(f"  {tag}[INFO] LLM Accuracy: {acc:.2%} ({correct}/{total}) "
              f"mode={s['mode']}")

        if s["per_type_accuracy"]:
            parts = ", ".join(
                f"{k}={v:.0%}" for k, v in sorted(s["per_type_accuracy"].items())
            )
            print(f"  {tag}[INFO] Per-type: {parts}")

        if s["prediction_dist"]:
            dist = ", ".join(
                f"{k}={v}" for k, v in sorted(s["prediction_dist"].items())
            )
            print(f"  {tag}[INFO] Prediction dist: {dist}")

    def reset(self) -> None:
        """Reset all counters — call between episodes when reusing an instance."""
        self.total_predictions = 0
        self.confirmed_correct = 0
        self._prediction_dist.clear()
        self._per_type.clear()
        self._step_records.clear()
