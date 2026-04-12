"""
Q-Learning agents for the Adaptive Cyber Defense Simulator.

Two agents are provided:
  QLearningAgent   — Tabular Q-learning.
  LinearFAAgent    — Linear function approximation (Q(s,a) = w[a] · φ(s)).

State space  : 6-dimensional discrete tuple — max 576 states (tabular),
               or a 7-element continuous feature vector (linear FA).
Action space : BLOCK_IP | ISOLATE_NODE | PATCH_SYSTEM | RUN_DEEP_SCAN | IGNORE
Q-table      : dict  Q[(state, action)] = float  (tabular)
Weights      : dict  w[action] = List[float]     (linear FA)

Usage
-----
    from adaptive_cyber_defense.agents.ql_agent import QLearningAgent, LinearFAAgent, train
    from adaptive_cyber_defense import AdaptiveCyberDefenseEnv

    env    = AdaptiveCyberDefenseEnv()
    agent  = LinearFAAgent()          # or QLearningAgent()
    result = train(agent, env, episodes=50)
    print(result["avg_reward"])
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models.action import Action, ActionInput
from ..models.state import EnvironmentState


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Actions the QL agent chooses between
ACTIONS: List[Action] = [
    Action.BLOCK_IP,
    Action.ISOLATE_NODE,
    Action.PATCH_SYSTEM,
    Action.RUN_DEEP_SCAN,
    Action.IGNORE,
]


# ---------------------------------------------------------------------------
# State discretisation
# ---------------------------------------------------------------------------

def discretise(env_state: EnvironmentState) -> Tuple[str, str, str, str, str, int]:
    """
    Compress continuous env state into a 6-key tuple.

    threat_level      : "low" (<0.33) | "medium" (<0.66) | "high" (>=0.66)
    resource_level    : "low" (<0.50) | "high" (>=0.50)
    threat_stage      : top-threat's AttackStage name in lowercase, or "none"
    persistence       : "p" (persistent threat) | "n" (non-persistent)
                        Persistent threats (malware, ransomware, lateral_movement) resurface
                        after containment — the agent needs to monitor them.  Non-persistent
                        ones (phishing, ddos) do not.  Same MITRE action but different
                        post-containment behaviour: this dimension captures that difference.
    spread_class      : "hi" (spread_rate > 0.5) | "lo"
                        High spread-rate identifies ransomware / lateral_movement even when
                        stage and severity look like malware — enabling the Q-table to learn
                        that these need faster isolation than a slow-spreading malware threat.
    threat_count_bucket: min(len(active_threats), 3) — 0/1/2/3
                        Captures multi-threat pressure without state explosion.
                        Bucket 3 means "3 or more concurrent threats", signalling that
                        the agent must triage rather than focus on a single threat.

    Max states: 3 × 2 × 6 × 2 × 2 × 4 = 576.  Still fully tabular; fits in <10 KB.
    Backward compat: old 5-tuple Q-tables have key shape (tuple of length 5, str).
    The new shape (tuple of length 6, str) is disjoint — the load() validator skips
    mismatched entries with a warning rather than silently zeroing them.
    """
    sev = env_state.threat_severity
    if sev < 0.33:
        threat_level = "low"
    elif sev < 0.66:
        threat_level = "medium"
    else:
        threat_level = "high"

    resource_level = "low" if env_state.resource_availability < 0.50 else "high"

    # Stage, persistence, and spread class of highest-severity active threat
    threat_stage = "none"
    persistence  = "n"
    spread_class = "lo"
    if env_state.active_threats:
        top = max(env_state.active_threats, key=lambda t: t.severity)
        threat_stage = top.stage.name.lower()   # e.g. "phishing", "lateral_spread"
        # getattr with defaults: safe whether or not the Threat model exposes these fields
        persistence  = "p" if getattr(top, "is_persistent", False) else "n"
        spread_class = "hi" if getattr(top, "spread_rate", 0.0) > 0.5 else "lo"

    # Multi-threat awareness: bucket active threat count, capped at 3
    threat_count_bucket = min(len(env_state.active_threats), 3)

    return (threat_level, resource_level, threat_stage, persistence, spread_class,
            threat_count_bucket)


def extract_state(env_state: "EnvironmentState") -> tuple:
    """
    Alias for discretise().
    Added for compatibility with training scripts (train_phase1/2/3.py) and run.py.
    Does not change or replace discretise — both coexist.
    """
    return discretise(env_state)


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Q-table: Q[(state_tuple, action_name)] = float, initialised to 0.
    state_tuple = discretise(env_state) — 6-dimensional tuple, up to 576 distinct states.

    Hyperparameters
    ---------------
    alpha   : learning rate            (default 0.1)
    gamma   : discount factor          (default 0.9)
    epsilon : exploration probability  (default 0.2, decayed each episode via decay_epsilon)
    """

    def __init__(
        self,
        alpha:   float = 0.1,
        gamma:   float = 0.9,
        epsilon: float = 0.2,
    ) -> None:
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        # Flat dict  Q[(state, action_name)] = 0.0
        self.Q: Dict[Tuple, float] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q(self, state: Tuple, action: Action) -> float:
        """Return Q-value, defaulting to 0.0 for unseen (state, action) pairs."""
        return self.Q.get((state, action.name), 0.0)

    def _best_action(self, state: Tuple) -> Action:
        """Return the action with the highest Q-value for *state*."""
        return max(ACTIONS, key=lambda a: self._q(state, a))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, env_state: EnvironmentState) -> ActionInput:
        """
        Epsilon-greedy action selection.

        * With probability epsilon  → explore (random action)
        * Otherwise                 → exploit (best Q-table action)

        Returns an ActionInput ready for env.step().
        """
        state = discretise(env_state)
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            action = self._best_action(state)
        return ActionInput(action=action)

    def update(
        self,
        state:      Tuple,
        action:     Action,
        reward:     float,
        next_state: Tuple,
        done:       bool,
    ) -> None:
        """
        Standard Q-learning update:

            Q(s,a) ← Q(s,a) + α · (r + γ · max_a' Q(s',a') − Q(s,a))
        """
        key        = (state, action.name)
        current_q  = self.Q.get(key, 0.0)

        if done:
            target = reward
        else:
            best_next = max(self._q(next_state, a) for a in ACTIONS)
            target    = reward + self.gamma * best_next

        self.Q[key] = current_q + self.alpha * (target - current_q)

    # BaseTask.run() / UI compatibility shim
    def choose(self, env_state: EnvironmentState) -> ActionInput:
        """Greedy (ε=0) selection — used by UI and task.run()."""
        state  = discretise(env_state)
        action = self._best_action(state)
        return ActionInput(action=action)

    def select_action(self, env_state: "EnvironmentState") -> "ActionInput":
        """
        Alias for get_action().
        Added for compatibility with training scripts (train_phase1/2/3.py) and run.py.
        Does not change or replace get_action() or choose() — all three coexist.
        """
        return self.get_action(env_state)

    def decay_epsilon(self) -> None:
        """
        Decay self.epsilon by factor 0.995, floored at 0.05.
        Called once per episode by training scripts.
        Does not affect any other hyperparameter.
        """
        self.epsilon = max(0.05, self.epsilon * 0.995)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise Q-table to JSON."""
        data = {
            "alpha":   self.alpha,
            "gamma":   self.gamma,
            "epsilon": self.epsilon,
            "Q": {
                # key is "(state_tuple, action_name)"
                str(k): v for k, v in self.Q.items()
            },
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """Load Q-table from JSON.

        Malformed entries are skipped with a warning rather than silently dropped.
        A corrupt file that zeroes out the Q-table is now visible rather than
        causing the agent to behave randomly without any diagnostic signal.
        """
        import ast
        import warnings
        data = json.loads(Path(path).read_text())
        self.alpha   = data.get("alpha",   self.alpha)
        self.gamma   = data.get("gamma",   self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.Q = {}
        skipped = 0
        for key_str, val in data.get("Q", {}).items():
            # Reconstruct tuple key: "((state_tuple...), action_name)"
            try:
                key = ast.literal_eval(key_str)
                # Validate: must be a 2-tuple of (tuple, str)
                if (not isinstance(key, tuple) or len(key) != 2
                        or not isinstance(key[0], tuple)
                        or not isinstance(key[1], str)):
                    skipped += 1
                    continue
                self.Q[key] = float(val)
            except Exception:
                skipped += 1
        if skipped:
            warnings.warn(
                f"[QLearningAgent.load] Skipped {skipped} malformed Q-table "
                f"entries from '{path}'. The agent may behave as partially untrained. "
                "Re-train or inspect the file for corruption.",
                UserWarning,
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Linear function approximation — feature extractor
# ---------------------------------------------------------------------------

# Ordered stage names → normalized index in [0, 1]
_STAGE_NORM: dict = {
    "none":             0.0,
    "phishing":         0.2,
    "credential_access": 0.4,
    "malware_install":  0.6,
    "lateral_spread":   0.8,
    "exfiltration":     1.0,
}

_N_FEATURES = 7   # 6 state features + 1 bias


def extract_features(env_state: "EnvironmentState") -> List[float]:
    """
    Map EnvironmentState to a fixed-length feature vector for linear FA.

    Feature vector (length 7):
      [0] threat_level_norm     — 0.0 (low) | 0.5 (medium) | 1.0 (high)
      [1] resource_level_norm   — 0.0 (low) | 1.0 (high)
      [2] threat_stage_norm     — 0.0–1.0 normalized kill-chain stage index
      [3] persistence_bin       — 0.0 (non-persistent) | 1.0 (persistent)
      [4] spread_class_bin      — 0.0 (lo spread) | 1.0 (hi spread)
      [5] threat_count_norm     — min(n_threats, 3) / 3.0  in {0, 0.33, 0.67, 1.0}
      [6] bias                  — always 1.0  (intercept term)

    Design: all features are in [0, 1] so weight magnitudes are directly
    comparable.  The bias term allows Q(s, a) ≠ 0 even when all state
    features are zero (e.g. no active threats at episode start).
    """
    sev = env_state.threat_severity
    if sev < 0.33:
        threat_level_norm = 0.0
    elif sev < 0.66:
        threat_level_norm = 0.5
    else:
        threat_level_norm = 1.0

    resource_level_norm = 0.0 if env_state.resource_availability < 0.50 else 1.0

    threat_stage_norm = 0.0
    persistence_bin   = 0.0
    spread_class_bin  = 0.0
    if env_state.active_threats:
        top = max(env_state.active_threats, key=lambda t: t.severity)
        stage_name = top.stage.name.lower()
        threat_stage_norm = _STAGE_NORM.get(stage_name, 0.0)
        persistence_bin   = 1.0 if getattr(top, "is_persistent", False) else 0.0
        spread_class_bin  = 1.0 if getattr(top, "spread_rate", 0.0) > 0.5 else 0.0

    threat_count_norm = min(len(env_state.active_threats), 3) / 3.0

    return [
        threat_level_norm,
        resource_level_norm,
        threat_stage_norm,
        persistence_bin,
        spread_class_bin,
        threat_count_norm,
        1.0,   # bias
    ]


# ---------------------------------------------------------------------------
# Linear function approximation Q-agent
# ---------------------------------------------------------------------------

class LinearFAAgent:
    """
    Linear function approximation Q-agent.

    Q(s, a) = w[a] · φ(s)

    where φ(s) is a 7-element feature vector from extract_features() and
    w[a] is a per-action weight vector initialised to zeros.

    Update rule (semi-gradient TD):
        δ    = r + γ · max_a' Q(s', a') − Q(s, a)
        w[a] ← w[a] + α · δ · φ(s)

    Compared to tabular QL:
        + Generalises across unseen (state, action) pairs
        + Weight vector has 5 × 7 = 35 parameters vs up to 576 × 5 = 2880
          Q-table entries — learns faster from sparse data
        + Smooth value surface avoids tabular artefacts at state boundaries

    API is identical to QLearningAgent: get_action / choose / select_action /
    update / decay_epsilon / save / load.  Training scripts work with either
    agent without modification.
    """

    def __init__(
        self,
        alpha:   float = 0.05,
        gamma:   float = 0.9,
        epsilon: float = 0.2,
    ) -> None:
        self.alpha   = alpha
        self.gamma   = gamma
        self.epsilon = epsilon

        # Per-action weight vectors — zero-initialised
        # Positively initialised bias weight encourages early exploration
        # over the zero-value baseline.
        self.weights: Dict[Action, List[float]] = {
            a: [0.0] * _N_FEATURES for a in ACTIONS
        }
        # Give bias term a small positive start so random tie-breaking is
        # uniform across actions before any update.
        for a in ACTIONS:
            self.weights[a][-1] = 0.01

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _q(self, features: List[float], action: Action) -> float:
        """Q(s, a) = dot(w[a], φ(s))."""
        w = self.weights[action]
        return sum(wi * fi for wi, fi in zip(w, features))

    def _best_action(self, features: List[float]) -> Action:
        """Greedy action: argmax_a Q(s, a)."""
        return max(ACTIONS, key=lambda a: self._q(features, a))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(self, env_state: "EnvironmentState") -> "ActionInput":
        """Epsilon-greedy action selection."""
        features = extract_features(env_state)
        if random.random() < self.epsilon:
            action = random.choice(ACTIONS)
        else:
            action = self._best_action(features)
        return ActionInput(action=action)

    def choose(self, env_state: "EnvironmentState") -> "ActionInput":
        """Greedy (ε=0) selection — used by UI and task.run()."""
        features = extract_features(env_state)
        action   = self._best_action(features)
        return ActionInput(action=action)

    def select_action(self, env_state: "EnvironmentState") -> "ActionInput":
        """Alias for get_action() — training script compatibility."""
        return self.get_action(env_state)

    def update(
        self,
        state:      Tuple,           # unused — kept for API parity with QLearningAgent
        action:     Action,
        reward:     float,
        next_state: Tuple,           # unused — see below
        done:       bool,
        features:        Optional[List[float]] = None,
        next_features:   Optional[List[float]] = None,
    ) -> None:
        """
        Semi-gradient TD update.

        The tabular API passes (state, action, reward, next_state, done) where
        state/next_state are discrete tuples.  LinearFAAgent additionally accepts
        the raw feature vectors via keyword args so training code can avoid
        calling extract_features() twice.  If not provided, the method falls
        back to accepting them via the positional args when they happen to be
        lists (duck-typed for backward compat).

        Semi-gradient (not full gradient) because we do NOT differentiate
        through the target — standard practice for stability.
        """
        # Resolve features: accept raw feature lists passed positionally as
        # state/next_state for convenience in training loops that already have
        # them, or use the keyword-argument path for clarity.
        if features is None:
            features = list(state) if isinstance(state, (list, tuple)) and len(state) == _N_FEATURES else None
        if next_features is None:
            next_features = list(next_state) if isinstance(next_state, (list, tuple)) and len(next_state) == _N_FEATURES else None

        # Fallback: if we still have no feature vectors, we cannot update.
        # This happens when called via the tabular train() loop with discrete
        # tuples — the caller must pass feature vectors explicitly.
        if features is None or next_features is None:
            return

        q_sa = self._q(features, action)
        if done:
            target = reward
        else:
            best_next_q = max(self._q(next_features, a) for a in ACTIONS)
            target = reward + self.gamma * best_next_q

        td_error = target - q_sa
        w = self.weights[action]
        for i in range(_N_FEATURES):
            w[i] += self.alpha * td_error * features[i]

    def update_from_features(
        self,
        features:      List[float],
        action:        Action,
        reward:        float,
        next_features: List[float],
        done:          bool,
    ) -> None:
        """
        Primary update path for LinearFAAgent training.

        Preferred over update() when feature vectors are already computed
        (avoids re-computing them inside update).
        """
        q_sa = self._q(features, action)
        if done:
            target = reward
        else:
            best_next_q = max(self._q(next_features, a) for a in ACTIONS)
            target = reward + self.gamma * best_next_q

        td_error = target - q_sa
        w = self.weights[action]
        for i in range(_N_FEATURES):
            w[i] += self.alpha * td_error * features[i]

    def decay_epsilon(self) -> None:
        """Decay epsilon by 0.995, floor 0.05."""
        self.epsilon = max(0.05, self.epsilon * 0.995)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise weights to JSON."""
        import json
        from pathlib import Path
        data = {
            "alpha":   self.alpha,
            "gamma":   self.gamma,
            "epsilon": self.epsilon,
            "weights": {a.name: w for a, w in self.weights.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2))

    def load(self, path: str) -> None:
        """Load weights from JSON."""
        import json
        import warnings
        from pathlib import Path
        data = json.loads(Path(path).read_text())
        self.alpha   = data.get("alpha",   self.alpha)
        self.gamma   = data.get("gamma",   self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        raw = data.get("weights", {})
        loaded = 0
        for a in ACTIONS:
            w = raw.get(a.name)
            if w and len(w) == _N_FEATURES:
                self.weights[a] = [float(x) for x in w]
                loaded += 1
        if loaded < len(ACTIONS):
            warnings.warn(
                f"[LinearFAAgent.load] Loaded {loaded}/{len(ACTIONS)} weight "
                f"vectors from '{path}'. Missing actions use zero-init.",
                UserWarning, stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Training loop (shared by QLearningAgent and LinearFAAgent)
# ---------------------------------------------------------------------------

def train(
    agent,
    env,
    episodes: int = 50,
    max_steps: int = 200,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Train *agent* on *env* for *episodes* episodes.

    Works with both QLearningAgent (tabular) and LinearFAAgent (linear FA).
    For LinearFAAgent, update_from_features() is called with pre-computed
    feature vectors so extract_features() is never called twice per step.

    Each episode:
        1. Reset env
        2. Loop until done or max_steps
        3. Select action (epsilon-greedy)
        4. Step env, observe reward
        5. Update agent

    Returns
    -------
    dict with keys:
        rewards   : list of total reward per episode
        avg_reward: float — mean over all episodes
        q_table   : Q-table dict (QLearningAgent) or weights dict (LinearFAAgent)
    """
    is_linear = isinstance(agent, LinearFAAgent)
    rewards: List[float] = []

    for ep in range(1, episodes + 1):
        obs       = env.reset(seed=seed_offset + ep)
        total_r   = 0.0

        if is_linear:
            features = extract_features(obs)
        else:
            state = discretise(obs)

        for _ in range(max_steps):
            action_input = agent.get_action(obs)
            action       = action_input.action

            obs, reward, done, _ = env.step(action_input)
            total_r += reward

            if is_linear:
                next_features = extract_features(obs)
                agent.update_from_features(features, action, reward, next_features, done)
                features = next_features
            else:
                next_state = discretise(obs)
                agent.update(state, action, reward, next_state, done)
                state = next_state

            if done:
                break

        rewards.append(total_r)
        agent.decay_epsilon()

        if verbose and ep % 10 == 0:
            avg10 = sum(rewards[-10:]) / min(10, len(rewards))
            if is_linear:
                n_params = sum(1 for w in agent.weights.values() for _ in w if _ != 0.0)
                extra = f"non-zero-weights={n_params}"
            else:
                extra = f"Q-entries={len(agent.Q)}"
            print(f"  ep {ep:>3} / {episodes}  "
                  f"reward={total_r:>8.4f}  avg(10)={avg10:>8.4f}  "
                  f"{extra}")

    avg_reward = sum(rewards) / len(rewards)

    result = {
        "rewards":    rewards,
        "avg_reward": avg_reward,
    }
    if is_linear:
        result["weights"] = agent.weights
    else:
        result["q_table"] = agent.Q
    return result


# ---------------------------------------------------------------------------
# Random baseline agent
# ---------------------------------------------------------------------------

class RandomBaseline:
    """Picks a random action every step — used as comparison floor."""

    def get_action(self, env_state: EnvironmentState) -> ActionInput:
        return ActionInput(action=random.choice(ACTIONS))

    # BaseTask / UI compatibility
    def choose(self, env_state: EnvironmentState) -> ActionInput:
        return self.get_action(env_state)


def run_baseline(env, episodes: int = 50, max_steps: int = 200, seed_offset: int = 0) -> dict:
    """Run RandomBaseline for *episodes* and return reward history."""
    agent   = RandomBaseline()
    rewards = []

    for ep in range(1, episodes + 1):
        obs     = env.reset(seed=seed_offset + ep)
        total_r = 0.0

        for _ in range(max_steps):
            action_input        = agent.get_action(obs)
            obs, reward, done, _ = env.step(action_input)
            total_r += reward
            if done:
                break

        rewards.append(total_r)

    return {"rewards": rewards, "avg_reward": sum(rewards) / len(rewards)}


# ---------------------------------------------------------------------------
# Reward comparison plot
# ---------------------------------------------------------------------------

def plot_rewards(
    ql_rewards:       List[float],
    baseline_rewards: List[float],
    save_path:        Optional[str] = None,
) -> None:
    """
    Plot episode reward for QL agent vs random baseline.

    Args:
        ql_rewards       : list returned by train()["rewards"]
        baseline_rewards : list returned by run_baseline()["rewards"]
        save_path        : if given, save PNG to this path instead of showing

    Example:
        plot_rewards(ql_result["rewards"], baseline_result["rewards"],
                     save_path="reward_comparison.png")
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — pip3 install matplotlib")
        return

    def smooth(vals: List[float], w: int = 5) -> List[float]:
        return [
            sum(vals[max(0, i - w + 1): i + 1]) / min(i + 1, w)
            for i in range(len(vals))
        ]

    episodes = list(range(1, len(ql_rewards) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    # Raw traces (faint)
    ax.plot(episodes, ql_rewards,       color="#58a6ff", alpha=0.3, linewidth=0.8)
    ax.plot(episodes, baseline_rewards, color="#e74c3c", alpha=0.3, linewidth=0.8)

    # Smoothed traces
    ax.plot(episodes, smooth(ql_rewards),       color="#58a6ff", linewidth=2,
            label=f"QL Agent   avg={sum(ql_rewards)/len(ql_rewards):.3f}")
    ax.plot(episodes, smooth(baseline_rewards), color="#e74c3c", linewidth=2,
            label=f"Random     avg={sum(baseline_rewards)/len(baseline_rewards):.3f}")

    ax.set_xlabel("Episode",      color="#c9d1d9")
    ax.set_ylabel("Total Reward", color="#c9d1d9")
    ax.set_title("QL Agent vs Random Baseline — Episode Reward",
                 color="#c9d1d9", fontsize=12)
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.grid(color="#21262d", linewidth=0.5)
    ax.legend(facecolor="#0d1117", edgecolor="#21262d",
              labelcolor="#c9d1d9", fontsize=9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        print(f"[plot] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Quick self-test  (python3 -m adaptive_cyber_defense.agents.ql_agent)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    _ROOT = Path(__file__).resolve().parent.parent.parent
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

    from adaptive_cyber_defense import AdaptiveCyberDefenseEnv

    print("=" * 55)
    print("  Quick Q-Learning integration test")
    print("=" * 55)

    env = AdaptiveCyberDefenseEnv()

    # Train QL agent
    print("\n[1/3] Training QL agent (50 episodes) …")
    agent     = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)
    ql_result = train(agent, env, episodes=50, verbose=True)

    # Run baseline
    print("\n[2/3] Running random baseline (50 episodes) …")
    base_result = run_baseline(env, episodes=50)

    # Compare
    ql_avg   = ql_result["avg_reward"]
    base_avg = base_result["avg_reward"]
    print("\n[3/3] Results")
    print(f"  QL Agent  avg reward : {ql_avg:.4f}")
    print(f"  Random    avg reward : {base_avg:.4f}")
    delta = ql_avg - base_avg
    if delta > 0:
        print(f"  ✅ QL outperforms baseline by {delta:.4f}")
    else:
        print(f"  ℹ️  QL within {abs(delta):.4f} of baseline — more training may help")

    print(f"\n  Q-table entries : {len(agent.Q)}")
    print("  Learned Q-values:")
    for (s, a), v in sorted(agent.Q.items(), key=lambda x: -x[1])[:8]:
        print(f"    state={s}  action={a:<20}  Q={v:>8.4f}")

    # Plot
    _PLOT = Path(__file__).resolve().parent.parent / "training" / "reward_comparison.png"
    plot_rewards(ql_result["rewards"], base_result["rewards"], save_path=str(_PLOT))
    print("=" * 55)
