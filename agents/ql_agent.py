"""
Simple Q-Learning agent for the Adaptive Cyber Defense Simulator.

State space  : (threat_level, resource_level)  — 6 states max
Action space : BLOCK_IP | ISOLATE_NODE | PATCH_SYSTEM | RUN_DEEP_SCAN | IGNORE
Q-table      : dict  Q[(state, action)] = float

Usage
-----
    from adaptive_cyber_defense.agents.ql_agent import QLearningAgent, train
    from adaptive_cyber_defense import AdaptiveCyberDefenseEnv

    env    = AdaptiveCyberDefenseEnv()
    agent  = QLearningAgent()
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

def discretise(env_state: EnvironmentState) -> Tuple[str, str]:
    """
    Compress continuous env state into a 2-key tuple.

    threat_level  : "low" (<0.33) | "medium" (<0.66) | "high" (>=0.66)
    resource_level: "low" (<0.50) | "high" (>=0.50)
    """
    sev = env_state.threat_severity
    if sev < 0.33:
        threat_level = "low"
    elif sev < 0.66:
        threat_level = "medium"
    else:
        threat_level = "high"

    resource_level = "low" if env_state.resource_availability < 0.50 else "high"

    return (threat_level, resource_level)


# ---------------------------------------------------------------------------
# Q-Learning agent
# ---------------------------------------------------------------------------

class QLearningAgent:
    """
    Tabular Q-Learning agent.

    Q-table: Q[(state_tuple, action_name)] = float, initialised to 0.

    Hyperparameters
    ---------------
    alpha   : learning rate            (default 0.1)
    gamma   : discount factor          (default 0.9)
    epsilon : exploration probability  (default 0.2, fixed)
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
        """Load Q-table from JSON."""
        data = json.loads(Path(path).read_text())
        self.alpha   = data.get("alpha",   self.alpha)
        self.gamma   = data.get("gamma",   self.gamma)
        self.epsilon = data.get("epsilon", self.epsilon)
        self.Q = {}
        for key_str, val in data.get("Q", {}).items():
            # Reconstruct tuple key from string representation
            # Format: "((threat_level, resource_level), action_name)"
            import ast
            try:
                key = ast.literal_eval(key_str)
                self.Q[key] = float(val)
            except Exception:
                pass  # skip malformed entries


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    agent:    QLearningAgent,
    env,
    episodes: int = 50,
    max_steps: int = 200,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Train *agent* on *env* for *episodes* episodes.

    Each episode:
        1. Reset env
        2. Loop until done or max_steps
        3. Select action (epsilon-greedy)
        4. Step env, observe reward
        5. Update Q-table

    Returns
    -------
    dict with keys:
        rewards   : list of total reward per episode
        avg_reward: float — mean over all episodes
        q_table   : the trained Q-table dict
    """
    rewards: List[float] = []

    for ep in range(1, episodes + 1):
        obs       = env.reset(seed=seed_offset + ep)
        state     = discretise(obs)
        total_r   = 0.0

        for _ in range(max_steps):
            action_input = agent.get_action(obs)
            action       = action_input.action

            obs, reward, done, _ = env.step(action_input)
            next_state           = discretise(obs)

            agent.update(state, action, reward, next_state, done)

            state   = next_state
            total_r += reward

            if done:
                break

        rewards.append(total_r)

        if verbose and ep % 10 == 0:
            avg10 = sum(rewards[-10:]) / min(10, len(rewards))
            print(f"  ep {ep:>3} / {episodes}  "
                  f"reward={total_r:>8.4f}  avg(10)={avg10:>8.4f}  "
                  f"Q-entries={len(agent.Q)}")

    avg_reward = sum(rewards) / len(rewards)

    return {
        "rewards":    rewards,
        "avg_reward": avg_reward,
        "q_table":    agent.Q,
    }


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
