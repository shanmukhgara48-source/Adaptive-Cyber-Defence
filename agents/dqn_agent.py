"""
DQN Agent — pure NumPy implementation (no PyTorch dependency).

Implements Deep Q-Network with:
  - 3-layer MLP (input → 128 → 128 → n_actions) with ReLU hidden units
  - Experience replay buffer
  - Target network periodically hard-updated from online network
  - Adam optimizer (implemented manually)
  - Epsilon-greedy exploration with linear decay

State representation
--------------------
Uses EnvironmentState.to_vector() — stable 54-dim float32 vector:
  8 nodes × 6 features + 6 global scalars
  Nodes (sorted): db-01, fw-01, router-01, srv-db, srv-web, ws-01, ws-02, ws-03
  Per-node : is_compromised, is_isolated, health, patch_level, criticality, vulnerability
  Global   : threat_severity, network_load, resource_availability,
             detection_confidence, time_step/100, episode_score

Action space (5 discrete actions, index → Action enum)
-------------------------------------------------------
  0 → BLOCK_IP        1 → ISOLATE_NODE    2 → PATCH_SYSTEM
  3 → RUN_DEEP_SCAN   4 → IGNORE

Target-node selection
---------------------
For non-IGNORE actions the agent picks the current node of the highest
effective-severity active threat. If no threats are visible it falls back
to the first compromised node, then to the highest-criticality asset.

Usage (via BaseTask.run — choose() interface)
---------------------------------------------
    from adaptive_cyber_defense.agents.dqn_agent import DQNAgent
    agent = DQNAgent.load("agents/dqn_weights.json")
    result = EasyTask().run(agent, seed=42)

Usage (training loop)
---------------------
    agent = DQNAgent()
    for ep in range(n_episodes):
        state = env.reset(seed=ep)
        done = False
        while not done:
            action_input, action_idx = agent.select_action(state)
            next_state, reward, done, _ = env.step(action_input)
            agent.store(state, action_idx, reward, next_state, done)
            agent.train_step()
            state = next_state
        agent.maybe_update_target(ep)
        agent.decay_epsilon(ep, n_episodes)
"""

from __future__ import annotations

import json
import random as _stdlib_random
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..models.state import EnvironmentState

from ..models.action import Action, ActionInput


# ---------------------------------------------------------------------------
# Constants — must stay in sync with config/dqn_config.yaml
# ---------------------------------------------------------------------------

STATE_DIM: int = 54          # EnvironmentState.to_vector() length
N_ACTIONS: int = 5           # BLOCK_IP | ISOLATE_NODE | PATCH_SYSTEM | RUN_DEEP_SCAN | IGNORE

# Ordered list of actions indexed 0-4; must match config/dqn_config.yaml mapping
_ACTIONS: List[Action] = [
    Action.BLOCK_IP,       # 0
    Action.ISOLATE_NODE,   # 1
    Action.PATCH_SYSTEM,   # 2
    Action.RUN_DEEP_SCAN,  # 3
    Action.IGNORE,         # 4
]


# ---------------------------------------------------------------------------
# Minimal MLP — forward + backward in NumPy
# ---------------------------------------------------------------------------

class _NumpyMLP:
    """
    3-layer MLP with ReLU hidden units and linear output layer.

    Architecture: state_dim → h1 → h2 → n_actions

    Uses He initialisation and a manual Adam optimiser.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.weights: List[np.ndarray] = []
        self.biases:  List[np.ndarray] = []

        for i in range(len(layer_sizes) - 1):
            fan_in  = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            std = np.sqrt(2.0 / fan_in)   # He init
            W = rng.standard_normal((fan_in, fan_out)).astype(np.float32) * std
            b = np.zeros(fan_out, dtype=np.float32)
            self.weights.append(W)
            self.biases.append(b)

        # Adam first and second moment estimates
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._t: int = 0   # Adam step counter

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass — no cache stored. Accepts (N, D) or (D,) inputs."""
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W + b
            if i < len(self.weights) - 1:
                h = np.maximum(0.0, h)   # ReLU
        return h

    # ── Training ─────────────────────────────────────────────────────────────

    def forward_with_cache(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
        """Forward pass that also stores activations needed for backprop."""
        cache: List[Tuple[np.ndarray, np.ndarray]] = []
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W + b
            cache.append((h, z))
            if i < len(self.weights) - 1:
                h = np.maximum(0.0, z)
            else:
                h = z   # linear output
        return h, cache

    def backward(
        self,
        cache: List[Tuple[np.ndarray, np.ndarray]],
        dout: np.ndarray,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Backprop. Returns list of (dW, db) per layer, shallowest first."""
        grads: List[Tuple[np.ndarray, np.ndarray]] = []
        d = dout
        for i in reversed(range(len(self.weights))):
            h_in, z = cache[i]
            if i < len(self.weights) - 1:
                d = d * (z > 0)   # ReLU gradient
            dW = h_in.T @ d
            db = d.sum(axis=0)
            d  = d @ self.weights[i].T
            grads.insert(0, (dW, db))
        return grads

    def adam_update(
        self,
        grads: List[Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """In-place Adam update."""
        self._t += 1
        bc1 = 1.0 - beta1 ** self._t
        bc2 = 1.0 - beta2 ** self._t

        for i, (dW, db) in enumerate(grads):
            # Weights
            self._m_w[i] = beta1 * self._m_w[i] + (1.0 - beta1) * dW
            self._v_w[i] = beta2 * self._v_w[i] + (1.0 - beta2) * (dW * dW)
            self.weights[i] -= lr * (self._m_w[i] / bc1) / (np.sqrt(self._v_w[i] / bc2) + eps)

            # Biases
            self._m_b[i] = beta1 * self._m_b[i] + (1.0 - beta1) * db
            self._v_b[i] = beta2 * self._v_b[i] + (1.0 - beta2) * (db * db)
            self.biases[i] -= lr * (self._m_b[i] / bc1) / (np.sqrt(self._v_b[i] / bc2) + eps)

    # ── Weight transfer ───────────────────────────────────────────────────────

    def copy_from(self, other: "_NumpyMLP") -> None:
        """Hard-copy weights from another network (target network update)."""
        for i in range(len(self.weights)):
            np.copyto(self.weights[i], other.weights[i])
            np.copyto(self.biases[i],  other.biases[i])

    # ── Persistence ──────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "weights": [w.tolist() for w in self.weights],
            "biases":  [b.tolist() for b in self.biases],
        }

    def load_state_dict(self, d: dict) -> None:
        self.weights = [np.array(w, dtype=np.float32) for w in d["weights"]]
        self.biases  = [np.array(b, dtype=np.float32) for b in d["biases"]]
        # Reset Adam state to match new weight shapes
        self._m_w = [np.zeros_like(w) for w in self.weights]
        self._v_w = [np.zeros_like(w) for w in self.weights]
        self._m_b = [np.zeros_like(b) for b in self.biases]
        self._v_b = [np.zeros_like(b) for b in self.biases]
        self._t = 0


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    """Circular experience replay buffer backed by a collections.deque."""

    def __init__(self, capacity: int) -> None:
        self._buf: Deque[Tuple] = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action_idx: int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ) -> None:
        self._buf.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        batch = _stdlib_random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int32),
            np.array(rewards,     dtype=np.float32),
            np.stack(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buf)


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    Deep Q-Network agent for the Adaptive Cyber Defense Simulator.

    Parameters
    ----------
    state_dim          : input dimension (default: STATE_DIM = 54)
    n_actions          : number of discrete actions (default: N_ACTIONS = 5)
    hidden_sizes       : hidden layer widths (default: [128, 128])
    lr                 : Adam learning rate (default: 0.001)
    gamma              : discount factor (default: 0.95)
    epsilon_start      : initial exploration rate (default: 1.0)
    epsilon_end        : minimum exploration rate (default: 0.05)
    epsilon_decay_eps  : episodes over which epsilon decays linearly (default: 150)
    batch_size         : minibatch size for gradient updates (default: 64)
    buffer_capacity    : max replay buffer entries (default: 10000)
    target_update_freq : episodes between target network syncs (default: 20)
    min_buffer_steps   : steps before first gradient update (default: 256)
    seed               : RNG seed (default: 0)
    """

    def __init__(
        self,
        state_dim:          int   = STATE_DIM,
        n_actions:          int   = N_ACTIONS,
        hidden_sizes:       List[int] = None,
        lr:                 float = 0.001,
        gamma:              float = 0.95,
        epsilon_start:      float = 1.0,
        epsilon_end:        float = 0.05,
        epsilon_decay_eps:  int   = 150,
        batch_size:         int   = 64,
        buffer_capacity:    int   = 10000,
        target_update_freq: int   = 20,
        min_buffer_steps:   int   = 256,
        seed:               int   = 0,
    ) -> None:
        if hidden_sizes is None:
            hidden_sizes = [128, 128]

        self.state_dim          = state_dim
        self.n_actions          = n_actions
        self.lr                 = lr
        self.gamma              = gamma
        self.epsilon            = epsilon_start
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay_eps  = epsilon_decay_eps
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_steps   = min_buffer_steps

        layer_sizes = [state_dim] + hidden_sizes + [n_actions]
        self._online = _NumpyMLP(layer_sizes, seed=seed)
        self._target = _NumpyMLP(layer_sizes, seed=seed + 1)
        self._target.copy_from(self._online)

        self._buffer = _ReplayBuffer(buffer_capacity)
        self._rng    = np.random.default_rng(seed)
        self._step_count = 0

    # ── State flattening ─────────────────────────────────────────────────────

    @staticmethod
    def flatten(state: "EnvironmentState") -> np.ndarray:
        """Convert EnvironmentState → fixed-size float32 vector via to_vector()."""
        return np.array(state.to_vector(), dtype=np.float32)

    # ── Target node selection ─────────────────────────────────────────────────

    @staticmethod
    def _pick_target(state: "EnvironmentState") -> Optional[str]:
        """Return the most important node to act on for the current state."""
        if state.active_threats:
            worst = max(state.active_threats, key=lambda t: t.effective_severity())
            return worst.current_node
        if state.compromised_nodes:
            return state.compromised_nodes[0]
        if state.assets:
            return max(state.assets.keys(), key=lambda k: state.assets[k].criticality)
        return None

    def _action_to_input(self, action_idx: int, state: "EnvironmentState") -> ActionInput:
        """Map action index → ActionInput with a sensible target node."""
        action = _ACTIONS[action_idx]
        if action == Action.IGNORE:
            return ActionInput(action=Action.IGNORE)
        return ActionInput(action=action, target_node=self._pick_target(state))

    # ── Action selection ──────────────────────────────────────────────────────

    def select_action(
        self,
        state: "EnvironmentState",
        greedy: bool = False,
    ) -> Tuple[ActionInput, int]:
        """
        Epsilon-greedy action selection.

        Args:
            state:  Current EnvironmentState from env.reset() or env.step().
            greedy: If True, always exploit (epsilon=0). Use for evaluation.

        Returns:
            (ActionInput, action_idx)  — pass the ActionInput to env.step(),
            store action_idx in the replay buffer.
        """
        vec = self.flatten(state)
        if not greedy and self._rng.random() < self.epsilon:
            action_idx = int(self._rng.integers(self.n_actions))
        else:
            q_vals     = self._online.predict(vec[np.newaxis, :])[0]
            action_idx = int(np.argmax(q_vals))

        return self._action_to_input(action_idx, state), action_idx

    def choose(self, state: "EnvironmentState") -> ActionInput:
        """
        Greedy action selection with no action index returned.
        This signature matches BaseTask.run(agent) so the agent can be
        passed directly to task.run() for evaluation.
        """
        action_input, _ = self.select_action(state, greedy=True)
        return action_input

    # ── Replay buffer ─────────────────────────────────────────────────────────

    def store(
        self,
        state:      "EnvironmentState",
        action_idx: int,
        reward:     float,
        next_state: "EnvironmentState",
        done:       bool,
    ) -> None:
        """Push one (s, a, r, s', done) transition into the replay buffer."""
        self._buffer.push(
            self.flatten(state),
            action_idx,
            float(reward),
            self.flatten(next_state),
            bool(done),
        )
        self._step_count += 1

    # ── Training ─────────────────────────────────────────────────────────────

    def train_step(self) -> Optional[float]:
        """
        Sample one minibatch and perform one gradient update.

        Returns the mean Huber loss, or None if the buffer is too small.
        """
        if len(self._buffer) < self.min_buffer_steps:
            return None
        if len(self._buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self._buffer.sample(self.batch_size)

        # Target Q-values: r + γ · max_a' Q_target(s', a')  (Bellman target)
        next_q     = self._target.predict(next_states)   # (batch, n_actions)
        max_next_q = next_q.max(axis=1)                  # (batch,)
        targets    = rewards + self.gamma * max_next_q * (1.0 - dones)

        # Online forward pass
        q_out, cache = self._online.forward_with_cache(states)  # (batch, n_actions)

        # Build error tensor: only propagate gradient for taken action
        errors = np.zeros_like(q_out)
        batch_idx = np.arange(self.batch_size)
        delta = q_out[batch_idx, actions] - targets

        # Huber loss gradient (δ=1)
        grad_delta = np.where(np.abs(delta) <= 1.0, delta, np.sign(delta))
        errors[batch_idx, actions] = grad_delta / self.batch_size

        # Backward + Adam update
        grads = self._online.backward(cache, errors)
        self._online.adam_update(grads, lr=self.lr)

        loss = float(np.mean(0.5 * delta ** 2 * (np.abs(delta) <= 1.0)
                             + (np.abs(delta) - 0.5) * (np.abs(delta) > 1.0)))
        return loss

    def maybe_update_target(self, episode: int) -> bool:
        """Sync target ← online weights every target_update_freq episodes."""
        if episode > 0 and episode % self.target_update_freq == 0:
            self._target.copy_from(self._online)
            return True
        return False

    def decay_epsilon(self, episode: int, total_episodes: int = None) -> None:
        """
        Linearly decay epsilon from epsilon_start to epsilon_end
        over epsilon_decay_eps episodes, then hold at epsilon_end.
        """
        if episode >= self.epsilon_decay_eps:
            self.epsilon = self.epsilon_end
        else:
            frac = episode / max(1, self.epsilon_decay_eps)
            self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save weights and hyperparameters to a JSON file."""
        data = {
            "state_dim":          self.state_dim,
            "n_actions":          self.n_actions,
            "lr":                 self.lr,
            "gamma":              self.gamma,
            "epsilon":            self.epsilon,
            "epsilon_start":      self.epsilon_start,
            "epsilon_end":        self.epsilon_end,
            "epsilon_decay_eps":  self.epsilon_decay_eps,
            "batch_size":         self.batch_size,
            "target_update_freq": self.target_update_freq,
            "min_buffer_steps":   self.min_buffer_steps,
            "online_weights":     self._online.state_dict(),
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str) -> "DQNAgent":
        """Load a saved DQN agent from a JSON file."""
        data = json.loads(Path(path).read_text())
        agent = cls(
            state_dim=          data.get("state_dim",          STATE_DIM),
            n_actions=          data.get("n_actions",          N_ACTIONS),
            lr=                 data.get("lr",                 0.001),
            gamma=              data.get("gamma",              0.95),
            epsilon_start=      data.get("epsilon_start",      1.0),
            epsilon_end=        data.get("epsilon_end",        0.05),
            epsilon_decay_eps=  data.get("epsilon_decay_eps",  150),
            batch_size=         data.get("batch_size",         64),
            target_update_freq= data.get("target_update_freq", 20),
            min_buffer_steps=   data.get("min_buffer_steps",   256),
        )
        agent.epsilon = data.get("epsilon", agent.epsilon_end)
        agent._online.load_state_dict(data["online_weights"])
        agent._target.copy_from(agent._online)
        return agent
