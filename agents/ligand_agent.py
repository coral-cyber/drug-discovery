"""LigandDesigner — step-level actor-critic agent (Agent_A).

Policy: π(a|s) = N(μ(s), σ)
Value: V(s) linear on features
Advantage normalization, entropy bonus, potential-based shaping.
All RL maths in numpy — no torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.spaces import Box


@dataclass
class Transition:
    obs: NDArray
    action: NDArray
    reward: float
    next_obs: NDArray
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


class LigandDesigner:
    """Step-level learner with Gaussian policy and linear value baseline."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        lr_policy: float = 3e-3,
        lr_value: float = 1e-2,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        sigma_init: float = 1.0,
        sigma_min: float = 0.1,
        sigma_decay: float = 0.999,
        seed: int | None = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

        self._rng = np.random.default_rng(seed)

        # Policy params: μ = W_mu @ obs + b_mu
        self.W_mu = self._rng.standard_normal((act_dim, obs_dim)) * 0.01
        self.b_mu = np.zeros(act_dim)
        self.log_sigma = np.full(act_dim, np.log(sigma_init))

        # Value params: V = w_v @ obs + b_v
        self.w_v = self._rng.standard_normal(obs_dim) * 0.01
        self.b_v = 0.0

        self._buffer: list[Transition] = []

        self._llm_bias: NDArray | None = None
        self._llm_weight: float = 0.0

        self.action_space = Box(low=-3.0, high=3.0, shape=(act_dim,))

        self._train_steps = 0

    @property
    def sigma(self) -> NDArray:
        return np.exp(self.log_sigma)

    def _mu(self, obs: NDArray) -> NDArray:
        return self.W_mu @ obs + self.b_mu

    def value(self, obs: NDArray) -> float:
        return float(self.w_v @ obs + self.b_v)

    def act(self, obs: NDArray, deterministic: bool = False) -> NDArray:
        mu = self._mu(obs)
        if self._llm_bias is not None:
            mu = (1 - self._llm_weight) * mu + self._llm_weight * self._llm_bias
        if deterministic:
            return self.action_space.clip(mu)
        noise = self._rng.normal(0, self.sigma, self.act_dim)
        action = mu + noise
        return self.action_space.clip(action)

    def inject_llm_bias(self, bias_vector: NDArray, weight: float = 0.25) -> None:
        self._llm_bias = np.asarray(bias_vector, dtype=np.float64)
        self._llm_weight = weight

    def clear_llm_bias(self) -> None:
        self._llm_bias = None
        self._llm_weight = 0.0

    def store(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def clear_buffer(self) -> None:
        self._buffer.clear()

    def _compute_returns(self) -> NDArray:
        """Monte Carlo discounted returns."""
        rewards = [t.reward for t in self._buffer]
        G = np.zeros(len(rewards))
        running = 0.0
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1 - int(self._buffer[t].done))
            G[t] = running
        return G

    def _log_prob(self, action: NDArray, mu: NDArray) -> float:
        """Log probability under diagonal Gaussian."""
        sigma = self.sigma
        return float(
            -0.5 * np.sum(((action - mu) / sigma) ** 2)
            - np.sum(np.log(sigma))
            - 0.5 * self.act_dim * np.log(2 * np.pi)
        )

    def _entropy(self) -> float:
        """Gaussian entropy: H = 0.5 * sum(log(2πeσ²))."""
        return float(0.5 * np.sum(np.log(2 * np.pi * np.e * self.sigma**2)))

    def learn(self) -> dict[str, float]:
        """Single-episode policy gradient + value update."""
        if len(self._buffer) == 0:
            return {}

        returns = self._compute_returns()
        obs_batch = np.array([t.obs for t in self._buffer])
        act_batch = np.array([t.action for t in self._buffer])
        values = np.array([self.value(o) for o in obs_batch])

        advantages = returns - values
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - np.mean(advantages)) / adv_std

        # Policy gradient
        dW_mu = np.zeros_like(self.W_mu)
        db_mu = np.zeros_like(self.b_mu)
        d_log_sigma = np.zeros_like(self.log_sigma)

        sigma = self.sigma
        for i, t in enumerate(self._buffer):
            mu = self._mu(t.obs)
            diff = t.action - mu
            # ∂log_π/∂μ = (a - μ)/σ²
            dmu = diff / (sigma**2)
            # ∂log_π/∂log_σ = (a - μ)²/σ² - 1
            dsig = (diff**2) / (sigma**2) - 1

            dW_mu += advantages[i] * np.outer(dmu, t.obs)
            db_mu += advantages[i] * dmu
            d_log_sigma += advantages[i] * dsig

        n = len(self._buffer)
        self.W_mu += self.lr_policy * dW_mu / n
        self.b_mu += self.lr_policy * db_mu / n
        self.log_sigma += self.lr_policy * (d_log_sigma / n + self.entropy_beta)

        # Sigma decay
        self.log_sigma = np.maximum(
            self.log_sigma + np.log(self.sigma_decay),
            np.log(self.sigma_min),
        )

        # Value update (MSE gradient)
        for i, t in enumerate(self._buffer):
            v = self.value(t.obs)
            err = returns[i] - v
            self.w_v += self.lr_value * err * t.obs / n
            self.b_v += self.lr_value * err / n

        self._train_steps += 1

        loss_policy = float(-np.mean(advantages * np.array([
            self._log_prob(act_batch[i], self._mu(obs_batch[i]))
            for i in range(n)
        ])))
        loss_value = float(np.mean((returns - values) ** 2))
        entropy = self._entropy()

        self.clear_buffer()

        return {
            "loss_policy": loss_policy,
            "loss_value": loss_value,
            "entropy": entropy,
            "sigma_mean": float(np.mean(self.sigma)),
            "train_steps": self._train_steps,
        }

    def get_probe_readings(
        self, receptor: NDArray, n_probes: int, oracle: Any
    ) -> NDArray:
        """Generate binding probe readings for the receptor env."""
        readings = []
        for _ in range(n_probes):
            ligand = self.act(
                np.concatenate([receptor, [0.5, 0.0]]),
                deterministic=False,
            )
            readings.append(oracle.score(ligand, receptor))
        return np.array(readings)

    def get_state(self) -> dict[str, Any]:
        return {
            "W_mu": self.W_mu.copy(),
            "b_mu": self.b_mu.copy(),
            "log_sigma": self.log_sigma.copy(),
            "w_v": self.w_v.copy(),
            "b_v": float(self.b_v),
            "train_steps": self._train_steps,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.W_mu = state["W_mu"].copy()
        self.b_mu = state["b_mu"].copy()
        self.log_sigma = state["log_sigma"].copy()
        self.w_v = state["w_v"].copy()
        self.b_v = state["b_v"]
        self._train_steps = state["train_steps"]
