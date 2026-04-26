"""ReceptorMutator — episode-level REINFORCE agent (Agent_B).

Policy: π(δ|s_agg) = N(μ, σ_per_dim)
Episode-level learner with per-dimension sigma and Adam optimizer in numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.spaces import Box


class AdamState:
    """Minimal Adam optimizer state for numpy arrays."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m: dict[str, NDArray] = {}
        self.v: dict[str, NDArray] = {}
        self.t = 0

    def step(self, params: dict[str, NDArray], grads: dict[str, NDArray]) -> dict[str, NDArray]:
        self.t += 1
        updated = {}
        for key in grads:
            if key not in self.m:
                self.m[key] = np.zeros_like(grads[key])
                self.v[key] = np.zeros_like(grads[key])
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            updated[key] = params[key] + self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return updated


class ReceptorMutator:
    """Episode-level REINFORCE agent with per-dimension Gaussian policy."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_beta: float = 0.01,
        sigma_init: float = 0.5,
        sigma_min: float = 0.05,
        mutation_cap: float = 3.0,
        baseline_alpha: float = 0.05,
        diversity_coeff: float = 0.05,
        seed: int | None = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.sigma_min = sigma_min
        self.mutation_cap = mutation_cap
        self.baseline_alpha = baseline_alpha
        self.diversity_coeff = diversity_coeff

        self._rng = np.random.default_rng(seed)

        self.W_mu = self._rng.standard_normal((act_dim, obs_dim)) * 0.01
        self.b_mu = np.zeros(act_dim)
        self.log_sigma = np.full(act_dim, np.log(sigma_init))

        # Value baseline: linear
        self.w_v = self._rng.standard_normal(obs_dim) * 0.01
        self.b_v = 0.0

        self._baseline_mean = 0.0

        self._adam = AdamState(lr=lr)

        bio_bound = 3.0
        self.action_space = Box(low=-bio_bound, high=bio_bound, shape=(act_dim,))

        self._prev_actions: list[NDArray] = []
        self._train_episodes = 0

        self._llm_bias: NDArray | None = None
        self._llm_weight: float = 0.0

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
            action = mu
        else:
            noise = self._rng.normal(0, self.sigma, self.act_dim)
            action = mu + noise

        # L2 norm cap
        norm = float(np.linalg.norm(action))
        if norm > self.mutation_cap:
            action = action * (self.mutation_cap / norm)

        return self.action_space.clip(action)

    def inject_llm_bias(self, bias_vector: NDArray, weight: float = 0.25) -> None:
        self._llm_bias = np.asarray(bias_vector, dtype=np.float64)
        self._llm_weight = weight

    def clear_llm_bias(self) -> None:
        self._llm_bias = None
        self._llm_weight = 0.0

    def _diversity_bonus(self, action: NDArray) -> float:
        if not self._prev_actions:
            return 0.0
        dists = []
        for prev in self._prev_actions[-10:]:
            cos_sim = np.dot(action, prev) / (
                np.linalg.norm(action) * np.linalg.norm(prev) + 1e-8
            )
            dists.append(1.0 - cos_sim)
        return float(np.mean(dists)) * self.diversity_coeff

    def _entropy(self) -> float:
        return float(0.5 * np.sum(np.log(2 * np.pi * np.e * self.sigma**2)))

    def learn(
        self, obs: NDArray, action: NDArray, episode_reward: float
    ) -> dict[str, float]:
        """REINFORCE update from a single episode."""
        self._prev_actions.append(action.copy())
        diversity = self._diversity_bonus(action)
        total_reward = episode_reward + diversity

        # EMA baseline
        self._baseline_mean = (
            self.baseline_alpha * total_reward
            + (1 - self.baseline_alpha) * self._baseline_mean
        )
        advantage = total_reward - self._baseline_mean

        mu = self._mu(obs)
        sigma = self.sigma
        diff = action - mu

        # Policy gradients
        dmu = diff / (sigma**2)
        dW_mu = advantage * np.outer(dmu, obs)
        db_mu = advantage * dmu

        dsig = (diff**2 / sigma**2 - 1)
        d_log_sigma = advantage * dsig + self.entropy_beta

        # Value gradient
        v = self.value(obs)
        v_err = total_reward - v
        dw_v = v_err * obs
        db_v_val = v_err

        grads = {
            "W_mu": dW_mu,
            "b_mu": db_mu,
            "log_sigma": d_log_sigma,
            "w_v": dw_v,
            "b_v": np.array([db_v_val]),
        }
        params = {
            "W_mu": self.W_mu,
            "b_mu": self.b_mu,
            "log_sigma": self.log_sigma,
            "w_v": self.w_v,
            "b_v": np.array([self.b_v]),
        }

        updated = self._adam.step(params, grads)
        self.W_mu = updated["W_mu"]
        self.b_mu = updated["b_mu"]
        self.log_sigma = np.maximum(updated["log_sigma"], np.log(self.sigma_min))
        self.w_v = updated["w_v"]
        self.b_v = float(updated["b_v"][0])

        self._train_episodes += 1

        return {
            "advantage": float(advantage),
            "entropy": self._entropy(),
            "sigma_mean": float(np.mean(self.sigma)),
            "diversity_bonus": diversity,
            "baseline": self._baseline_mean,
            "train_episodes": self._train_episodes,
        }

    def get_state(self) -> dict[str, Any]:
        return {
            "W_mu": self.W_mu.copy(),
            "b_mu": self.b_mu.copy(),
            "log_sigma": self.log_sigma.copy(),
            "w_v": self.w_v.copy(),
            "b_v": float(self.b_v),
            "baseline_mean": self._baseline_mean,
            "train_episodes": self._train_episodes,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.W_mu = state["W_mu"].copy()
        self.b_mu = state["b_mu"].copy()
        self.log_sigma = state["log_sigma"].copy()
        self.w_v = state["w_v"].copy()
        self.b_v = state["b_v"]
        self._baseline_mean = state["baseline_mean"]
        self._train_episodes = state["train_episodes"]
