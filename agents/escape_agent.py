"""EscapeAgent — Phase 2 adversarial agent.

Trained against frozen best-ligand policy to discover escape resistance motifs
and binding-site counter-strategies.  Same REINFORCE structure as ReceptorMutator
with added diversity bonus and mutation magnitude constraints.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from agents.receptor_agent import AdamState
from core.spaces import Box


class EscapeAgent:
    """Adversarial escape agent for Phase 2 robustness refinement."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_beta: float = 0.02,
        sigma_init: float = 0.5,
        sigma_min: float = 0.05,
        mutation_cap: float = 3.0,
        diversity_coeff: float = 0.1,
        baseline_alpha: float = 0.05,
        seed: int | None = None,
    ):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.sigma_min = sigma_min
        self.mutation_cap = mutation_cap
        self.diversity_coeff = diversity_coeff
        self.baseline_alpha = baseline_alpha

        self._rng = np.random.default_rng(seed)

        self.W_mu = self._rng.standard_normal((act_dim, obs_dim)) * 0.01
        self.b_mu = np.zeros(act_dim)
        self.log_sigma = np.full(act_dim, np.log(sigma_init))

        self.w_v = self._rng.standard_normal(obs_dim) * 0.01
        self.b_v = 0.0

        self._baseline_mean = 0.0
        self._adam = AdamState(lr=lr)

        self.action_space = Box(low=-3.0, high=3.0, shape=(act_dim,))

        self._prev_mutations: list[NDArray] = []
        self._train_episodes = 0
        self._escape_motifs: list[dict[str, Any]] = []

    @property
    def sigma(self) -> NDArray:
        return np.exp(self.log_sigma)

    def _mu(self, obs: NDArray) -> NDArray:
        return self.W_mu @ obs + self.b_mu

    def value(self, obs: NDArray) -> float:
        return float(self.w_v @ obs + self.b_v)

    def act(self, obs: NDArray, deterministic: bool = False) -> NDArray:
        mu = self._mu(obs)
        if deterministic:
            action = mu
        else:
            action = mu + self._rng.normal(0, self.sigma, self.act_dim)

        norm = float(np.linalg.norm(action))
        if norm > self.mutation_cap:
            action = action * (self.mutation_cap / norm)
        return self.action_space.clip(action)

    def _diversity_bonus(self, action: NDArray) -> float:
        if not self._prev_mutations:
            return 0.0
        dists = []
        for prev in self._prev_mutations[-20:]:
            cos_dist = 1.0 - np.dot(action, prev) / (
                np.linalg.norm(action) * np.linalg.norm(prev) + 1e-8
            )
            dists.append(cos_dist)
        return float(np.mean(dists)) * self.diversity_coeff

    def _entropy(self) -> float:
        return float(0.5 * np.sum(np.log(2 * np.pi * np.e * self.sigma**2)))

    def learn(
        self,
        obs: NDArray,
        action: NDArray,
        disruption_reward: float,
    ) -> dict[str, float]:
        """REINFORCE update for escape agent."""
        self._prev_mutations.append(action.copy())
        diversity = self._diversity_bonus(action)
        total_reward = disruption_reward + diversity

        self._baseline_mean = (
            self.baseline_alpha * total_reward
            + (1 - self.baseline_alpha) * self._baseline_mean
        )
        advantage = total_reward - self._baseline_mean

        mu = self._mu(obs)
        sigma = self.sigma
        diff = action - mu

        dmu = diff / (sigma**2)
        dW_mu = advantage * np.outer(dmu, obs)
        db_mu = advantage * dmu
        d_log_sigma = advantage * (diff**2 / sigma**2 - 1) + self.entropy_beta

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

        if disruption_reward > 0.7:
            self._escape_motifs.append({
                "episode": self._train_episodes,
                "mutation": action.tolist(),
                "disruption": disruption_reward,
                "diversity": diversity,
            })

        return {
            "advantage": float(advantage),
            "entropy": self._entropy(),
            "sigma_mean": float(np.mean(self.sigma)),
            "diversity_bonus": diversity,
            "disruption_reward": disruption_reward,
            "train_episodes": self._train_episodes,
        }

    @property
    def escape_motifs(self) -> list[dict[str, Any]]:
        return list(self._escape_motifs)

    def get_hard_negatives(self, top_k: int = 5) -> list[NDArray]:
        """Return top-k escape mutations as hard negatives for Phase 1 refinement."""
        sorted_motifs = sorted(
            self._escape_motifs, key=lambda m: m["disruption"], reverse=True
        )
        return [np.array(m["mutation"]) for m in sorted_motifs[:top_k]]

    def get_state(self) -> dict[str, Any]:
        return {
            "W_mu": self.W_mu.copy(),
            "b_mu": self.b_mu.copy(),
            "log_sigma": self.log_sigma.copy(),
            "w_v": self.w_v.copy(),
            "b_v": float(self.b_v),
            "baseline_mean": self._baseline_mean,
            "train_episodes": self._train_episodes,
            "escape_motifs": list(self._escape_motifs),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self.W_mu = state["W_mu"].copy()
        self.b_mu = state["b_mu"].copy()
        self.log_sigma = state["log_sigma"].copy()
        self.w_v = state["w_v"].copy()
        self.b_v = state["b_v"]
        self._baseline_mean = state["baseline_mean"]
        self._train_episodes = state["train_episodes"]
        self._escape_motifs = list(state.get("escape_motifs", []))
