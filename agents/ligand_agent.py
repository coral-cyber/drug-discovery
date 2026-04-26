from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.spaces import Box
from core.utils import EpisodeMemory, engineered_features, gaussian_entropy, gaussian_log_prob, monte_carlo_returns, normalize_advantages


@dataclass
class LigandUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    sigma_mean: float
    reward_mean: float


class LigandDesignerAgent:
    env_id = "LigandDesigner-v1"

    def __init__(
        self,
        obs_dim: int,
        action_space: Box,
        gamma: float = 0.99,
        lr_policy: float = 0.03,
        lr_value: float = 0.05,
        value_alpha: float = 0.2,
        entropy_beta: float = 0.02,
        sigma_init: float = 0.6,
        sigma_decay: float = 0.995,
        min_sigma: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.gamma = gamma
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.value_alpha = value_alpha
        self.entropy_beta = entropy_beta
        self.sigma_decay = sigma_decay
        self.min_sigma = min_sigma
        self.rng = np.random.default_rng(seed)
        self.policy_w = self.rng.normal(0.0, 0.08, size=(self.obs_dim, self.action_dim))
        self.policy_b = np.zeros(self.action_dim, dtype=np.float64)
        self.value_w = self.rng.normal(0.0, 0.05, size=(2 * self.obs_dim + 1,))
        self.sigma = np.full(self.action_dim, sigma_init, dtype=np.float64)
        self.memory = EpisodeMemory()
        self.last_bias = np.zeros(self.action_dim, dtype=np.float64)
        self.hard_negatives: list[np.ndarray] = []

    def policy_mean(self, obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.float64) @ self.policy_w + self.policy_b

    def value(self, obs: np.ndarray) -> float:
        return float(engineered_features(obs) @ self.value_w)

    def inject_llm_bias(self, vector: np.ndarray, weight: float = 0.25) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        self.last_bias = weight * arr
        return self.last_bias.copy()

    def select_action(self, obs: np.ndarray, llm_bias: np.ndarray | None = None, llm_weight: float = 0.25, deterministic: bool = False) -> dict[str, Any]:
        observation = np.asarray(obs, dtype=np.float64)
        mean = self.policy_mean(observation)
        raw_action = mean.copy() if deterministic else mean + self.rng.normal(0.0, self.sigma, size=self.action_dim)
        raw_action = self.action_space.clip(raw_action)
        bias = np.zeros(self.action_dim, dtype=np.float64)
        if llm_bias is not None:
            bias = self.inject_llm_bias(llm_bias, weight=llm_weight)
        biased_action = self.action_space.clip((1.0 - llm_weight) * raw_action + bias)
        log_prob = gaussian_log_prob(biased_action, mean, self.sigma)
        entropy = gaussian_entropy(self.sigma) * self.entropy_beta
        return {
            "mean": mean,
            "raw_action": raw_action,
            "action": biased_action,
            "log_prob": log_prob,
            "entropy": entropy,
            "sigma": self.sigma.copy(),
            "bias": bias.copy(),
        }

    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, mean: np.ndarray, entropy: float, info: dict[str, Any]) -> None:
        self.memory.observations.append(np.asarray(obs, dtype=np.float64))
        self.memory.actions.append(np.asarray(action, dtype=np.float64))
        self.memory.rewards.append(float(reward))
        self.memory.next_observations.append(np.asarray(next_obs, dtype=np.float64))
        self.memory.means.append(np.asarray(mean, dtype=np.float64))
        self.memory.entropies.append(float(entropy))
        self.memory.infos.append(info)

    def probe_candidates(self, receptor_vector: np.ndarray, count: int = 4) -> list[np.ndarray]:
        receptor_arr = np.asarray(receptor_vector, dtype=np.float64)
        probes: list[np.ndarray] = []
        for index in range(count):
            obs = np.concatenate([receptor_arr, np.array([index / max(count, 1), 0.0], dtype=np.float64)])
            probes.append(self.action_space.clip(self.policy_mean(obs)))
        return probes

    def add_hard_negative(self, mutation: np.ndarray) -> None:
        self.hard_negatives.append(np.asarray(mutation, dtype=np.float64))

    def ensure_entropy_floor(self, min_entropy: float) -> bool:
        current_entropy = gaussian_entropy(self.sigma)
        if current_entropy >= min_entropy:
            return False
        self.sigma = np.maximum(self.sigma * 1.1, self.min_sigma)
        return True

    def learn(self) -> LigandUpdateStats:
        if not self.memory.rewards:
            return LigandUpdateStats(0.0, 0.0, gaussian_entropy(self.sigma), float(self.sigma.mean()), 0.0)

        returns = monte_carlo_returns(self.memory.rewards, self.gamma)
        values = np.array([self.value(obs) for obs in self.memory.observations], dtype=np.float64)
        advantages = normalize_advantages(returns - values)
        policy_loss = 0.0
        value_loss = 0.0
        for obs, action, mean, ret, adv in zip(self.memory.observations, self.memory.actions, self.memory.means, returns, advantages):
            sigma_sq = self.sigma * self.sigma + 1e-8
            grad_mean = (action - mean) / sigma_sq
            self.policy_w += self.lr_policy * np.outer(obs, grad_mean * adv)
            self.policy_b += self.lr_policy * grad_mean * adv
            policy_loss += -gaussian_log_prob(action, mean, self.sigma) * float(adv)
            features = engineered_features(obs)
            value_prediction = float(features @ self.value_w)
            value_error = value_prediction - float(ret)
            self.value_w -= self.lr_value * 2.0 * self.value_alpha * value_error * features
            value_loss += value_error * value_error

        if self.hard_negatives:
            recent_negatives = self.hard_negatives[-5:]
            for obs, action in zip(self.memory.observations, self.memory.actions):
                for neg in recent_negatives:
                    neg_arr = np.asarray(neg, dtype=np.float64)
                    if neg_arr.shape == action.shape:
                        similarity = float(np.dot(action, neg_arr) / (np.linalg.norm(action) * np.linalg.norm(neg_arr) + 1e-8))
                        if similarity > 0.5:
                            repulsion = 0.02 * similarity * (action - neg_arr) / (np.linalg.norm(action - neg_arr) + 1e-8)
                            self.policy_b -= self.lr_policy * repulsion

        self.sigma = np.maximum(self.min_sigma, self.sigma * self.sigma_decay)
        entropy = gaussian_entropy(self.sigma) * self.entropy_beta
        reward_mean = float(np.mean(self.memory.rewards))
        stats = LigandUpdateStats(
            policy_loss=float(policy_loss / len(self.memory.rewards)),
            value_loss=float(value_loss / len(self.memory.rewards)),
            entropy=float(entropy),
            sigma_mean=float(self.sigma.mean()),
            reward_mean=reward_mean,
        )
        self.memory.clear()
        return stats
