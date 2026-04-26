from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.spaces import Box
from core.utils import engineered_features, gaussian_entropy, gaussian_log_prob, normalize_advantages


@dataclass
class ReceptorUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    reward_mean: float
    baseline: float


class ReceptorMutatorAgent:
    env_id = "ReceptorMutator-v1"

    def __init__(
        self,
        obs_dim: int,
        action_space: Box,
        lr: float = 0.02,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        entropy_beta: float = 0.015,
        baseline_alpha: float = 0.05,
        sigma_init: float = 0.45,
        sigma_floor: float = 0.05,
        seed: int = 0,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_space = action_space
        self.action_dim = int(np.prod(action_space.shape))
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.entropy_beta = entropy_beta
        self.baseline_alpha = baseline_alpha
        self.sigma_floor = sigma_floor
        self.rng = np.random.default_rng(seed)
        self.policy_w = self.rng.normal(0.0, 0.06, size=(self.obs_dim, self.action_dim))
        self.policy_b = np.zeros(self.action_dim, dtype=np.float64)
        self.value_w = self.rng.normal(0.0, 0.04, size=(2 * self.obs_dim + 1,))
        self.sigma = np.linspace(sigma_init * 0.8, sigma_init * 1.2, self.action_dim)
        self.baseline = 0.0
        self.timestep = 0
        self.m_policy_w = np.zeros_like(self.policy_w)
        self.v_policy_w = np.zeros_like(self.policy_w)
        self.m_policy_b = np.zeros_like(self.policy_b)
        self.v_policy_b = np.zeros_like(self.policy_b)
        self.m_value_w = np.zeros_like(self.value_w)
        self.v_value_w = np.zeros_like(self.value_w)
        self.last_episode: dict[str, Any] | None = None
        self.last_bias = np.zeros(self.action_dim, dtype=np.float64)
        self.mutation_archive: list[np.ndarray] = []

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
        action = self.action_space.clip((1.0 - llm_weight) * raw_action + bias)
        return {
            "mean": mean,
            "raw_action": raw_action,
            "action": action,
            "log_prob": gaussian_log_prob(action, mean, self.sigma),
            "entropy": gaussian_entropy(self.sigma) * self.entropy_beta,
            "bias": bias.copy(),
        }

    def diversity_bonus(self, mutation: np.ndarray) -> float:
        mutation_arr = np.asarray(mutation, dtype=np.float64)
        if not self.mutation_archive:
            return 0.05
        normalized_current = mutation_arr / (np.linalg.norm(mutation_arr) + 1e-8)
        similarities = []
        for archived in self.mutation_archive[-8:]:
            normalized_archived = archived / (np.linalg.norm(archived) + 1e-8)
            similarities.append(float(np.dot(normalized_current, normalized_archived)))
        return float(max(0.0, 1.0 - np.mean(similarities)) * 0.1)

    def store_episode(self, obs: np.ndarray, action: np.ndarray, reward: float, info: dict[str, Any]) -> None:
        self.last_episode = {
            "obs": np.asarray(obs, dtype=np.float64),
            "action": np.asarray(action, dtype=np.float64),
            "reward": float(reward),
            "info": info,
        }
        self.mutation_archive.append(np.asarray(action, dtype=np.float64))

    def _adam_step(self, grad: np.ndarray, m: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.timestep += 1
        m = self.beta1 * m + (1.0 - self.beta1) * grad
        v = self.beta2 * v + (1.0 - self.beta2) * (grad * grad)
        m_hat = m / (1.0 - self.beta1 ** self.timestep)
        v_hat = v / (1.0 - self.beta2 ** self.timestep)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return update, m, v

    def ensure_entropy_floor(self, min_entropy: float) -> bool:
        current_entropy = gaussian_entropy(self.sigma)
        if current_entropy >= min_entropy:
            return False
        self.sigma = np.maximum(self.sigma * 1.08, self.sigma_floor)
        return True

    def learn(self) -> ReceptorUpdateStats:
        if self.last_episode is None:
            return ReceptorUpdateStats(0.0, 0.0, gaussian_entropy(self.sigma), 0.0, self.baseline)

        obs = self.last_episode["obs"]
        action = self.last_episode["action"]
        reward = float(self.last_episode["reward"]) + self.diversity_bonus(action)
        value_prediction = self.value(obs)
        advantage = normalize_advantages(np.array([reward - value_prediction]))[0]
        if abs(advantage) < 1e-8:
            advantage = reward - value_prediction
        mean = self.policy_mean(obs)
        sigma_sq = self.sigma * self.sigma + 1e-8
        grad_mean = (action - mean) / sigma_sq
        grad_w = np.outer(obs, grad_mean * advantage)
        grad_b = grad_mean * advantage
        policy_update_w, self.m_policy_w, self.v_policy_w = self._adam_step(grad_w, self.m_policy_w, self.v_policy_w)
        policy_update_b, self.m_policy_b, self.v_policy_b = self._adam_step(grad_b, self.m_policy_b, self.v_policy_b)
        self.policy_w += policy_update_w
        self.policy_b += policy_update_b
        features = engineered_features(obs)
        td_error = value_prediction - reward
        value_grad = 2.0 * td_error * features
        value_update, self.m_value_w, self.v_value_w = self._adam_step(value_grad, self.m_value_w, self.v_value_w)
        self.value_w -= value_update
        self.baseline = (1.0 - self.baseline_alpha) * self.baseline + self.baseline_alpha * reward
        entropy = gaussian_entropy(self.sigma) * self.entropy_beta
        stats = ReceptorUpdateStats(
            policy_loss=float(-gaussian_log_prob(action, mean, self.sigma) * advantage),
            value_loss=float(td_error * td_error),
            entropy=float(entropy),
            reward_mean=reward,
            baseline=float(self.baseline),
        )
        self.last_episode = None
        return stats
