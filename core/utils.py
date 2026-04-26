from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


EPSILON = 1e-8


def engineered_features(obs: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float64).ravel()
    return np.concatenate([arr, arr * arr, np.array([1.0], dtype=np.float64)])


def monte_carlo_returns(rewards: list[float], gamma: float) -> np.ndarray:
    returns = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def normalize_advantages(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    std = float(arr.std())
    if std < EPSILON:
        return arr - arr.mean()
    return (arr - arr.mean()) / (std + EPSILON)


def gaussian_log_prob(
    action: Iterable[float] | np.ndarray,
    mean: Iterable[float] | np.ndarray,
    sigma: Iterable[float] | np.ndarray,
) -> float:
    action_arr = np.asarray(action, dtype=np.float64)
    mean_arr = np.asarray(mean, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    var = sigma_arr * sigma_arr + EPSILON
    diff = action_arr - mean_arr
    return float(-0.5 * np.sum(np.log(2.0 * np.pi * var) + (diff * diff) / var))


def gaussian_entropy(sigma: Iterable[float] | np.ndarray) -> float:
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    var = sigma_arr * sigma_arr + EPSILON
    return float(0.5 * np.sum(np.log(2.0 * np.pi * np.e * var)))


def potential_shaping(prev_phi: float, curr_phi: float, gamma: float) -> float:
    return float(gamma * curr_phi - prev_phi)


def clip_l2(vector: Iterable[float] | np.ndarray, cap: float) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if cap <= 0 or norm <= cap:
        return arr
    return arr * (cap / (norm + EPSILON))


def safe_ratio(first: float, second: float, eps: float = 1e-6) -> float:
    numerator = max(abs(first), abs(second))
    denominator = max(min(abs(first), abs(second)), eps)
    return float(numerator / denominator)


@dataclass
class RewardNormalizer:
    momentum: float = 0.05
    mean: float = 0.0
    variance: float = 1.0
    initialized: bool = False

    def update(self, reward: float) -> float:
        value = float(reward)
        if not self.initialized:
            self.mean = value
            self.variance = 1.0
            self.initialized = True
            return 0.0
        delta = value - self.mean
        self.mean += self.momentum * delta
        self.variance = (1.0 - self.momentum) * self.variance + self.momentum * delta * delta
        std = float(np.sqrt(max(self.variance, EPSILON)))
        return (value - self.mean) / (std + EPSILON)


@dataclass
class EpisodeMemory:
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    means: list[np.ndarray] = field(default_factory=list)
    entropies: list[float] = field(default_factory=list)
    next_observations: list[np.ndarray] = field(default_factory=list)
    infos: list[dict] = field(default_factory=list)

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.means.clear()
        self.entropies.clear()
        self.next_observations.clear()
        self.infos.clear()
