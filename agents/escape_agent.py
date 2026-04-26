from __future__ import annotations

from typing import Any

import numpy as np

from agents.receptor_agent import ReceptorMutatorAgent
from core.spaces import Box


class EscapeAgent(ReceptorMutatorAgent):
    env_id = "EscapeAgent-v1"

    def __init__(self, obs_dim: int, action_space: Box, **kwargs: Any) -> None:
        super().__init__(obs_dim=obs_dim, action_space=action_space, **kwargs)
        self.escape_archive: list[np.ndarray] = []
        self.hard_negative_vectors: list[np.ndarray] = []

    def diversity_bonus(self, mutation: np.ndarray) -> float:
        mutation_arr = np.asarray(mutation, dtype=np.float64)
        if not self.escape_archive:
            return 0.1
        norm_current = np.linalg.norm(mutation_arr) + 1e-8
        normalized_current = mutation_arr / norm_current
        similarities = []
        for archived in self.escape_archive[-8:]:
            norm_archived = np.linalg.norm(archived) + 1e-8
            similarities.append(float(np.dot(normalized_current, archived / norm_archived)))
        cosine_distance = float(np.mean(1.0 - np.array(similarities)))
        return max(0.0, cosine_distance) * 0.2

    def store_episode(self, obs: np.ndarray, action: np.ndarray, reward: float, info: dict[str, Any]) -> None:
        super().store_episode(obs, action, reward, info)
        self.escape_archive.append(np.asarray(action, dtype=np.float64))
        if "mutated_receptor" in info:
            self.hard_negative_vectors.append(np.asarray(info["mutated_receptor"], dtype=np.float64))
