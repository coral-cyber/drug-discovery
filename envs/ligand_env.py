from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.receptor import ReceptorState
from core.spaces import Box
from core.utils import potential_shaping


TransitionHook = Callable[[dict[str, Any]], dict[str, Any]]


class LigandEnv:
    env_id = "LigandVsReceptor-v1"

    def __init__(
        self,
        receptor_state: ReceptorState,
        max_steps: int = 8,
        gamma: float = 0.99,
        binding_threshold: float | None = None,
        wildtype_noise: float | None = None,
        seed: int = 0,
    ) -> None:
        self.receptor_state = receptor_state
        self.dimension = receptor_state.dimension
        self.max_steps = max_steps
        self.gamma = gamma
        self.binding_threshold = binding_threshold or receptor_state.binding_threshold
        self.wildtype_noise = wildtype_noise if wildtype_noise is not None else receptor_state.noise_std
        self.rng = np.random.default_rng(seed)
        self.observation_space = Box(low=-3.0, high=3.0, shape=(self.dimension + 2,))
        self.action_space = Box(low=-2.5, high=2.5, shape=(self.dimension,))
        self.transition_hook: TransitionHook | None = None
        self.episode = 0
        self.step_count = 0
        self.prev_binding = 0.0
        self.prev_ligand = np.zeros(self.dimension, dtype=np.float64)
        self.last_info: dict[str, Any] = {}

    def set_transition_hook(self, hook: TransitionHook | None) -> None:
        self.transition_hook = hook

    def set_env_vars(self, noise_std: float | None = None, binding_threshold: float | None = None, wildtype_seed: int | None = None) -> None:
        if noise_std is not None:
            self.wildtype_noise = noise_std
            self.receptor_state.noise_std = noise_std
        if binding_threshold is not None:
            self.binding_threshold = binding_threshold
            self.receptor_state.binding_threshold = binding_threshold
        if wildtype_seed is not None:
            self.receptor_state.wildtype_seed = wildtype_seed

    def get_state(self) -> dict[str, Any]:
        return {
            "receptor_vector": self.receptor_state.clone_vector(),
            "step": self.step_count,
            "episode": self.episode,
            "prev_binding": self.prev_binding,
            "prev_ligand": self.prev_ligand.copy(),
        }

    def _observation(self) -> np.ndarray:
        step_fraction = self.step_count / max(self.max_steps, 1)
        return np.concatenate([self.receptor_state.clone_vector(), np.array([step_fraction, self.prev_binding], dtype=np.float64)])

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.episode += 1
        self.step_count = 0
        self.prev_binding = 0.0
        self.prev_ligand = np.zeros(self.dimension, dtype=np.float64)
        self.receptor_state.reset(noise_std=self.wildtype_noise, seed=seed, preserve_history=True)
        obs = self._observation()
        info = {
            "binding_score": 0.0,
            "step": self.step_count,
            "episode": self.episode,
            "env_id": self.env_id,
            "full_state": self.get_state(),
        }
        self.last_info = info
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        ligand = self.action_space.clip(action)
        binding = self.receptor_state.binding_oracle(ligand)
        shaping = potential_shaping(self.prev_binding, binding.binding_score, self.gamma)
        novelty_bonus = float(np.linalg.norm(ligand - self.prev_ligand) / np.sqrt(self.dimension)) * 0.1
        reward = float(binding.binding_score + shaping + novelty_bonus - binding.synthesis_cost)
        self.step_count += 1
        done = bool(binding.binding_score >= self.binding_threshold or self.step_count >= self.max_steps)
        self.prev_binding = binding.binding_score
        self.prev_ligand = ligand.copy()
        info = {
            "binding_score": binding.binding_score,
            "step": self.step_count,
            "episode": self.episode,
            "env_id": self.env_id,
            "reward_breakdown": {
                "binding": binding.binding_score,
                "potential_shaping": shaping,
                "novelty_bonus": novelty_bonus,
                "synthesis_cost": binding.synthesis_cost,
            },
            "binding_result": binding.to_dict(),
            "full_state": self.get_state(),
        }
        if done and self.transition_hook is not None:
            info["transition_preview"] = self.transition_hook(
                {
                    "episode": self.episode,
                    "avg_binding": binding.binding_score,
                    "receptor_vector": self.receptor_state.clone_vector(),
                    "last_ligand": ligand.copy(),
                }
            )
        self.last_info = info
        return self._observation(), reward, done, info

    def render(self) -> str:
        filled = int(round(self.prev_binding * 20))
        bar = "#" * filled + "-" * max(0, 20 - filled)
        return f"[{bar}] binding={self.prev_binding:.3f} step={self.step_count}"
