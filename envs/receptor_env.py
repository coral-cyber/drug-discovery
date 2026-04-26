from __future__ import annotations

from typing import Any, Callable

import numpy as np

from core.receptor import ReceptorState
from core.spaces import Box
from core.utils import clip_l2, potential_shaping


ProbeHook = Callable[[int], list[np.ndarray]]


class ReceptorEnv:
    env_id = "ReceptorAdversary-v1"

    def __init__(self, receptor_state: ReceptorState, probe_count: int = 4, gamma: float = 0.99, sigma_bio: float = 1.0, mutation_l2_cap: float = 1.4, seed: int = 0) -> None:
        self.receptor_state = receptor_state
        self.dimension = receptor_state.dimension
        self.probe_count = probe_count
        self.gamma = gamma
        self.sigma_bio = sigma_bio
        self.mutation_l2_cap = mutation_l2_cap
        self.rng = np.random.default_rng(seed)
        self.observation_space = Box(low=-3.0, high=3.0, shape=(self.dimension + self.probe_count,))
        self.action_space = Box(low=-3.0 * sigma_bio, high=3.0 * sigma_bio, shape=(self.dimension,))
        self.episode = 0
        self.step_count = 0
        self.prev_avg_binding = 0.0
        self.probe_hook: ProbeHook | None = None

    def set_probe_hook(self, hook: ProbeHook | None) -> None:
        self.probe_hook = hook

    def _default_probes(self, count: int) -> list[np.ndarray]:
        probes = []
        for index in range(count):
            probes.append(np.roll(self.receptor_state.wildtype_vector, index).astype(np.float64))
        return probes

    def _probe_readings(self) -> list[float]:
        return self.probe_readings()

    def probe_readings(self) -> list[float]:
        probes = self.probe_hook(self.probe_count) if self.probe_hook is not None else self._default_probes(self.probe_count)
        return self.receptor_state.probe_bindings(probes)

    def _observation(self, probe_readings: list[float]) -> np.ndarray:
        return np.concatenate([self.receptor_state.clone_vector(), np.asarray(probe_readings, dtype=np.float64)])

    def get_state(self, probe_readings: list[float]) -> dict[str, Any]:
        return {
            "receptor_vector": self.receptor_state.clone_vector(),
            "probe_readings": list(probe_readings),
            "step": self.step_count,
            "episode": self.episode,
        }

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]:
        self.episode += 1
        self.step_count = 0
        probe_readings = self._probe_readings()
        self.prev_avg_binding = float(np.mean(probe_readings)) if probe_readings else 0.0
        info = {
            "avg_binding": self.prev_avg_binding,
            "probe_readings": probe_readings,
            "step": self.step_count,
            "episode": self.episode,
            "env_id": self.env_id,
            "full_state": self.get_state(probe_readings),
        }
        return self._observation(probe_readings), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        clipped = self.action_space.clip(action)
        constrained = clip_l2(clipped, self.mutation_l2_cap)
        mutated_vector = self.receptor_state.apply_mutation(constrained, l2_cap=self.mutation_l2_cap)
        self.step_count += 1
        probe_readings = self._probe_readings()
        avg_binding = float(np.mean(probe_readings)) if probe_readings else 0.0
        escape_bonus = max(0.0, self.prev_avg_binding - avg_binding)
        functionality = self.receptor_state.functionality()
        constraint_penalty = max(0.0, self.receptor_state.functionality_floor - functionality)
        escape_potential = 1.0 - avg_binding
        prev_potential = 1.0 - self.prev_avg_binding
        shaping = potential_shaping(prev_potential, escape_potential, self.gamma)
        reward = float((1.0 - avg_binding) + escape_bonus + shaping - constraint_penalty)
        self.prev_avg_binding = avg_binding
        info = {
            "avg_binding": avg_binding,
            "escape_bonus": escape_bonus,
            "constraint_penalty": constraint_penalty,
            "probe_readings": probe_readings,
            "functionality": functionality,
            "mutated_receptor": mutated_vector.copy(),
            "step": self.step_count,
            "episode": self.episode,
            "env_id": self.env_id,
            "full_state": self.get_state(probe_readings),
            "motifs": self.receptor_state.summarize_escape_motifs(),
        }
        return self._observation(probe_readings), reward, True, info

    def render(self) -> str:
        inverse_binding = 1.0 - self.prev_avg_binding
        filled = int(round(inverse_binding * 20))
        bar = "#" * filled + "-" * max(0, 20 - filled)
        return f"[{bar}] escape={inverse_binding:.3f} step={self.step_count}"
