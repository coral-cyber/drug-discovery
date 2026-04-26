"""LigandEnv — environment faced by Agent_A (LigandDesigner).

Full OpenEnv spec. No gymnasium dependency.
Transition dynamics at episode boundaries are driven by Agent_B's mutation policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.receptor import BindingOracle, ReceptorState
from core.spaces import Box


class LigandEnv:
    """Environment where the ligand-designer agent proposes binding configurations.

    Observations include the receptor vector plus partial-observability features
    (step_fraction, prev_binding).  Full state is maintained internally.
    """

    metadata = {"env_id": "LigandVsReceptor-v1", "render_modes": ["ansi"]}

    def __init__(
        self,
        receptor_state: ReceptorState,
        oracle: BindingOracle,
        *,
        max_steps: int = 50,
        gamma: float = 0.99,
        binding_threshold: float = 0.85,
        noise_std: float = 0.05,
        synthesis_cost_coeff: float = 0.01,
        novelty_coeff: float = 0.05,
        seed: int | None = None,
    ):
        self.receptor_state = receptor_state
        self.oracle = oracle
        self.dim = receptor_state.dim
        self.max_steps = max_steps
        self.gamma = gamma
        self.binding_threshold = binding_threshold
        self.noise_std = noise_std
        self.synthesis_cost_coeff = synthesis_cost_coeff
        self.novelty_coeff = novelty_coeff

        self.observation_space = Box(
            low=-10.0, high=10.0, shape=(self.dim + 2,)
        )
        self.action_space = Box(low=-3.0, high=3.0, shape=(self.dim,))

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._episode = 0
        self._prev_binding = 0.0
        self._prev_potential = 0.0
        self._state: NDArray = np.zeros(self.dim)
        self._ligand_history: list[NDArray] = []

        self._mutator_hook: Any = None

    @property
    def env_id(self) -> str:
        return self.metadata["env_id"]

    def set_mutator_hook(self, hook: Any) -> None:
        """Register Agent_B's mutation callback for episode transitions."""
        self._mutator_hook = hook

    def _build_obs(self) -> NDArray:
        step_frac = self._step_count / self.max_steps
        obs = np.concatenate([
            self.receptor_state.vector,
            [step_frac, self._prev_binding],
        ])
        return obs.astype(np.float64)

    def _get_full_state(self) -> NDArray:
        return np.concatenate([
            self.receptor_state.vector,
            self._state,
            [self._step_count, self._prev_binding],
        ])

    def _potential(self, binding: float) -> float:
        """Potential function for reward shaping: Φ(s) = binding_score."""
        return binding

    def reset(self, seed: int | None = None) -> tuple[NDArray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.receptor_state.reset(self._rng)
        self._step_count = 0
        self._episode += 1
        self._prev_binding = 0.0
        self._prev_potential = 0.0
        self._state = self._rng.normal(0, self.noise_std, self.dim)
        self._ligand_history = []

        obs = self._build_obs()
        info = self._info(0.0)
        return obs, info

    def step(self, action: NDArray) -> tuple[NDArray, float, bool, bool, dict[str, Any]]:
        action = self.action_space.clip(np.asarray(action, dtype=np.float64))
        self._step_count += 1
        self._state = action.copy()
        self._ligand_history.append(action.copy())

        binding = self.oracle.score(action, self.receptor_state.vector)
        current_potential = self._potential(binding)

        # Potential-based reward shaping (Ng 1999)
        shaping = self.gamma * current_potential - self._prev_potential

        # Novelty bonus — cosine distance from mean of previous ligands
        novelty = 0.0
        if len(self._ligand_history) > 1:
            prev_mean = np.mean(self._ligand_history[:-1], axis=0)
            cos_sim = np.dot(action, prev_mean) / (
                np.linalg.norm(action) * np.linalg.norm(prev_mean) + 1e-8
            )
            novelty = (1.0 - cos_sim) * self.novelty_coeff

        synthesis_cost = self.synthesis_cost_coeff * float(np.linalg.norm(action))

        reward = binding + shaping + novelty - synthesis_cost
        self._prev_binding = binding
        self._prev_potential = current_potential

        terminated = binding >= self.binding_threshold
        truncated = self._step_count >= self.max_steps

        obs = self._build_obs()
        info = self._info(binding)
        return obs, float(reward), terminated, truncated, info

    def _info(self, binding: float) -> dict[str, Any]:
        return {
            "binding_score": binding,
            "step": self._step_count,
            "episode": self._episode,
            "env_id": self.env_id,
            "full_state": self._get_full_state().tolist(),
            "receptor_displacement": self.receptor_state.displacement_from_wildtype,
        }

    def render(self, mode: str = "ansi") -> str:
        bar_len = 40
        fill = int(self._prev_binding * bar_len)
        bar = "█" * fill + "░" * (bar_len - fill)
        return (
            f"[{self.env_id}] ep={self._episode} step={self._step_count}/{self.max_steps}\n"
            f"  binding: |{bar}| {self._prev_binding:.4f}\n"
            f"  receptor Δ: {self.receptor_state.displacement_from_wildtype:.4f}"
        )

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)
