"""ReceptorEnv — environment faced by Agent_B (ReceptorMutator).

Full OpenEnv spec.  Agent_B sees aggregated binding probe readings from Agent_A
and outputs a mutation delta vector.  One action per episode (mirrors biology).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.receptor import BindingOracle, ReceptorState
from core.spaces import Box


class ReceptorEnv:
    """Episode-level environment for the receptor mutator agent.

    Each episode:
      1. Receive receptor vector + binding probe readings (from ligand agent).
      2. Output a single mutation delta.
      3. Receive scalar episode reward.
    """

    metadata = {"env_id": "ReceptorAdversary-v1", "render_modes": ["ansi"]}

    def __init__(
        self,
        receptor_state: ReceptorState,
        oracle: BindingOracle,
        *,
        probe_steps: int = 10,
        gamma: float = 0.99,
        mutation_cap: float = 3.0,
        escape_bonus_coeff: float = 0.1,
        constraint_penalty_coeff: float = 0.5,
        seed: int | None = None,
    ):
        self.receptor_state = receptor_state
        self.oracle = oracle
        self.dim = receptor_state.dim
        self.probe_steps = probe_steps
        self.gamma = gamma
        self.mutation_cap = mutation_cap
        self.escape_bonus_coeff = escape_bonus_coeff
        self.constraint_penalty_coeff = constraint_penalty_coeff

        obs_dim = self.dim + self.probe_steps
        self.observation_space = Box(low=-10.0, high=10.0, shape=(obs_dim,))

        bio_bound = 3.0
        self.action_space = Box(
            low=-bio_bound, high=bio_bound, shape=(self.dim,)
        )

        self._rng = np.random.default_rng(seed)
        self._episode = 0
        self._binding_probes: NDArray = np.zeros(self.probe_steps)
        self._prev_potential = 0.0
        self._probe_hook: Any = None

    @property
    def env_id(self) -> str:
        return self.metadata["env_id"]

    def set_probe_hook(self, hook: Any) -> None:
        """Register Agent_A's ligand probe callback for binding readings."""
        self._probe_hook = hook

    def _collect_probes(self) -> NDArray:
        """Run ligand probes to collect binding readings."""
        if self._probe_hook is not None:
            probes = self._probe_hook(self.receptor_state.vector, self.probe_steps)
            self._binding_probes = np.asarray(probes[:self.probe_steps])
        else:
            self._binding_probes = self._rng.uniform(0, 1, self.probe_steps)
        return self._binding_probes

    def _build_obs(self) -> NDArray:
        return np.concatenate([
            self.receptor_state.vector,
            self._binding_probes,
        ]).astype(np.float64)

    def reset(self, seed: int | None = None) -> tuple[NDArray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.receptor_state.reset(self._rng)
        self._episode += 1
        self._binding_probes = np.zeros(self.probe_steps)
        self._prev_potential = 0.0
        self._collect_probes()
        obs = self._build_obs()
        return obs, self._info(0.0)

    def step(
        self, action: NDArray
    ) -> tuple[NDArray, float, bool, bool, dict[str, Any]]:
        """Single-step episode: apply mutation, compute reward, done."""
        action = np.asarray(action, dtype=np.float64)

        # L2 norm cap — biological mutation magnitude constraint
        norm = float(np.linalg.norm(action))
        if norm > self.mutation_cap:
            action = action * (self.mutation_cap / norm)

        action = self.action_space.clip(action)

        self.receptor_state.mutate(action)

        self._collect_probes()
        avg_binding = float(np.mean(self._binding_probes))

        # Escape reward: 1 - avg_binding (lower binding = better escape)
        escape_score = 1.0 - avg_binding

        # Potential-based shaping
        current_potential = escape_score
        shaping = self.gamma * current_potential - self._prev_potential
        self._prev_potential = current_potential

        # Escape bonus for novel mutations
        displacement = self.receptor_state.displacement_from_wildtype
        escape_bonus = self.escape_bonus_coeff * min(displacement, self.mutation_cap)

        # Constraint penalty: penalize extreme displacement
        constraint_penalty = 0.0
        if displacement > self.mutation_cap:
            overshoot = displacement - self.mutation_cap
            constraint_penalty = self.constraint_penalty_coeff * overshoot

        reward = escape_score + shaping + escape_bonus - constraint_penalty

        obs = self._build_obs()
        terminated = True  # single action per episode
        truncated = False
        info = self._info(avg_binding)
        return obs, float(reward), terminated, truncated, info

    def _info(self, avg_binding: float) -> dict[str, Any]:
        return {
            "avg_binding": avg_binding,
            "episode": self._episode,
            "env_id": self.env_id,
            "receptor_displacement": self.receptor_state.displacement_from_wildtype,
            "receptor_vector": self.receptor_state.vector.tolist(),
        }

    def render(self, mode: str = "ansi") -> str:
        avg = float(np.mean(self._binding_probes))
        bar_len = 40
        fill = int((1 - avg) * bar_len)
        bar = "█" * fill + "░" * (bar_len - fill)
        return (
            f"[{self.env_id}] ep={self._episode}\n"
            f"  escape:  |{bar}| {1 - avg:.4f}\n"
            f"  Δ(wt):   {self.receptor_state.displacement_from_wildtype:.4f}"
        )

    def seed(self, s: int) -> None:
        self._rng = np.random.default_rng(s)
