"""Receptor state model and binding oracle for the ligand-receptor simulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class ReceptorState:
    """Shared receptor state that both environments reference.

    The receptor is modeled as a real-valued vector representing binding-site
    residue properties (hydrophobicity, charge, steric bulk, etc.).
    """

    def __init__(
        self,
        dim: int = 16,
        wildtype_seed: int = 42,
        noise_std: float = 0.05,
        mutation_cap: float = 3.0,
    ):
        self.dim = dim
        self.wildtype_seed = wildtype_seed
        self.noise_std = noise_std
        self.mutation_cap = mutation_cap

        wt_rng = np.random.default_rng(wildtype_seed)
        self._wildtype = wt_rng.standard_normal(dim).astype(np.float64)
        self.vector: NDArray = self._wildtype.copy()
        self._history: list[NDArray] = [self.vector.copy()]

    @property
    def wildtype(self) -> NDArray:
        return self._wildtype.copy()

    def reset(self, rng: np.random.Generator | None = None) -> NDArray:
        """Reset to wildtype + small Gaussian noise."""
        rng = rng or np.random.default_rng()
        self.vector = self._wildtype + rng.normal(0, self.noise_std, self.dim)
        self._history = [self.vector.copy()]
        return self.vector.copy()

    def mutate(self, delta: NDArray) -> NDArray:
        """Apply a mutation delta, clamped to L2 norm <= mutation_cap."""
        delta = np.asarray(delta, dtype=np.float64)
        norm = np.linalg.norm(delta)
        if norm > self.mutation_cap:
            delta = delta * (self.mutation_cap / norm)
        self.vector = self.vector + delta
        self._history.append(self.vector.copy())
        return self.vector.copy()

    @property
    def displacement_from_wildtype(self) -> float:
        return float(np.linalg.norm(self.vector - self._wildtype))

    @property
    def history(self) -> list[NDArray]:
        return [h.copy() for h in self._history]


class BindingOracle:
    """Computes binding affinity between a ligand configuration and receptor.

    Uses a simplified energy model:
      binding_score = exp(-||W·(ligand - receptor)||² / temperature)
    where W is a learned interaction matrix.
    """

    def __init__(
        self,
        dim: int = 16,
        temperature: float = 2.0,
        seed: int = 123,
    ):
        self.dim = dim
        self.temperature = temperature
        rng = np.random.default_rng(seed)
        self._W = rng.standard_normal((dim, dim)) * 0.3
        self._W = (self._W + self._W.T) / 2  # symmetric interaction

    def score(self, ligand: NDArray, receptor: NDArray) -> float:
        """Binding score in [0, 1]. Higher = stronger binding."""
        diff = np.asarray(ligand) - np.asarray(receptor)
        energy = diff @ self._W @ diff
        return float(np.exp(-energy**2 / self.temperature))

    def off_target_score(self, ligand: NDArray, decoy: NDArray) -> float:
        """Off-target affinity against a decoy receptor (penalized)."""
        return self.score(ligand, decoy)

    def selectivity(
        self, ligand: NDArray, target: NDArray, decoys: list[NDArray]
    ) -> float:
        """Selectivity = target_score - max(off_target_scores)."""
        on = self.score(ligand, target)
        off = max(self.off_target_score(ligand, d) for d in decoys) if decoys else 0.0
        return on - off
