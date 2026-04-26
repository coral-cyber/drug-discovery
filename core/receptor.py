from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class BindingResult:
    binding_score: float
    off_target_affinity: float
    selectivity: float
    stability: float
    functionality: float
    synthesis_cost: float
    total_score: float

    def to_dict(self) -> dict[str, float]:
        return {
            "binding_score": self.binding_score,
            "off_target_affinity": self.off_target_affinity,
            "selectivity": self.selectivity,
            "stability": self.stability,
            "functionality": self.functionality,
            "synthesis_cost": self.synthesis_cost,
            "total_score": self.total_score,
        }


@dataclass
class ReceptorState:
    dimension: int
    wildtype_seed: int = 11
    noise_std: float = 0.05
    binding_threshold: float = 0.82
    mutation_scale: float = 0.2
    functionality_floor: float = 0.55
    rng: np.random.Generator = field(init=False, repr=False)
    wildtype_vector: np.ndarray = field(init=False)
    vector: np.ndarray = field(init=False)
    off_target_panel: np.ndarray = field(init=False)
    mutation_history: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.wildtype_seed)
        self.wildtype_vector = self._generate_wildtype()
        self.off_target_panel = self._generate_off_targets()
        self.vector = self.wildtype_vector.copy()

    def _generate_wildtype(self) -> np.ndarray:
        lin = np.linspace(-1.0, 1.0, self.dimension)
        periodic = np.sin(np.linspace(0.0, np.pi, self.dimension))
        template = 0.65 * lin + 0.35 * periodic
        return template.astype(np.float64)

    def _generate_off_targets(self) -> np.ndarray:
        panel = [np.roll(self.wildtype_vector, shift) for shift in range(1, min(4, self.dimension))]
        if not panel:
            panel = [self.wildtype_vector[::-1]]
        return np.vstack(panel).astype(np.float64)

    def clone_vector(self) -> np.ndarray:
        return self.vector.copy()

    def reset(self, noise_std: float | None = None, seed: int | None = None, preserve_history: bool = False) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        std = self.noise_std if noise_std is None else noise_std
        noise = self.rng.normal(0.0, std, size=self.dimension)
        self.vector = self.wildtype_vector + noise
        if not preserve_history:
            self.mutation_history.clear()
        return self.clone_vector()

    def sync(self, new_vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(new_vector, dtype=np.float64)
        if arr.shape != (self.dimension,):
            raise ValueError("receptor vector shape mismatch")
        self.vector = arr.copy()
        return self.clone_vector()

    def functionality(self) -> float:
        deviation = float(np.linalg.norm(self.vector - self.wildtype_vector) / np.sqrt(self.dimension))
        return float(np.exp(-deviation))

    def apply_mutation(self, delta: np.ndarray, l2_cap: float | None = None) -> np.ndarray:
        mutation = np.asarray(delta, dtype=np.float64)
        if mutation.shape != (self.dimension,):
            raise ValueError("mutation shape mismatch")
        if l2_cap is not None and l2_cap > 0:
            norm = float(np.linalg.norm(mutation))
            if norm > l2_cap:
                mutation = mutation * (l2_cap / (norm + 1e-8))
        self.vector = self.vector + mutation * self.mutation_scale
        self.mutation_history.append(mutation.copy())
        return self.clone_vector()

    def binding_oracle(self, ligand: np.ndarray) -> BindingResult:
        ligand_arr = np.asarray(ligand, dtype=np.float64)
        if ligand_arr.shape != (self.dimension,):
            raise ValueError("ligand shape mismatch")

        receptor = self.vector
        distance = float(np.linalg.norm(ligand_arr - receptor) / np.sqrt(self.dimension))
        off_target_distances = np.linalg.norm(self.off_target_panel - ligand_arr, axis=1) / np.sqrt(self.dimension)
        off_target_affinity = float(np.exp(-float(np.min(off_target_distances))))
        binding_score = float(np.exp(-distance))
        selectivity = float(binding_score - 0.65 * off_target_affinity)
        stability = float(np.exp(-np.mean(np.abs(np.gradient(ligand_arr)))))
        functionality = self.functionality()
        synthesis_cost = float(0.08 * np.linalg.norm(ligand_arr) / np.sqrt(self.dimension))
        total_score = float(0.45 * binding_score + 0.2 * stability + 0.2 * selectivity + 0.15 * functionality)
        return BindingResult(
            binding_score=binding_score,
            off_target_affinity=off_target_affinity,
            selectivity=selectivity,
            stability=stability,
            functionality=functionality,
            synthesis_cost=synthesis_cost,
            total_score=total_score,
        )

    def probe_bindings(self, ligands: list[np.ndarray]) -> list[float]:
        return [self.binding_oracle(ligand).binding_score for ligand in ligands]

    def mutation_diversity(self) -> float:
        if len(self.mutation_history) < 2:
            return 0.0
        normalized = []
        for mutation in self.mutation_history:
            norm = float(np.linalg.norm(mutation))
            normalized.append(mutation / norm if norm > 0 else mutation.copy())
        pairwise = []
        for first, second in zip(normalized[:-1], normalized[1:]):
            pairwise.append(1.0 - float(np.dot(first, second)))
        return float(np.mean(pairwise))

    def summarize_escape_motifs(self, top_k: int = 3) -> list[dict[str, Any]]:
        if not self.mutation_history:
            return []
        magnitudes = [(idx, float(np.linalg.norm(mutation)), mutation) for idx, mutation in enumerate(self.mutation_history)]
        magnitudes.sort(key=lambda item: item[1], reverse=True)
        motifs: list[dict[str, Any]] = []
        for idx, magnitude, mutation in magnitudes[:top_k]:
            dominant_idx = np.argsort(np.abs(mutation))[-3:][::-1]
            motifs.append(
                {
                    "mutation_index": idx,
                    "magnitude": magnitude,
                    "dominant_sites": dominant_idx.tolist(),
                    "signature": mutation[dominant_idx].round(4).tolist(),
                }
            )
        return motifs

    def state_vector(self) -> np.ndarray:
        return np.concatenate([self.vector, self.wildtype_vector, np.array([self.functionality()], dtype=np.float64)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "wildtype_seed": self.wildtype_seed,
            "noise_std": self.noise_std,
            "binding_threshold": self.binding_threshold,
            "vector": self.vector.round(6).tolist(),
            "wildtype_vector": self.wildtype_vector.round(6).tolist(),
            "functionality": self.functionality(),
            "mutation_diversity": self.mutation_diversity(),
            "mutation_count": len(self.mutation_history),
        }
