from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Box:
    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: type = np.float64

    def __init__(
        self,
        low: float | Sequence[float] | np.ndarray,
        high: float | Sequence[float] | np.ndarray,
        shape: tuple[int, ...] | None = None,
        dtype: type = np.float64,
    ) -> None:
        low_arr = np.asarray(low, dtype=dtype)
        high_arr = np.asarray(high, dtype=dtype)
        if shape is None:
            inferred_shape = low_arr.shape or high_arr.shape
            if not inferred_shape:
                raise ValueError("shape must be provided for scalar bounds")
            shape = inferred_shape
        if low_arr.shape == ():
            low_arr = np.full(shape, float(low_arr), dtype=dtype)
        if high_arr.shape == ():
            high_arr = np.full(shape, float(high_arr), dtype=dtype)
        if low_arr.shape != shape or high_arr.shape != shape:
            raise ValueError("bounds must match shape")
        if np.any(low_arr > high_arr):
            raise ValueError("low must be <= high")
        object.__setattr__(self, "low", low_arr.astype(dtype))
        object.__setattr__(self, "high", high_arr.astype(dtype))
        object.__setattr__(self, "shape", tuple(shape))
        object.__setattr__(self, "dtype", dtype)

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return generator.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: Iterable[float] | np.ndarray) -> bool:
        arr = np.asarray(x, dtype=self.dtype)
        return arr.shape == self.shape and np.all(arr >= self.low) and np.all(arr <= self.high)

    def clip(self, x: Iterable[float] | np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(x, dtype=self.dtype), self.low, self.high).astype(self.dtype)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "type": "Box",
            "shape": self.shape,
            "low": self.low.tolist(),
            "high": self.high.tolist(),
        }


@dataclass(frozen=True)
class Discrete:
    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")

    def sample(self, rng: np.random.Generator | None = None) -> int:
        generator = rng or np.random.default_rng()
        return int(generator.integers(0, self.n))

    def contains(self, x: object) -> bool:
        return isinstance(x, (int, np.integer)) and 0 <= int(x) < self.n

    def to_jsonable(self) -> dict[str, object]:
        return {"type": "Discrete", "n": self.n}


@dataclass(frozen=True)
class MultiDiscrete:
    nvec: np.ndarray

    def __init__(self, nvec: Sequence[int] | np.ndarray) -> None:
        arr = np.asarray(nvec, dtype=np.int64)
        if arr.ndim != 1 or np.any(arr <= 0):
            raise ValueError("nvec must be a positive 1D array")
        object.__setattr__(self, "nvec", arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.nvec.shape

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng or np.random.default_rng()
        return np.array([generator.integers(0, n) for n in self.nvec], dtype=np.int64)

    def contains(self, x: Sequence[int] | np.ndarray) -> bool:
        arr = np.asarray(x, dtype=np.int64)
        return arr.shape == self.nvec.shape and np.all(arr >= 0) and np.all(arr < self.nvec)

    def to_jsonable(self) -> dict[str, object]:
        return {"type": "MultiDiscrete", "nvec": self.nvec.tolist()}
