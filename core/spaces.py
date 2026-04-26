"""Action and observation spaces (Box, Discrete, MultiDiscrete). No gymnasium dependency."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Space:
    """Abstract base for all spaces."""

    def __init__(self, shape: tuple[int, ...], dtype: np.dtype | type = np.float64):
        self.shape = shape
        self.dtype = np.dtype(dtype)

    def sample(self, rng: np.random.Generator | None = None) -> NDArray:
        raise NotImplementedError

    def contains(self, x: NDArray) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(shape={self.shape}, dtype={self.dtype})"


class Box(Space):
    """Continuous multi-dimensional space bounded by low/high."""

    def __init__(
        self,
        low: float | NDArray,
        high: float | NDArray,
        shape: tuple[int, ...] | None = None,
        dtype: np.dtype | type = np.float64,
    ):
        if shape is None:
            low_arr = np.asarray(low, dtype=dtype)
            high_arr = np.asarray(high, dtype=dtype)
            if low_arr.ndim == 0 and high_arr.ndim == 0:
                raise ValueError("shape must be provided when low/high are scalars")
            shape = np.broadcast_shapes(low_arr.shape, high_arr.shape)
        super().__init__(shape, dtype)
        self.low = np.broadcast_to(np.asarray(low, dtype=dtype), shape).copy()
        self.high = np.broadcast_to(np.asarray(high, dtype=dtype), shape).copy()

    def sample(self, rng: np.random.Generator | None = None) -> NDArray:
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x: NDArray) -> bool:
        x = np.asarray(x, dtype=self.dtype)
        if x.shape != self.shape:
            return False
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def clip(self, x: NDArray) -> NDArray:
        return np.clip(np.asarray(x, dtype=self.dtype), self.low, self.high)

    def __repr__(self) -> str:
        return f"Box(low={self.low.min():.2f}, high={self.high.max():.2f}, shape={self.shape})"


class Discrete(Space):
    """Integer space {0, 1, ..., n-1}."""

    def __init__(self, n: int):
        super().__init__(shape=(), dtype=np.int64)
        self.n = n

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.n))

    def contains(self, x: int | NDArray) -> bool:
        x_int = int(x)
        return 0 <= x_int < self.n

    def __repr__(self) -> str:
        return f"Discrete(n={self.n})"


class MultiDiscrete(Space):
    """Multiple independent discrete spaces."""

    def __init__(self, nvec: list[int] | NDArray):
        self.nvec = np.asarray(nvec, dtype=np.int64)
        super().__init__(shape=self.nvec.shape, dtype=np.int64)

    def sample(self, rng: np.random.Generator | None = None) -> NDArray:
        rng = rng or np.random.default_rng()
        return np.array([rng.integers(0, n) for n in self.nvec], dtype=np.int64)

    def contains(self, x: NDArray) -> bool:
        x = np.asarray(x, dtype=np.int64)
        if x.shape != self.shape:
            return False
        return bool(np.all(x >= 0) and np.all(x < self.nvec))

    def __repr__(self) -> str:
        return f"MultiDiscrete(nvec={self.nvec.tolist()})"
