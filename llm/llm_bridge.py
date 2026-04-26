"""LLM bridge — Claude API integration for bias injection.

Every N episodes the agent constructs a structured query from current state,
calls Claude, parses the response into a bias vector, and injects it.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import httpx

    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

SYSTEM_PROMPT = (
    "You are an expert computational chemist and optimization advisor. "
    "Given the current state of a ligand-receptor binding simulation, "
    "suggest a bias vector (list of floats) to nudge the agent's policy "
    "toward more promising binding configurations. "
    "Respond ONLY with a JSON object: {\"bias\": [float, ...]}. "
    "The vector must have exactly {dim} elements, each in [-2, 2]."
)


class LLMBridge:
    """Handles Claude API calls and bias vector extraction."""

    def __init__(
        self,
        dim: int,
        *,
        call_every_n: int = 10,
        weight: float = 0.25,
        model: str = "claude-sonnet-4-20250514",
        api_key: str | None = None,
        enabled: bool = True,
    ):
        self.dim = dim
        self.call_every_n = call_every_n
        self.weight = weight
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.enabled = enabled and bool(self.api_key) and _HAS_HTTPX
        self._call_count = 0
        self._last_bias: NDArray | None = None
        self._history: list[dict[str, Any]] = []

    def _build_query(
        self,
        receptor_vector: NDArray,
        binding_score: float,
        episode: int,
        agent_name: str,
    ) -> str:
        return json.dumps(
            {
                "agent": agent_name,
                "episode": episode,
                "receptor_vector": receptor_vector.tolist()[:8],
                "binding_score": round(binding_score, 4),
                "request": f"Suggest a {self.dim}-dimensional bias vector to improve binding.",
            }
        )

    def _parse_response(self, text: str) -> NDArray | None:
        try:
            match = re.search(r"\{[^}]*\"bias\"\s*:\s*\[([^\]]+)\][^}]*\}", text)
            if match:
                full_match = match.group(0)
                data = json.loads(full_match)
                vec = np.array(data["bias"], dtype=np.float64)
                if len(vec) >= self.dim:
                    vec = vec[: self.dim]
                else:
                    vec = np.pad(vec, (0, self.dim - len(vec)))
                return np.clip(vec, -2.0, 2.0)
        except (json.JSONDecodeError, KeyError, ValueError):
            pass
        return None

    def maybe_call(
        self,
        episode: int,
        receptor_vector: NDArray,
        binding_score: float,
        agent_name: str = "LigandDesigner",
    ) -> NDArray | None:
        """Call LLM every N episodes and return bias vector, or None."""
        if not self.enabled:
            return self._fallback_bias()

        self._call_count += 1
        if self._call_count % self.call_every_n != 0:
            return self._last_bias

        query = self._build_query(receptor_vector, binding_score, episode, agent_name)

        try:
            bias = self._call_claude(query)
            if bias is not None:
                self._last_bias = bias
                self._history.append(
                    {
                        "episode": episode,
                        "binding_score": binding_score,
                        "bias": bias.tolist(),
                    }
                )
            return bias
        except Exception:
            return self._fallback_bias()

    def _call_claude(self, query: str) -> NDArray | None:
        if not _HAS_HTTPX or not self.api_key:
            return self._fallback_bias()

        headers = {
            "x-api-key": self.api_key,
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": self.model,
            "max_tokens": 256,
            "system": SYSTEM_PROMPT.format(dim=self.dim),
            "messages": [{"role": "user", "content": query}],
        }
        resp = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=15.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = data["content"][0]["text"]
            return self._parse_response(text)
        return self._fallback_bias()

    def _fallback_bias(self) -> NDArray | None:
        """Deterministic fallback when API is unavailable."""
        if self._last_bias is not None:
            return self._last_bias
        return None

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
