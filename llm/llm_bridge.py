from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Any
from urllib import request

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LLMCallResult:
    query: dict[str, Any]
    response_text: str
    bias_vector: np.ndarray
    used_remote_api: bool


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class ClaudeLLMBridge:
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None, bias_weight: float = 0.25, timeout_seconds: int = 20) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("CLAUDE_API_KEY")
        self.bias_weight = bias_weight
        self.timeout_seconds = timeout_seconds

    def build_query(self, agent_name: str, state: dict[str, Any], objective: str, action_dim: int, episode: int, step: int = 0) -> dict[str, Any]:
        return {
            "flow_stage": "input(query)",
            "agent_name": agent_name,
            "objective": objective,
            "episode": episode,
            "step": step,
            "action_dim": action_dim,
            "state": _sanitize_for_json(state),
            "instruction": "Return JSON with a bias_vector field containing action_dim floats.",
        }

    def _mock_response(self, query: dict[str, Any]) -> str:
        digest = sha256(json.dumps(query, sort_keys=True, default=str).encode("utf-8")).digest()
        action_dim = int(query["action_dim"])
        values = []
        for index in range(action_dim):
            byte = digest[index % len(digest)]
            values.append(((byte / 255.0) * 2.0) - 1.0)
        return json.dumps({"bias_vector": values, "mode": "mock"})

    def _remote_response(self, query: dict[str, Any]) -> str:
        safe_query = _sanitize_for_json(query)
        payload = {
            "model": self.model,
            "system": "You are a chemistry and optimization assistant. Output JSON only.",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": json.dumps(safe_query)}],
        }
        req = request.Request(
            "https://api.anthropic.com/v1/messages",
            method="POST",
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key or "",
                "anthropic-version": "2023-06-01",
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = response.read().decode("utf-8")
        parsed = json.loads(body)
        content = parsed.get("content", [])
        if content and isinstance(content, list):
            first = content[0]
            if isinstance(first, dict):
                return str(first.get("text", ""))
        return body

    def parse_bias(self, response_text: str, action_dim: int) -> np.ndarray:
        bias = np.zeros(action_dim, dtype=np.float64)
        try:
            parsed = json.loads(response_text)
            vector = parsed.get("bias_vector", [])
            if isinstance(vector, list):
                for index, value in enumerate(vector[:action_dim]):
                    bias[index] = float(value)
        except Exception:
            numbers = []
            token = ""
            for char in response_text:
                if char in "-+.0123456789eE":
                    token += char
                else:
                    if token:
                        try:
                            numbers.append(float(token))
                        except ValueError:
                            pass
                        token = ""
            if token:
                try:
                    numbers.append(float(token))
                except ValueError:
                    pass
            for index, value in enumerate(numbers[:action_dim]):
                bias[index] = value
        return bias

    def generate_bias(self, agent_name: str, state: dict[str, Any], objective: str, action_dim: int, episode: int, step: int = 0) -> LLMCallResult:
        query = self.build_query(agent_name, state, objective, action_dim, episode, step)
        if self.api_key:
            try:
                response_text = self._remote_response(query)
                logger.info("LLM remote call succeeded for %s ep=%d", agent_name, episode)
                return LLMCallResult(query, response_text, self.parse_bias(response_text, action_dim), True)
            except Exception as exc:
                logger.warning("LLM remote call failed for %s ep=%d: %s — using mock fallback", agent_name, episode, exc)
                response_text = self._mock_response(query)
                return LLMCallResult(query, response_text, self.parse_bias(response_text, action_dim), False)
        response_text = self._mock_response(query)
        return LLMCallResult(query, response_text, self.parse_bias(response_text, action_dim), False)

    def noop_result(self, agent_name: str, state: dict[str, Any], objective: str, action_dim: int, episode: int, step: int = 0) -> LLMCallResult:
        """Return a zero-bias result for non-LLM episodes (replaces dynamic type hack)."""
        query = self.build_query(agent_name, state, objective, action_dim, episode, step)
        return LLMCallResult(query, "{}", np.zeros(action_dim, dtype=np.float64), False)

    def inject_llm_bias(self, policy_action: np.ndarray, bias_vector: np.ndarray, weight: float | None = None) -> np.ndarray:
        used_weight = self.bias_weight if weight is None else weight
        return (1.0 - used_weight) * np.asarray(policy_action, dtype=np.float64) + used_weight * np.asarray(bias_vector, dtype=np.float64)
