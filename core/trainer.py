"""BidirectionalTrainer (Phase 1) and AdversarialTrainer (Phase 2).

Orchestrates the tight co-training loop between Agent_A and Agent_B,
with reward monitoring, entropy floor, and LLM bias injection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from agents.escape_agent import EscapeAgent
from agents.ligand_agent import LigandDesigner, Transition
from agents.receptor_agent import ReceptorMutator
from core.receptor import BindingOracle, ReceptorState
from envs.ligand_env import LigandEnv
from envs.receptor_env import ReceptorEnv
from llm.llm_bridge import LLMBridge

logger = logging.getLogger(__name__)


class BidirectionalTrainer:
    """Phase 1: co-trains LigandDesigner + ReceptorMutator in a tight loop."""

    def __init__(
        self,
        *,
        dim: int = 16,
        max_steps: int = 50,
        episodes: int = 200,
        gamma: float = 0.99,
        binding_threshold: float = 0.85,
        reward_ratio_alert: float = 3.0,
        entropy_floor: float = 0.5,
        noise_injection: float = 0.1,
        llm_call_every: int = 10,
        llm_weight: float = 0.25,
        seed: int = 0,
    ):
        self.dim = dim
        self.max_steps = max_steps
        self.episodes = episodes
        self.gamma = gamma
        self.binding_threshold = binding_threshold
        self.reward_ratio_alert = reward_ratio_alert
        self.entropy_floor = entropy_floor
        self.noise_injection = noise_injection

        self._rng = np.random.default_rng(seed)

        self.receptor_state = ReceptorState(dim=dim, noise_std=0.05)
        self.oracle = BindingOracle(dim=dim)

        self.ligand_env = LigandEnv(
            self.receptor_state,
            self.oracle,
            max_steps=max_steps,
            gamma=gamma,
            binding_threshold=binding_threshold,
            seed=seed,
        )
        self.receptor_env = ReceptorEnv(
            self.receptor_state,
            self.oracle,
            seed=seed + 1,
        )

        obs_dim_a = dim + 2
        obs_dim_b = dim + self.receptor_env.probe_steps
        self.agent_a = LigandDesigner(obs_dim_a, dim, seed=seed + 2)
        self.agent_b = ReceptorMutator(obs_dim_b, dim, seed=seed + 3)

        self.llm_bridge = LLMBridge(
            dim=dim,
            call_every_n=llm_call_every,
            weight=llm_weight,
        )

        # Wire bidirectional hooks
        def probe_hook(receptor_vec: NDArray, n: int) -> NDArray:
            return self.agent_a.get_probe_readings(receptor_vec, n, self.oracle)

        self.receptor_env.set_probe_hook(probe_hook)

        self._metrics: list[dict[str, Any]] = []
        self._best_ligand_state: dict[str, Any] | None = None
        self._best_binding = 0.0

    def _check_entropy_floor(self) -> None:
        ent_a = self.agent_a._entropy()
        ent_b = self.agent_b._entropy()
        if ent_a < self.entropy_floor:
            self.agent_a.log_sigma += self.noise_injection
            logger.info("Agent_A entropy below floor, injecting noise")
        if ent_b < self.entropy_floor:
            self.agent_b.log_sigma += self.noise_injection
            logger.info("Agent_B entropy below floor, injecting noise")

    def _check_reward_balance(
        self, reward_a: float, reward_b: float
    ) -> bool:
        abs_a = abs(reward_a) + 1e-8
        abs_b = abs(reward_b) + 1e-8
        ratio = max(abs_a / abs_b, abs_b / abs_a)
        if ratio > self.reward_ratio_alert:
            logger.warning(
                f"Reward imbalance: ratio={ratio:.2f} (A={reward_a:.4f}, B={reward_b:.4f})"
            )
            return True
        return False

    def train(self, callback: Any = None) -> list[dict[str, Any]]:
        """Run Phase 1 co-training loop."""
        for ep in range(1, self.episodes + 1):
            ep_metrics = self._run_episode(ep)

            if callback:
                callback(ep, ep_metrics)

            self._metrics.append(ep_metrics)
            self._check_entropy_floor()

        return self._metrics

    def _run_episode(self, episode: int) -> dict[str, Any]:
        obs, info = self.ligand_env.reset()
        ep_reward_a = 0.0
        ep_bindings: list[float] = []
        step = 0

        # LLM bias injection
        bias = self.llm_bridge.maybe_call(
            episode,
            self.receptor_state.vector,
            self._best_binding,
            "LigandDesigner",
        )
        if bias is not None:
            self.agent_a.inject_llm_bias(bias, self.llm_bridge.weight)
        else:
            self.agent_a.clear_llm_bias()

        while True:
            action = self.agent_a.act(obs)
            next_obs, reward, terminated, truncated, info = self.ligand_env.step(action)
            done = terminated or truncated

            self.agent_a.store(Transition(obs, action, reward, next_obs, done, info))
            ep_reward_a += reward
            ep_bindings.append(info["binding_score"])
            obs = next_obs
            step += 1

            if done:
                break

        learn_a = self.agent_a.learn()

        # Agent_B: episode-level mutation
        receptor_obs, _ = self.receptor_env.reset()
        mutation = self.agent_b.act(receptor_obs)
        _, reward_b, _, _, info_b = self.receptor_env.step(mutation)
        learn_b = self.agent_b.learn(receptor_obs, mutation, reward_b)

        avg_binding = float(np.mean(ep_bindings)) if ep_bindings else 0.0
        if avg_binding > self._best_binding:
            self._best_binding = avg_binding
            self._best_ligand_state = self.agent_a.get_state()

        self._check_reward_balance(ep_reward_a, reward_b)

        return {
            "episode": episode,
            "reward_a": ep_reward_a,
            "reward_b": float(reward_b),
            "avg_binding": avg_binding,
            "max_binding": float(max(ep_bindings)) if ep_bindings else 0.0,
            "steps": step,
            "receptor_displacement": self.receptor_state.displacement_from_wildtype,
            "learn_a": learn_a,
            "learn_b": learn_b,
        }

    @property
    def best_ligand_state(self) -> dict[str, Any] | None:
        return self._best_ligand_state

    @property
    def metrics(self) -> list[dict[str, Any]]:
        return list(self._metrics)


class AdversarialTrainer:
    """Phase 2: trains EscapeAgent against frozen best-ligand policy."""

    def __init__(
        self,
        *,
        dim: int = 16,
        episodes: int = 100,
        probe_steps: int = 10,
        mutation_cap: float = 3.0,
        diversity_coeff: float = 0.1,
        seed: int = 100,
        frozen_ligand_state: dict[str, Any] | None = None,
    ):
        self.dim = dim
        self.episodes = episodes
        self.probe_steps = probe_steps

        self._rng = np.random.default_rng(seed)
        self.receptor_state = ReceptorState(dim=dim)
        self.oracle = BindingOracle(dim=dim)

        obs_dim_a = dim + 2
        self.frozen_ligand = LigandDesigner(obs_dim_a, dim, seed=seed)
        if frozen_ligand_state:
            self.frozen_ligand.load_state(frozen_ligand_state)

        obs_dim_escape = dim + probe_steps
        self.escape_agent = EscapeAgent(
            obs_dim_escape,
            dim,
            mutation_cap=mutation_cap,
            diversity_coeff=diversity_coeff,
            seed=seed + 1,
        )

        self._metrics: list[dict[str, Any]] = []

    def _compute_disruption(self, receptor_vec: NDArray) -> float:
        """Measure how much the mutation disrupts the frozen ligand's binding."""
        total = 0.0
        for _ in range(self.probe_steps):
            obs = np.concatenate([receptor_vec, [0.5, 0.0]])
            ligand = self.frozen_ligand.act(obs, deterministic=True)
            score = self.oracle.score(ligand, receptor_vec)
            total += (1.0 - score)
        return total / self.probe_steps

    def train(self, callback: Any = None) -> list[dict[str, Any]]:
        for ep in range(1, self.episodes + 1):
            ep_metrics = self._run_episode(ep)
            self._metrics.append(ep_metrics)
            if callback:
                callback(ep, ep_metrics)
        return self._metrics

    def _run_episode(self, episode: int) -> dict[str, Any]:
        self.receptor_state.reset(self._rng)

        probes = self.frozen_ligand.get_probe_readings(
            self.receptor_state.vector, self.probe_steps, self.oracle
        )
        obs = np.concatenate([self.receptor_state.vector, probes])

        mutation = self.escape_agent.act(obs)
        self.receptor_state.mutate(mutation)

        disruption = self._compute_disruption(self.receptor_state.vector)
        learn_info = self.escape_agent.learn(obs, mutation, disruption)

        return {
            "episode": episode,
            "disruption": disruption,
            "receptor_displacement": self.receptor_state.displacement_from_wildtype,
            **learn_info,
        }

    @property
    def metrics(self) -> list[dict[str, Any]]:
        return list(self._metrics)

    @property
    def hard_negatives(self) -> list[NDArray]:
        return self.escape_agent.get_hard_negatives()
