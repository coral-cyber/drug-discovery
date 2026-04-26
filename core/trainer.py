from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from agents.escape_agent import EscapeAgent
from agents.ligand_agent import LigandDesignerAgent
from agents.receptor_agent import ReceptorMutatorAgent
from core.receptor import ReceptorState
from core.utils import RewardNormalizer, safe_ratio
from envs.ligand_env import LigandEnv
from envs.receptor_env import ReceptorEnv
from llm.llm_bridge import ClaudeLLMBridge


@dataclass
class PhaseSummary:
    phase: str
    episodes: int
    metrics: dict[str, Any]


class BidirectionalTrainer:
    def __init__(self, dimension: int = 8, max_steps: int = 6, llm_interval: int = 2, seed: int = 0, min_entropy: float = 1.2) -> None:
        self.dimension = dimension
        self.max_steps = max_steps
        self.llm_interval = max(1, llm_interval)
        self.seed = seed
        self.min_entropy = min_entropy
        self.receptor = ReceptorState(dimension=dimension, wildtype_seed=seed + 11)
        self.ligand_env = LigandEnv(self.receptor, max_steps=max_steps, seed=seed)
        self.receptor_env = ReceptorEnv(self.receptor, seed=seed)
        self.ligand_agent = LigandDesignerAgent(obs_dim=self.ligand_env.observation_space.shape[0], action_space=self.ligand_env.action_space, seed=seed)
        self.receptor_agent = ReceptorMutatorAgent(obs_dim=self.receptor_env.observation_space.shape[0], action_space=self.receptor_env.action_space, seed=seed + 1)
        self.llm_bridge = ClaudeLLMBridge()
        self.ligand_rewards = RewardNormalizer()
        self.receptor_rewards = RewardNormalizer()
        self.flow_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.alerts: list[str] = []
        self.best_ligand: np.ndarray | None = None
        self.best_ligand_score = -float("inf")
        self.best_ligand_record: dict[str, Any] = {}
        self.ligand_env.set_transition_hook(self.preview_receptor_transition)
        self.receptor_env.set_probe_hook(self._ligand_probe_hook)

    def preview_receptor_transition(self, context: dict[str, Any]) -> dict[str, Any]:
        probe_readings = self.receptor_env._probe_readings()
        obs = np.concatenate([self.receptor.clone_vector(), np.asarray(probe_readings, dtype=np.float64)])
        preview = self.receptor_agent.policy_mean(obs)
        return {"proposed_mutation": self.receptor_env.action_space.clip(preview).round(6).tolist(), "source": "ReceptorMutator preview"}

    def _ligand_probe_hook(self, count: int) -> list[np.ndarray]:
        return self.ligand_agent.probe_candidates(self.receptor.clone_vector(), count=count)

    def _should_query_llm(self, episode: int) -> bool:
        return episode % self.llm_interval == 0

    def _balance_rewards(self, ligand_reward: float, receptor_reward: float) -> None:
        ligand_norm = self.ligand_rewards.update(ligand_reward)
        receptor_norm = self.receptor_rewards.update(receptor_reward)
        ratio = safe_ratio(ligand_norm + 1e-6, receptor_norm + 1e-6)
        if ratio > 3.0:
            self.alerts.append(f"reward dominance alert ratio={ratio:.2f}")

    def _entropy_guards(self) -> None:
        if self.ligand_agent.ensure_entropy_floor(self.min_entropy):
            self.alerts.append("ligand entropy floor activated")
        if self.receptor_agent.ensure_entropy_floor(self.min_entropy):
            self.alerts.append("receptor entropy floor activated")

    def _record_flow(self, env_name: str, agent_name: str, llm_result: dict[str, Any], raw_action: np.ndarray, biased_action: np.ndarray, reward: float, info: dict[str, Any]) -> None:
        self.flow_history.append(
            {
                "flow": ["input(query)", "agent", "LLM", "openenv", "reward", "biased_output", "repeat"],
                "env": env_name,
                "agent": agent_name,
                "query": llm_result["query"],
                "llm_response": llm_result["response_text"],
                "used_remote_api": llm_result["used_remote_api"],
                "agent_output": np.asarray(raw_action, dtype=np.float64).round(6).tolist(),
                "reward": float(reward),
                "biased_output": np.asarray(biased_action, dtype=np.float64).round(6).tolist(),
                "info": info,
            }
        )

    def train_phase1(self, episodes: int = 6) -> PhaseSummary:
        for episode in range(1, episodes + 1):
            observation, _ = self.ligand_env.reset(seed=self.seed + episode)
            episode_reward = 0.0
            best_episode_binding = -float("inf")
            best_episode_ligand = None
            binding_trace: list[float] = []

            for step in range(self.max_steps):
                if self._should_query_llm(episode):
                    llm_call = self.llm_bridge.generate_bias("LigandDesigner", self.ligand_env.get_state(), "maximize stable selective binding", self.dimension, episode, step)
                    ligand_decision = self.ligand_agent.select_action(observation, llm_bias=llm_call.bias_vector)
                else:
                    query = self.llm_bridge.build_query("LigandDesigner", self.ligand_env.get_state(), "maximize stable selective binding", self.dimension, episode, step)
                    llm_call = type("Call", (), {"query": query, "response_text": "{}", "bias_vector": np.zeros(self.dimension), "used_remote_api": False})
                    ligand_decision = self.ligand_agent.select_action(observation)
                next_obs, reward, done, info = self.ligand_env.step(ligand_decision["action"])
                self.ligand_agent.store_transition(observation, ligand_decision["action"], reward, next_obs, ligand_decision["mean"], ligand_decision["entropy"], info)
                self._record_flow(self.ligand_env.env_id, "LigandDesigner", {"query": llm_call.query, "response_text": llm_call.response_text, "used_remote_api": llm_call.used_remote_api}, ligand_decision["raw_action"], ligand_decision["action"], reward, info)
                observation = next_obs
                episode_reward += reward
                binding_trace.append(info["binding_score"])
                if info["binding_result"]["total_score"] > self.best_ligand_score:
                    self.best_ligand_score = info["binding_result"]["total_score"]
                    self.best_ligand = ligand_decision["action"].copy()
                    self.best_ligand_record = {"ligand": self.best_ligand.round(6).tolist(), "binding_score": info["binding_result"]["binding_score"], "selectivity": info["binding_result"]["selectivity"], "episode": episode}
                if info["binding_score"] > best_episode_binding:
                    best_episode_binding = info["binding_score"]
                    best_episode_ligand = ligand_decision["action"].copy()
                if done:
                    break

            ligand_stats = self.ligand_agent.learn()
            receptor_obs, receptor_reset_info = self.receptor_env.reset()
            if self._should_query_llm(episode):
                receptor_llm = self.llm_bridge.generate_bias("ReceptorMutator", receptor_reset_info["full_state"], "disrupt ligand binding while preserving receptor functionality", self.dimension, episode)
                receptor_decision = self.receptor_agent.select_action(receptor_obs, llm_bias=receptor_llm.bias_vector)
            else:
                query = self.llm_bridge.build_query("ReceptorMutator", receptor_reset_info["full_state"], "disrupt ligand binding while preserving receptor functionality", self.dimension, episode)
                receptor_llm = type("Call", (), {"query": query, "response_text": "{}", "bias_vector": np.zeros(self.dimension), "used_remote_api": False})
                receptor_decision = self.receptor_agent.select_action(receptor_obs)
            _, receptor_reward, _, receptor_info = self.receptor_env.step(receptor_decision["action"])
            self.receptor_agent.store_episode(receptor_obs, receptor_decision["action"], receptor_reward, receptor_info)
            receptor_stats = self.receptor_agent.learn()
            self._record_flow(self.receptor_env.env_id, "ReceptorMutator", {"query": receptor_llm.query, "response_text": receptor_llm.response_text, "used_remote_api": receptor_llm.used_remote_api}, receptor_decision["raw_action"], receptor_decision["action"], receptor_reward, receptor_info)
            self._balance_rewards(ligand_stats.reward_mean, receptor_stats.reward_mean)
            self._entropy_guards()
            self.metrics_history.append(
                {
                    "episode": episode,
                    "ligand_reward": episode_reward,
                    "avg_binding": float(np.mean(binding_trace)) if binding_trace else 0.0,
                    "best_episode_binding": best_episode_binding,
                    "best_episode_ligand": best_episode_ligand.round(6).tolist() if best_episode_ligand is not None else None,
                    "receptor_reward": receptor_reward,
                    "ligand_stats": ligand_stats.__dict__,
                    "receptor_stats": receptor_stats.__dict__,
                    "shared_state": self.receptor.as_dict(),
                }
            )

        equilibrium = self.metrics_history[-1] if self.metrics_history else {}
        return PhaseSummary(
            phase="phase1",
            episodes=episodes,
            metrics={
                "best_ligand": self.best_ligand_record,
                "equilibrium": equilibrium,
                "reward_alerts": self.alerts,
                "flow_events": len(self.flow_history),
                "shared_state": self.receptor.as_dict(),
            },
        )


class AdversarialTrainer:
    def __init__(self, base_trainer: BidirectionalTrainer, llm_interval: int | None = None, min_entropy: float = 1.2) -> None:
        self.base_trainer = base_trainer
        self.receptor = base_trainer.receptor
        self.best_ligand = np.asarray(base_trainer.best_ligand, dtype=np.float64) if base_trainer.best_ligand is not None else np.zeros(base_trainer.dimension, dtype=np.float64)
        self.receptor_env = ReceptorEnv(self.receptor, seed=base_trainer.seed + 99)
        self.escape_agent = EscapeAgent(obs_dim=self.receptor_env.observation_space.shape[0], action_space=self.receptor_env.action_space, seed=base_trainer.seed + 7)
        self.llm_bridge = ClaudeLLMBridge()
        self.llm_interval = llm_interval or base_trainer.llm_interval
        self.min_entropy = min_entropy
        self.flow_history: list[dict[str, Any]] = []
        self.metrics_history: list[dict[str, Any]] = []
        self.reward_normalizer = RewardNormalizer()
        self.receptor_env.set_probe_hook(self._fixed_probe_hook)

    def _fixed_probe_hook(self, count: int) -> list[np.ndarray]:
        return [self.best_ligand.copy() for _ in range(count)]

    def train_phase2(self, episodes: int = 5) -> PhaseSummary:
        for episode in range(1, episodes + 1):
            receptor_obs, reset_info = self.receptor_env.reset()
            if episode % self.llm_interval == 0:
                llm_call = self.llm_bridge.generate_bias("EscapeAgent", reset_info["full_state"], "maximize binding disruption against the frozen best ligand", self.base_trainer.dimension, episode)
                decision = self.escape_agent.select_action(receptor_obs, llm_bias=llm_call.bias_vector)
            else:
                query = self.llm_bridge.build_query("EscapeAgent", reset_info["full_state"], "maximize binding disruption against the frozen best ligand", self.base_trainer.dimension, episode)
                llm_call = type("Call", (), {"query": query, "response_text": "{}", "bias_vector": np.zeros(self.base_trainer.dimension), "used_remote_api": False})
                decision = self.escape_agent.select_action(receptor_obs)
            _, reward, _, info = self.receptor_env.step(decision["action"])
            fixed_binding = self.receptor.binding_oracle(self.best_ligand).binding_score
            disruption_reward = float((1.0 - fixed_binding) + self.escape_agent.diversity_bonus(decision["action"]))
            total_reward = reward + disruption_reward
            info["binding_disruption"] = 1.0 - fixed_binding
            self.escape_agent.store_episode(receptor_obs, decision["action"], total_reward, info)
            stats = self.escape_agent.learn()
            self.reward_normalizer.update(total_reward)
            if self.escape_agent.ensure_entropy_floor(self.min_entropy):
                self.base_trainer.alerts.append("escape entropy floor activated")
            self.base_trainer.ligand_agent.add_hard_negative(info["mutated_receptor"])
            self.flow_history.append({"flow": ["input(query)", "agent", "LLM", "openenv", "reward", "biased_output", "repeat"], "query": llm_call.query, "llm_response": llm_call.response_text, "reward": total_reward, "biased_output": np.asarray(decision["action"], dtype=np.float64).round(6).tolist(), "info": info})
            self.metrics_history.append({"episode": episode, "reward": total_reward, "binding_disruption": info["binding_disruption"], "motifs": info["motifs"], "stats": stats.__dict__})
        motifs = self.receptor.summarize_escape_motifs(top_k=5)
        return PhaseSummary(
            phase="phase2",
            episodes=episodes,
            metrics={
                "escape_motifs": motifs,
                "binding_site_counter_strategies": [
                    "increase receptor loop flexibility within functionality floor",
                    "shift dominant contact sites away from the frozen ligand centroid",
                    "reuse high-diversity mutations as hard negatives for the ligand learner",
                ],
                "hard_negative_count": len(self.base_trainer.ligand_agent.hard_negatives),
                "flow_events": len(self.flow_history),
            },
        )
