from __future__ import annotations
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json

from core.system import run_cli

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agents.escape_agent import EscapeAgent
from agents.ligand_agent import LigandDesignerAgent
from agents.receptor_agent import ReceptorMutatorAgent
from api.main import app
from core.receptor import ReceptorState
from core.spaces import Box, Discrete, MultiDiscrete
from core.trainer import AdversarialTrainer, BidirectionalTrainer
from core.utils import RewardNormalizer, clip_l2, engineered_features, gaussian_entropy, gaussian_log_prob, monte_carlo_returns, normalize_advantages, potential_shaping, safe_ratio
from envs.ligand_env import LigandEnv
from envs.receptor_env import ReceptorEnv
from llm.llm_bridge import ClaudeLLMBridge


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def receptor() -> ReceptorState:
    return ReceptorState(dimension=6, wildtype_seed=7)


@pytest.fixture()
def ligand_env(receptor: ReceptorState) -> LigandEnv:
    return LigandEnv(receptor, max_steps=4, seed=3)


@pytest.fixture()
def receptor_env(receptor: ReceptorState) -> ReceptorEnv:
    return ReceptorEnv(receptor, seed=5)


@pytest.fixture()
def ligand_agent(ligand_env: LigandEnv) -> LigandDesignerAgent:
    return LigandDesignerAgent(obs_dim=ligand_env.observation_space.shape[0], action_space=ligand_env.action_space, seed=1)


@pytest.fixture()
def receptor_agent(receptor_env: ReceptorEnv) -> ReceptorMutatorAgent:
    return ReceptorMutatorAgent(obs_dim=receptor_env.observation_space.shape[0], action_space=receptor_env.action_space, seed=2)


@pytest.fixture()
def escape_agent(receptor_env: ReceptorEnv) -> EscapeAgent:
    return EscapeAgent(obs_dim=receptor_env.observation_space.shape[0], action_space=receptor_env.action_space, seed=4)


@pytest.mark.parametrize("shape", [(3,), (5,), (2, 2), (1,)])
def test_box_sample_and_contains(shape):
    box = Box(low=-1.0, high=1.0, shape=shape)
    sample = box.sample(np.random.default_rng(0))
    assert sample.shape == shape
    assert box.contains(sample)


@pytest.mark.parametrize("value", [np.array([0.1, -0.1]), np.array([1.0, -1.0]), np.zeros(2), np.array([0.5, 0.5])])
def test_box_clip_contains(value):
    box = Box(low=-0.5, high=0.5, shape=(2,))
    clipped = box.clip(value)
    assert box.contains(clipped)


@pytest.mark.parametrize("n", [2, 3, 5, 9])
def test_discrete_space(n):
    space = Discrete(n)
    sample = space.sample(np.random.default_rng(1))
    assert space.contains(sample)
    assert space.to_jsonable()["n"] == n


@pytest.mark.parametrize("nvec", [[2, 3], [4, 5, 6], [3], [7, 2]])
def test_multidiscrete_space(nvec):
    space = MultiDiscrete(nvec)
    sample = space.sample(np.random.default_rng(1))
    assert space.contains(sample)
    assert space.shape == np.asarray(nvec).shape


@pytest.mark.parametrize("obs", [np.array([1.0, 2.0]), np.array([0.0, -1.0, 3.0]), np.ones(4), np.arange(5.0)])
def test_engineered_features(obs):
    features = engineered_features(obs)
    assert features.shape[0] == 2 * obs.shape[0] + 1
    assert features[-1] == 1.0


@pytest.mark.parametrize("rewards", [[1.0], [1.0, 2.0], [0.5, 0.5, 0.5], [1.0, -1.0, 2.0]])
def test_monte_carlo_returns(rewards):
    returns = monte_carlo_returns(rewards, gamma=0.9)
    assert returns.shape[0] == len(rewards)
    assert returns[0] >= min(rewards) - 1e-6


@pytest.mark.parametrize("values", [np.array([1.0, 2.0]), np.array([3.0, 3.0]), np.array([-1.0, 1.0, 2.0]), np.array([0.2])])
def test_normalize_advantages(values):
    normalized = normalize_advantages(values)
    assert normalized.shape == values.shape


@pytest.mark.parametrize("pair", [([0.1, 0.2], [0.1, 0.2]), ([1.0, -1.0], [0.5, -0.5]), ([0.0], [0.0]), ([2.0, 3.0], [1.0, 2.0])])
def test_gaussian_helpers(pair):
    action, mean = pair
    sigma = np.ones(len(action)) * 0.5
    assert np.isfinite(gaussian_log_prob(action, mean, sigma))
    assert gaussian_entropy(sigma) > 0


@pytest.mark.parametrize("cap", [0.5, 1.0, 2.0, 5.0])
def test_clip_l2(cap):
    clipped = clip_l2(np.array([3.0, 4.0]), cap)
    assert np.linalg.norm(clipped) <= cap + 1e-6


@pytest.mark.parametrize("pair", [(1.0, 0.5), (0.0, 1.0), (-2.0, 2.0), (3.0, 3.0)])
def test_safe_ratio(pair):
    assert safe_ratio(*pair) >= 1.0


@pytest.mark.parametrize("prev_curr", [(0.0, 0.5), (0.2, 0.4), (0.7, 0.8), (0.9, 0.1)])
def test_potential_shaping(prev_curr):
    assert np.isfinite(potential_shaping(prev_curr[0], prev_curr[1], 0.99))


@pytest.mark.parametrize("reward", [0.1, 0.5, -0.2, 1.4])
def test_reward_normalizer(reward):
    normalizer = RewardNormalizer()
    assert np.isfinite(normalizer.update(reward))


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6])
def test_receptor_reset_and_state(seed):
    local_receptor = ReceptorState(dimension=6, wildtype_seed=seed)
    local_receptor.reset(seed=seed + 10)
    assert local_receptor.clone_vector().shape == (6,)
    assert local_receptor.as_dict()["dimension"] == 6


@pytest.mark.parametrize("delta", [np.ones(6) * 0.1, np.arange(6) * 0.05, np.zeros(6), np.ones(6) * -0.2, np.linspace(-0.1, 0.1, 6), np.eye(1, 6, 0).ravel()])
def test_receptor_mutation_and_functionality(receptor, delta):
    before = receptor.clone_vector()
    receptor.apply_mutation(delta, l2_cap=1.0)
    after = receptor.clone_vector()
    assert after.shape == before.shape
    assert 0.0 < receptor.functionality() <= 1.0


@pytest.mark.parametrize("scale", [0.1, 0.3, 0.5, 0.7, 1.0, 1.3])
def test_binding_oracle_outputs(receptor, scale):
    ligand = receptor.clone_vector() * scale
    result = receptor.binding_oracle(ligand)
    assert 0.0 <= result.binding_score <= 1.0
    assert 0.0 <= result.functionality <= 1.0
    assert isinstance(result.to_dict(), dict)


@pytest.mark.parametrize("count", [1, 2, 3, 4, 5, 6])
def test_receptor_probe_bindings(receptor, count):
    ligands = [np.ones(receptor.dimension) * (i / max(count, 1)) for i in range(count)]
    bindings = receptor.probe_bindings(ligands)
    assert len(bindings) == count


@pytest.mark.parametrize("top_k", [1, 2, 3, 4])
def test_escape_motif_summary(receptor, top_k):
    for factor in [0.1, -0.2, 0.15]:
        receptor.apply_mutation(np.ones(receptor.dimension) * factor)
    motifs = receptor.summarize_escape_motifs(top_k=top_k)
    assert len(motifs) <= top_k


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_ligand_env_reset(ligand_env, seed):
    obs, info = ligand_env.reset(seed=seed)
    assert obs.shape == ligand_env.observation_space.shape
    assert info["env_id"] == ligand_env.env_id


@pytest.mark.parametrize("scale", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
def test_ligand_env_step(ligand_env, scale):
    ligand_env.reset(seed=22)
    action = np.ones(ligand_env.dimension) * scale
    next_obs, reward, done, info = ligand_env.step(action)
    assert next_obs.shape == ligand_env.observation_space.shape
    assert np.isfinite(reward)
    assert isinstance(done, bool)
    assert "reward_breakdown" in info


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_ligand_env_render(ligand_env, idx):
    ligand_env.reset(seed=31 + idx)
    ligand_env.step(np.ones(ligand_env.dimension) * 0.2)
    assert "binding=" in ligand_env.render()


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_receptor_env_reset(receptor_env, idx):
    obs, info = receptor_env.reset()
    assert obs.shape == receptor_env.observation_space.shape
    assert info["env_id"] == receptor_env.env_id


@pytest.mark.parametrize("scale", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
def test_receptor_env_step(receptor_env, scale):
    receptor_env.reset()
    action = np.ones(receptor_env.dimension) * scale
    next_obs, reward, done, info = receptor_env.step(action)
    assert done is True
    assert next_obs.shape == receptor_env.observation_space.shape
    assert "motifs" in info


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_receptor_env_render(receptor_env, idx):
    receptor_env.reset()
    receptor_env.step(np.ones(receptor_env.dimension) * 0.2)
    assert "escape=" in receptor_env.render()


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_ligand_agent_action(ligand_agent, ligand_env, idx):
    obs, _ = ligand_env.reset(seed=50 + idx)
    decision = ligand_agent.select_action(obs)
    assert ligand_env.action_space.contains(decision["action"])
    assert decision["mean"].shape == decision["action"].shape


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_ligand_agent_learning(ligand_agent, ligand_env, idx):
    obs, _ = ligand_env.reset(seed=60 + idx)
    for _ in range(2):
        decision = ligand_agent.select_action(obs)
        next_obs, reward, _, info = ligand_env.step(decision["action"])
        ligand_agent.store_transition(obs, decision["action"], reward, next_obs, decision["mean"], decision["entropy"], info)
        obs = next_obs
    stats = ligand_agent.learn()
    assert np.isfinite(stats.policy_loss)
    assert stats.sigma_mean >= ligand_agent.min_sigma


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_ligand_probe_candidates(ligand_agent, receptor, count):
    probes = ligand_agent.probe_candidates(receptor.clone_vector(), count=count)
    assert len(probes) == count
    assert all(probe.shape == (receptor.dimension,) for probe in probes)


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_receptor_agent_action(receptor_agent, receptor_env, idx):
    obs, _ = receptor_env.reset()
    decision = receptor_agent.select_action(obs)
    assert receptor_env.action_space.contains(decision["action"])


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 4, 5])
def test_receptor_agent_learning(receptor_agent, receptor_env, idx):
    obs, _ = receptor_env.reset()
    decision = receptor_agent.select_action(obs)
    _, reward, _, info = receptor_env.step(decision["action"])
    receptor_agent.store_episode(obs, decision["action"], reward, info)
    stats = receptor_agent.learn()
    assert np.isfinite(stats.policy_loss)
    assert np.isfinite(stats.value_loss)


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_escape_agent_diversity(escape_agent, receptor_env, idx):
    obs, _ = receptor_env.reset()
    decision = escape_agent.select_action(obs)
    _, reward, _, info = receptor_env.step(decision["action"])
    escape_agent.store_episode(obs, decision["action"], reward, info)
    assert escape_agent.diversity_bonus(decision["action"]) >= 0.0


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_claude_bridge_mock(idx):
    bridge = ClaudeLLMBridge()
    call = bridge.generate_bias("agent", {"x": idx}, "objective", action_dim=6, episode=idx)
    assert call.bias_vector.shape == (6,)
    assert isinstance(call.query, dict)


@pytest.mark.parametrize("response_text", [json.dumps({"bias_vector": [0.1, 0.2]}), "0.1 -0.2 0.3", json.dumps({"wrong": []}), "[]"])
def test_claude_bridge_parse_bias(response_text):
    bridge = ClaudeLLMBridge()
    bias = bridge.parse_bias(response_text, action_dim=4)
    assert bias.shape == (4,)


@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_llm_inject_bias(idx):
    bridge = ClaudeLLMBridge(bias_weight=0.25)
    output = bridge.inject_llm_bias(np.ones(4), np.zeros(4))
    assert output.shape == (4,)
    assert np.allclose(output, 0.75 * np.ones(4))


@pytest.mark.parametrize("episodes", [1, 2, 3, 4])
def test_bidirectional_trainer_phase1(episodes):
    trainer = BidirectionalTrainer(dimension=6, max_steps=3, llm_interval=2, seed=episodes)
    summary = trainer.train_phase1(episodes=episodes)
    assert summary.phase == "phase1"
    assert summary.metrics["flow_events"] >= episodes * 2
    assert trainer.best_ligand is not None


@pytest.mark.parametrize("episodes", [1, 2, 3, 4])
def test_adversarial_trainer_phase2(episodes):
    trainer = BidirectionalTrainer(dimension=6, max_steps=3, llm_interval=2, seed=episodes + 20)
    trainer.train_phase1(episodes=3)
    adversarial = AdversarialTrainer(trainer)
    summary = adversarial.train_phase2(episodes=episodes)
    assert summary.phase == "phase2"
    assert summary.metrics["hard_negative_count"] >= episodes


@pytest.mark.parametrize("endpoint", ["/", "/health", "/state"])
def test_api_basic_endpoints(client, endpoint):
    response = client.get(endpoint)
    assert response.status_code == 200


@pytest.mark.parametrize("episodes", [1, 2, 3, 4])
def test_api_phase1(client, episodes):
    response = client.post("/train/phase1", json={"episodes": episodes, "dimension": 6, "max_steps": 3, "llm_interval": 2, "seed": episodes})
    assert response.status_code == 200
    assert response.json()["phase"] == "phase1"


@pytest.mark.parametrize("episodes", [1, 2, 3, 4])
def test_api_phase2(client, episodes):
    client.post("/train/phase1", json={"episodes": 2, "dimension": 6, "max_steps": 3, "llm_interval": 2, "seed": 100 + episodes})
    response = client.post("/train/phase2", json={"episodes": episodes})
    assert response.status_code == 200
    assert response.json()["phase"] == "phase2"


@pytest.mark.parametrize("phase", ["phase1", "phase2", "unknown", "phase1"])
def test_api_flow_endpoint(client, phase):
    client.post("/train/phase1", json={"episodes": 2, "dimension": 6, "max_steps": 3, "llm_interval": 2, "seed": 11})
    client.post("/train/phase2", json={"episodes": 2})
    response = client.get(f"/flow/{phase}")
    assert response.status_code == 200
    assert "events" in response.json()


def test_api_phase2_requires_phase1(client):
    from api.main import _state
    _state["phase1_trainer"] = None
    response = client.post("/train/phase2", json={"episodes": 1})
    assert response.status_code == 400


@pytest.mark.parametrize("args", [["--phase1", "--episodes", "2", "--dimension", "6", "--max-steps", "3"], ["--phase2", "--episodes", "2", "--dimension", "6", "--max-steps", "3"], ["--phase1", "--phase2", "--episodes", "2", "--dimension", "6", "--max-steps", "3"], ["--episodes", "2", "--dimension", "6", "--max-steps", "3"]])
def test_cli_runs(args):
    result = run_cli(args)
    assert result.returncode == 0
    assert "phase" in result.stdout
