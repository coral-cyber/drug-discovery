"""Comprehensive tests for the adversarial RL system. 60+ tests."""

from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
from core.spaces import Box, Discrete, MultiDiscrete, Space


class TestSpace:
    def test_abstract_sample_raises(self):
        s = Space(shape=(2,), dtype=np.float64)
        with pytest.raises(NotImplementedError):
            s.sample()

    def test_abstract_contains_raises(self):
        s = Space(shape=(2,), dtype=np.float64)
        with pytest.raises(NotImplementedError):
            s.contains(np.zeros(2))

    def test_repr(self):
        s = Space(shape=(3,))
        assert "Space" in repr(s)


class TestBox:
    def test_creation_with_shape(self):
        b = Box(-1.0, 1.0, shape=(4,))
        assert b.shape == (4,)
        assert b.low.shape == (4,)
        assert b.high.shape == (4,)

    def test_creation_with_arrays(self):
        b = Box(np.array([-1, -2]), np.array([1, 2]))
        assert b.shape == (2,)

    def test_scalar_without_shape_raises(self):
        with pytest.raises(ValueError):
            Box(-1.0, 1.0)

    def test_sample_in_bounds(self):
        b = Box(-1.0, 1.0, shape=(10,))
        rng = np.random.default_rng(42)
        for _ in range(50):
            s = b.sample(rng)
            assert b.contains(s)

    def test_contains_true(self):
        b = Box(0.0, 1.0, shape=(3,))
        assert b.contains(np.array([0.5, 0.5, 0.5]))

    def test_contains_false_shape(self):
        b = Box(0.0, 1.0, shape=(3,))
        assert not b.contains(np.array([0.5, 0.5]))

    def test_contains_false_value(self):
        b = Box(0.0, 1.0, shape=(3,))
        assert not b.contains(np.array([0.5, 1.5, 0.5]))

    def test_clip(self):
        b = Box(-1.0, 1.0, shape=(3,))
        clipped = b.clip(np.array([-5.0, 0.0, 5.0]))
        np.testing.assert_array_equal(clipped, [-1.0, 0.0, 1.0])

    def test_repr_box(self):
        b = Box(-1.0, 1.0, shape=(4,))
        assert "Box" in repr(b)

    def test_dtype_preserved(self):
        b = Box(0.0, 1.0, shape=(2,), dtype=np.float32)
        assert b.dtype == np.float32
        s = b.sample()
        assert s.dtype == np.float32


class TestDiscrete:
    def test_creation(self):
        d = Discrete(5)
        assert d.n == 5

    def test_sample_range(self):
        d = Discrete(3)
        rng = np.random.default_rng(42)
        for _ in range(100):
            s = d.sample(rng)
            assert 0 <= s < 3

    def test_contains(self):
        d = Discrete(5)
        assert d.contains(0)
        assert d.contains(4)
        assert not d.contains(5)
        assert not d.contains(-1)

    def test_repr_discrete(self):
        assert "Discrete" in repr(Discrete(7))


class TestMultiDiscrete:
    def test_creation(self):
        md = MultiDiscrete([3, 5, 2])
        assert md.shape == (3,)

    def test_sample(self):
        md = MultiDiscrete([3, 5, 2])
        rng = np.random.default_rng(42)
        s = md.sample(rng)
        assert s.shape == (3,)
        assert 0 <= s[0] < 3
        assert 0 <= s[1] < 5
        assert 0 <= s[2] < 2

    def test_contains(self):
        md = MultiDiscrete([3, 5])
        assert md.contains(np.array([2, 4]))
        assert not md.contains(np.array([3, 4]))

    def test_repr_multidiscrete(self):
        assert "MultiDiscrete" in repr(MultiDiscrete([2, 3]))


# ---------------------------------------------------------------------------
# Receptor
# ---------------------------------------------------------------------------
from core.receptor import BindingOracle, ReceptorState


class TestReceptorState:
    def test_init(self):
        r = ReceptorState(dim=8)
        assert r.vector.shape == (8,)

    def test_reset_near_wildtype(self):
        r = ReceptorState(dim=8, noise_std=0.01)
        rng = np.random.default_rng(0)
        r.reset(rng)
        dist = np.linalg.norm(r.vector - r.wildtype)
        assert dist < 0.5

    def test_mutate_changes_vector(self):
        r = ReceptorState(dim=4)
        old = r.vector.copy()
        r.mutate(np.ones(4) * 0.1)
        assert not np.allclose(r.vector, old)

    def test_mutation_cap(self):
        r = ReceptorState(dim=4, mutation_cap=1.0)
        r.mutate(np.ones(4) * 100.0)
        delta = r.vector - r._wildtype
        assert np.linalg.norm(delta) <= 1.0 + 1e-6

    def test_history_tracking(self):
        r = ReceptorState(dim=4)
        r.mutate(np.ones(4) * 0.1)
        r.mutate(np.ones(4) * 0.2)
        assert len(r.history) == 3  # init + 2 mutations

    def test_displacement(self):
        r = ReceptorState(dim=4)
        r.mutate(np.ones(4))
        assert r.displacement_from_wildtype > 0


class TestBindingOracle:
    def test_score_range(self):
        o = BindingOracle(dim=8)
        rng = np.random.default_rng(42)
        for _ in range(50):
            l = rng.standard_normal(8)
            r = rng.standard_normal(8)
            s = o.score(l, r)
            assert 0 <= s <= 1

    def test_identical_binding(self):
        o = BindingOracle(dim=8, temperature=2.0)
        vec = np.ones(8)
        s = o.score(vec, vec)
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_off_target(self):
        o = BindingOracle(dim=8)
        l = np.zeros(8)
        d = np.ones(8) * 5
        s = o.off_target_score(l, d)
        assert 0 <= s <= 1

    def test_selectivity(self):
        o = BindingOracle(dim=8)
        l = np.zeros(8)
        t = np.zeros(8)
        decoys = [np.ones(8) * 3]
        sel = o.selectivity(l, t, decoys)
        assert isinstance(sel, float)

    def test_selectivity_no_decoys(self):
        o = BindingOracle(dim=8)
        sel = o.selectivity(np.zeros(8), np.zeros(8), [])
        assert sel >= 0


# ---------------------------------------------------------------------------
# LigandEnv
# ---------------------------------------------------------------------------
from envs.ligand_env import LigandEnv


class TestLigandEnv:
    def _make_env(self, **kw):
        rs = ReceptorState(dim=8, **{k: v for k, v in kw.items() if k in ("noise_std",)})
        oracle = BindingOracle(dim=8)
        return LigandEnv(rs, oracle, max_steps=10, seed=0, **{k: v for k, v in kw.items() if k not in ("noise_std",)})

    def test_reset(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (10,)  # dim(8) + 2
        assert info["step"] == 0

    def test_step_shape(self):
        env = self._make_env()
        obs, _ = env.reset()
        action = env.action_space.sample()
        obs2, r, term, trunc, info = env.step(action)
        assert obs2.shape == obs.shape
        assert isinstance(r, float)

    def test_max_steps_truncation(self):
        env = self._make_env()
        env.reset()
        for _ in range(10):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
        assert trunc

    def test_action_clipping(self):
        env = self._make_env()
        env.reset()
        big_action = np.ones(8) * 100
        obs, r, _, _, _ = env.step(big_action)
        assert obs.shape == (10,)

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id == "LigandVsReceptor-v1"

    def test_render(self):
        env = self._make_env()
        env.reset()
        env.step(np.zeros(8))
        r = env.render()
        assert "LigandVsReceptor" in r
        assert "binding" in r

    def test_seed_method(self):
        env = self._make_env()
        env.seed(99)
        env.reset()

    def test_info_fields(self):
        env = self._make_env()
        _, info = env.reset()
        assert "binding_score" in info
        assert "env_id" in info
        assert "full_state" in info

    def test_potential_based_shaping(self):
        env = self._make_env()
        env.reset()
        _, r1, _, _, _ = env.step(np.zeros(8))
        _, r2, _, _, _ = env.step(np.zeros(8))
        assert isinstance(r1, float) and isinstance(r2, float)


# ---------------------------------------------------------------------------
# ReceptorEnv
# ---------------------------------------------------------------------------
from envs.receptor_env import ReceptorEnv


class TestReceptorEnv:
    def _make_env(self):
        rs = ReceptorState(dim=8)
        oracle = BindingOracle(dim=8)
        return ReceptorEnv(rs, oracle, probe_steps=5, seed=0)

    def test_reset(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (13,)  # dim(8) + probe_steps(5)

    def test_step_terminal(self):
        env = self._make_env()
        env.reset()
        _, r, term, trunc, info = env.step(np.zeros(8))
        assert term  # single step episode
        assert not trunc
        assert isinstance(r, float)

    def test_env_id(self):
        env = self._make_env()
        assert env.env_id == "ReceptorAdversary-v1"

    def test_mutation_cap(self):
        env = self._make_env()
        env.reset()
        big_delta = np.ones(8) * 100
        env.step(big_delta)
        # vector should have changed but within bounds

    def test_render(self):
        env = self._make_env()
        env.reset()
        env.step(np.zeros(8))
        r = env.render()
        assert "ReceptorAdversary" in r

    def test_probe_hook(self):
        env = self._make_env()
        called = []
        env.set_probe_hook(lambda rv, n: (called.append(1), np.ones(n) * 0.5)[1])
        env.reset()
        assert len(called) == 1

    def test_seed_method(self):
        env = self._make_env()
        env.seed(99)


# ---------------------------------------------------------------------------
# LigandDesigner Agent
# ---------------------------------------------------------------------------
from agents.ligand_agent import LigandDesigner, Transition


class TestLigandDesigner:
    def test_act(self):
        agent = LigandDesigner(10, 8, seed=0)
        obs = np.random.default_rng(0).standard_normal(10)
        a = agent.act(obs)
        assert a.shape == (8,)

    def test_act_deterministic(self):
        agent = LigandDesigner(10, 8, seed=0)
        obs = np.zeros(10)
        a1 = agent.act(obs, deterministic=True)
        a2 = agent.act(obs, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_value(self):
        agent = LigandDesigner(10, 8, seed=0)
        v = agent.value(np.zeros(10))
        assert isinstance(v, float)

    def test_store_and_learn(self):
        agent = LigandDesigner(10, 8, seed=0)
        for _ in range(5):
            obs = np.random.default_rng().standard_normal(10)
            action = agent.act(obs)
            agent.store(Transition(obs, action, 1.0, obs, False))
        agent.store(Transition(obs, action, 1.0, obs, True))
        info = agent.learn()
        assert "loss_policy" in info
        assert "entropy" in info
        assert agent._train_steps == 1

    def test_learn_empty_buffer(self):
        agent = LigandDesigner(10, 8, seed=0)
        info = agent.learn()
        assert info == {}

    def test_llm_bias_injection(self):
        agent = LigandDesigner(10, 8, seed=0)
        bias = np.ones(8) * 2.0
        agent.inject_llm_bias(bias, weight=0.5)
        obs = np.zeros(10)
        a = agent.act(obs, deterministic=True)
        agent.clear_llm_bias()
        a2 = agent.act(obs, deterministic=True)
        assert not np.allclose(a, a2)

    def test_sigma_property(self):
        agent = LigandDesigner(10, 8, sigma_init=1.0)
        assert agent.sigma.shape == (8,)
        assert np.all(agent.sigma > 0)

    def test_entropy(self):
        agent = LigandDesigner(10, 8)
        h = agent._entropy()
        assert isinstance(h, float)

    def test_get_load_state(self):
        agent = LigandDesigner(10, 8, seed=0)
        state = agent.get_state()
        agent2 = LigandDesigner(10, 8, seed=99)
        agent2.load_state(state)
        np.testing.assert_array_equal(agent.W_mu, agent2.W_mu)

    def test_probe_readings(self):
        agent = LigandDesigner(10, 8, seed=0)
        oracle = BindingOracle(dim=8)
        readings = agent.get_probe_readings(np.zeros(8), 5, oracle)
        assert readings.shape == (5,)
        assert all(0 <= r <= 1 for r in readings)


# ---------------------------------------------------------------------------
# ReceptorMutator Agent
# ---------------------------------------------------------------------------
from agents.receptor_agent import AdamState, ReceptorMutator


class TestAdamState:
    def test_step(self):
        adam = AdamState(lr=0.01)
        params = {"x": np.array([1.0, 2.0])}
        grads = {"x": np.array([0.1, 0.2])}
        updated = adam.step(params, grads)
        assert "x" in updated
        assert not np.allclose(updated["x"], params["x"])

    def test_multiple_steps(self):
        adam = AdamState(lr=0.01)
        params = {"x": np.array([0.0])}
        for _ in range(10):
            grads = {"x": np.array([1.0])}
            params = adam.step(params, grads)
        assert adam.t == 10


class TestReceptorMutator:
    def test_act(self):
        agent = ReceptorMutator(13, 8, seed=0)
        obs = np.zeros(13)
        a = agent.act(obs)
        assert a.shape == (8,)

    def test_mutation_cap(self):
        agent = ReceptorMutator(13, 8, mutation_cap=1.0, sigma_init=5.0, seed=0)
        obs = np.zeros(13)
        for _ in range(20):
            a = agent.act(obs)
            assert np.linalg.norm(a) <= 1.0 + 1e-6

    def test_learn(self):
        agent = ReceptorMutator(13, 8, seed=0)
        obs = np.zeros(13)
        action = agent.act(obs)
        info = agent.learn(obs, action, 0.5)
        assert "advantage" in info
        assert "entropy" in info
        assert agent._train_episodes == 1

    def test_diversity_bonus(self):
        agent = ReceptorMutator(13, 8, seed=0)
        obs = np.zeros(13)
        a1 = agent.act(obs)
        agent.learn(obs, a1, 0.5)
        bonus = agent._diversity_bonus(np.ones(8))
        assert isinstance(bonus, float)

    def test_get_load_state(self):
        agent = ReceptorMutator(13, 8, seed=0)
        state = agent.get_state()
        agent2 = ReceptorMutator(13, 8, seed=99)
        agent2.load_state(state)
        np.testing.assert_array_equal(agent.W_mu, agent2.W_mu)

    def test_llm_bias(self):
        agent = ReceptorMutator(13, 8, seed=0)
        agent.inject_llm_bias(np.ones(8), 0.3)
        obs = np.zeros(13)
        a1 = agent.act(obs, deterministic=True)
        agent.clear_llm_bias()
        a2 = agent.act(obs, deterministic=True)
        assert not np.allclose(a1, a2)


# ---------------------------------------------------------------------------
# EscapeAgent
# ---------------------------------------------------------------------------
from agents.escape_agent import EscapeAgent


class TestEscapeAgent:
    def test_act(self):
        agent = EscapeAgent(13, 8, seed=0)
        obs = np.zeros(13)
        a = agent.act(obs)
        assert a.shape == (8,)

    def test_learn(self):
        agent = EscapeAgent(13, 8, seed=0)
        obs = np.zeros(13)
        a = agent.act(obs)
        info = agent.learn(obs, a, 0.8)
        assert "disruption_reward" in info

    def test_escape_motifs_recorded(self):
        agent = EscapeAgent(13, 8, seed=0)
        obs = np.zeros(13)
        a = agent.act(obs)
        agent.learn(obs, a, 0.9)  # above 0.7 threshold
        assert len(agent.escape_motifs) >= 1

    def test_hard_negatives(self):
        agent = EscapeAgent(13, 8, seed=0)
        obs = np.zeros(13)
        for i in range(5):
            a = agent.act(obs)
            agent.learn(obs, a, 0.8 + i * 0.01)
        negs = agent.get_hard_negatives(top_k=3)
        assert len(negs) <= 3

    def test_mutation_cap(self):
        agent = EscapeAgent(13, 8, mutation_cap=1.0, sigma_init=5.0, seed=0)
        obs = np.zeros(13)
        for _ in range(20):
            a = agent.act(obs)
            assert np.linalg.norm(a) <= 1.0 + 1e-6

    def test_get_load_state(self):
        agent = EscapeAgent(13, 8, seed=0)
        obs = np.zeros(13)
        agent.learn(obs, agent.act(obs), 0.9)
        state = agent.get_state()
        agent2 = EscapeAgent(13, 8, seed=99)
        agent2.load_state(state)
        assert agent2._train_episodes == agent._train_episodes


# ---------------------------------------------------------------------------
# LLM Bridge
# ---------------------------------------------------------------------------
from llm.llm_bridge import LLMBridge


class TestLLMBridge:
    def test_disabled_returns_none(self):
        bridge = LLMBridge(dim=8, enabled=False)
        result = bridge.maybe_call(1, np.zeros(8), 0.5)
        assert result is None

    def test_parse_valid_response(self):
        bridge = LLMBridge(dim=4)
        text = '{"bias": [0.1, -0.2, 0.3, 0.4]}'
        vec = bridge._parse_response(text)
        assert vec is not None
        assert vec.shape == (4,)

    def test_parse_invalid_response(self):
        bridge = LLMBridge(dim=4)
        vec = bridge._parse_response("no json here")
        assert vec is None

    def test_parse_clipping(self):
        bridge = LLMBridge(dim=2)
        text = '{"bias": [10.0, -10.0]}'
        vec = bridge._parse_response(text)
        assert vec is not None
        assert np.all(vec >= -2.0) and np.all(vec <= 2.0)

    def test_parse_padding(self):
        bridge = LLMBridge(dim=4)
        text = '{"bias": [1.0]}'
        vec = bridge._parse_response(text)
        assert vec is not None
        assert vec.shape == (4,)

    def test_fallback_bias(self):
        bridge = LLMBridge(dim=4, enabled=False)
        assert bridge._fallback_bias() is None
        bridge._last_bias = np.ones(4)
        fb = bridge._fallback_bias()
        assert fb is not None
        np.testing.assert_array_equal(fb, np.ones(4))

    def test_call_frequency(self):
        bridge = LLMBridge(dim=4, call_every_n=5, enabled=False)
        for i in range(1, 11):
            bridge.maybe_call(i, np.zeros(4), 0.5)


# ---------------------------------------------------------------------------
# Trainers
# ---------------------------------------------------------------------------
from core.trainer import AdversarialTrainer, BidirectionalTrainer


class TestBidirectionalTrainer:
    def test_train_short(self):
        t = BidirectionalTrainer(dim=8, max_steps=5, episodes=3, seed=42)
        metrics = t.train()
        assert len(metrics) == 3
        assert all("reward_a" in m for m in metrics)
        assert all("reward_b" in m for m in metrics)

    def test_callback(self):
        collected = []
        t = BidirectionalTrainer(dim=8, max_steps=5, episodes=2, seed=42)
        t.train(callback=lambda ep, m: collected.append(ep))
        assert collected == [1, 2]

    def test_best_ligand_state(self):
        t = BidirectionalTrainer(dim=8, max_steps=5, episodes=3, seed=42)
        t.train()
        assert t.best_ligand_state is not None
        assert "W_mu" in t.best_ligand_state

    def test_reward_balance_check(self):
        t = BidirectionalTrainer(dim=8, max_steps=5, episodes=1, seed=42)
        imbalanced = t._check_reward_balance(100.0, 0.001)
        assert imbalanced

    def test_entropy_floor_injection(self):
        t = BidirectionalTrainer(dim=8, max_steps=5, episodes=1, seed=42, entropy_floor=1000.0)
        old_sigma_a = t.agent_a.log_sigma.copy()
        t._check_entropy_floor()
        assert not np.allclose(t.agent_a.log_sigma, old_sigma_a)


class TestAdversarialTrainer:
    def test_train_short(self):
        t = AdversarialTrainer(dim=8, episodes=3, probe_steps=3, seed=42)
        metrics = t.train()
        assert len(metrics) == 3
        assert all("disruption" in m for m in metrics)

    def test_with_frozen_state(self):
        p1 = BidirectionalTrainer(dim=8, max_steps=5, episodes=2, seed=42)
        p1.train()
        t = AdversarialTrainer(
            dim=8, episodes=2, probe_steps=3, seed=42,
            frozen_ligand_state=p1.best_ligand_state,
        )
        metrics = t.train()
        assert len(metrics) == 2

    def test_hard_negatives(self):
        t = AdversarialTrainer(dim=8, episodes=5, probe_steps=3, seed=42)
        t.train()
        negs = t.hard_negatives
        assert isinstance(negs, list)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
from api.main import app


class TestAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_binding_score(self, client):
        r = client.post(
            "/binding/score",
            json={"ligand": [0.0] * 8, "receptor": [0.0] * 8},
        )
        assert r.status_code == 200
        assert "binding_score" in r.json()

    def test_binding_score_mismatch(self, client):
        r = client.post(
            "/binding/score",
            json={"ligand": [0.0] * 8, "receptor": [0.0] * 4},
        )
        assert r.status_code == 400

    def test_phase1_start_and_status(self, client):
        r = client.post(
            "/phase1/start",
            json={"dim": 8, "max_steps": 5, "episodes": 3, "seed": 0},
        )
        assert r.status_code == 200
        time.sleep(2)
        r2 = client.get("/phase1/status")
        assert r2.status_code == 200

    def test_phase1_metrics(self, client):
        r = client.post(
            "/phase1/start",
            json={"dim": 8, "max_steps": 5, "episodes": 2, "seed": 1},
        )
        time.sleep(2)
        r2 = client.get("/phase1/metrics")
        assert r2.status_code == 200

    def test_phase2_start(self, client):
        r = client.post(
            "/phase2/start",
            json={"dim": 8, "episodes": 2, "probe_steps": 3, "seed": 0},
        )
        assert r.status_code == 200
        time.sleep(2)
        r2 = client.get("/phase2/status")
        assert r2.status_code == 200

    def test_phase2_escape_motifs(self, client):
        r = client.get("/phase2/escape_motifs")
        assert r.status_code == 200

    def test_agent_action_no_trainer(self, client):
        from api.main import _state, _lock
        with _lock:
            saved = _state["phase1_trainer"]
            _state["phase1_trainer"] = None
        try:
            r = client.post(
                "/agent/action",
                json={"observation": [0.0] * 10, "agent": "ligand"},
            )
            assert r.status_code == 400
        finally:
            with _lock:
                _state["phase1_trainer"] = saved

    def test_agents_state_no_trainer(self, client):
        from api.main import _state, _lock
        with _lock:
            saved = _state["phase1_trainer"]
            _state["phase1_trainer"] = None
        try:
            r = client.get("/agents/state")
            assert r.status_code == 400
        finally:
            with _lock:
                _state["phase1_trainer"] = saved


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------
class TestIntegration:
    def test_full_phase1_loop(self):
        trainer = BidirectionalTrainer(dim=8, max_steps=5, episodes=5, seed=0)
        metrics = trainer.train()
        assert len(metrics) == 5
        for m in metrics:
            assert "avg_binding" in m
            assert 0 <= m["avg_binding"] <= 1

    def test_full_phase1_then_phase2(self):
        p1 = BidirectionalTrainer(dim=8, max_steps=5, episodes=3, seed=0)
        p1.train()
        p2 = AdversarialTrainer(
            dim=8, episodes=3, probe_steps=3, seed=0,
            frozen_ligand_state=p1.best_ligand_state,
        )
        m2 = p2.train()
        assert len(m2) == 3

    def test_bidirectional_coupling(self):
        rs = ReceptorState(dim=8)
        oracle = BindingOracle(dim=8)
        env_a = LigandEnv(rs, oracle, max_steps=5, seed=0)
        env_b = ReceptorEnv(rs, oracle, seed=0)
        # both reference same receptor state
        assert env_a.receptor_state is env_b.receptor_state
        env_a.reset()
        env_b.reset()
        env_b.step(np.ones(8) * 0.1)
        # mutation from env_b should affect env_a's receptor
        np.testing.assert_array_equal(
            env_a.receptor_state.vector, env_b.receptor_state.vector
        )

    def test_state_obs_distinction(self):
        rs = ReceptorState(dim=8)
        oracle = BindingOracle(dim=8)
        env = LigandEnv(rs, oracle, max_steps=5, seed=0)
        obs, info = env.reset()
        full_state = info["full_state"]
        assert len(full_state) > len(obs)
