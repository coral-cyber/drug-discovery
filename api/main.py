"""FastAPI REST endpoints for the adversarial RL system."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.trainer import AdversarialTrainer, BidirectionalTrainer

app = FastAPI(
    title="Adversarial RL — Ligand-Receptor Co-Training",
    version="1.0.0",
    description="Bidirectional adversarial RL system for ligand-receptor binding simulation.",
)

_state: dict[str, Any] = {
    "phase1_trainer": None,
    "phase2_trainer": None,
    "phase1_running": False,
    "phase2_running": False,
    "phase1_metrics": [],
    "phase2_metrics": [],
}
_lock = threading.Lock()


class Phase1Config(BaseModel):
    dim: int = Field(16, ge=2, le=128)
    max_steps: int = Field(50, ge=5, le=500)
    episodes: int = Field(200, ge=1, le=10000)
    gamma: float = Field(0.99, ge=0.0, le=1.0)
    binding_threshold: float = Field(0.85, ge=0.0, le=1.0)
    seed: int = 0
    llm_call_every: int = Field(10, ge=1)
    llm_weight: float = Field(0.25, ge=0.0, le=1.0)


class Phase2Config(BaseModel):
    dim: int = Field(16, ge=2, le=128)
    episodes: int = Field(100, ge=1, le=10000)
    probe_steps: int = Field(10, ge=1, le=100)
    mutation_cap: float = Field(3.0, ge=0.1, le=10.0)
    diversity_coeff: float = Field(0.1, ge=0.0, le=1.0)
    seed: int = 100


class BindingQuery(BaseModel):
    ligand: list[float]
    receptor: list[float]


class AgentActionQuery(BaseModel):
    observation: list[float]
    agent: str = "ligand"
    deterministic: bool = False


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/phase1/start")
def start_phase1(config: Phase1Config) -> dict[str, str]:
    with _lock:
        if _state["phase1_running"]:
            raise HTTPException(400, "Phase 1 already running")
        trainer = BidirectionalTrainer(
            dim=config.dim,
            max_steps=config.max_steps,
            episodes=config.episodes,
            gamma=config.gamma,
            binding_threshold=config.binding_threshold,
            seed=config.seed,
            llm_call_every=config.llm_call_every,
            llm_weight=config.llm_weight,
        )
        _state["phase1_trainer"] = trainer
        _state["phase1_running"] = True
        _state["phase1_metrics"] = []

    def _run() -> None:
        try:

            def _cb(ep: int, m: dict) -> None:
                with _lock:
                    _state["phase1_metrics"].append(m)

            trainer.train(callback=_cb)
        finally:
            with _lock:
                _state["phase1_running"] = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "episodes": str(config.episodes)}


@app.get("/phase1/status")
def phase1_status() -> dict[str, Any]:
    with _lock:
        running = _state["phase1_running"]
        n = len(_state["phase1_metrics"])
        last = _state["phase1_metrics"][-1] if n > 0 else None
    return {"running": running, "episodes_completed": n, "last_metrics": last}


@app.get("/phase1/metrics")
def phase1_metrics() -> list[dict[str, Any]]:
    with _lock:
        return list(_state["phase1_metrics"])


@app.post("/phase2/start")
def start_phase2(config: Phase2Config) -> dict[str, str]:
    with _lock:
        if _state["phase2_running"]:
            raise HTTPException(400, "Phase 2 already running")

        frozen = None
        if _state["phase1_trainer"] is not None:
            frozen = _state["phase1_trainer"].best_ligand_state

        trainer = AdversarialTrainer(
            dim=config.dim,
            episodes=config.episodes,
            probe_steps=config.probe_steps,
            mutation_cap=config.mutation_cap,
            diversity_coeff=config.diversity_coeff,
            seed=config.seed,
            frozen_ligand_state=frozen,
        )
        _state["phase2_trainer"] = trainer
        _state["phase2_running"] = True
        _state["phase2_metrics"] = []

    def _run() -> None:
        try:

            def _cb(ep: int, m: dict) -> None:
                with _lock:
                    _state["phase2_metrics"].append(m)

            trainer.train(callback=_cb)
        finally:
            with _lock:
                _state["phase2_running"] = False

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return {"status": "started", "episodes": str(config.episodes)}


@app.get("/phase2/status")
def phase2_status() -> dict[str, Any]:
    with _lock:
        running = _state["phase2_running"]
        n = len(_state["phase2_metrics"])
        last = _state["phase2_metrics"][-1] if n > 0 else None
    return {"running": running, "episodes_completed": n, "last_metrics": last}


@app.get("/phase2/metrics")
def phase2_metrics() -> list[dict[str, Any]]:
    with _lock:
        return list(_state["phase2_metrics"])


@app.get("/phase2/escape_motifs")
def escape_motifs() -> list[dict[str, Any]]:
    with _lock:
        trainer = _state.get("phase2_trainer")
        if trainer is None:
            return []
        return trainer.escape_agent.escape_motifs


@app.post("/binding/score")
def binding_score(query: BindingQuery) -> dict[str, float]:
    from core.receptor import BindingOracle

    dim = len(query.ligand)
    if len(query.receptor) != dim:
        raise HTTPException(400, "ligand and receptor must have same dimension")
    oracle = BindingOracle(dim=dim)
    score = oracle.score(np.array(query.ligand), np.array(query.receptor))
    return {"binding_score": score}


@app.post("/agent/action")
def agent_action(query: AgentActionQuery) -> dict[str, Any]:
    with _lock:
        trainer = _state.get("phase1_trainer")
    if trainer is None:
        raise HTTPException(400, "Phase 1 trainer not initialized")
    obs = np.array(query.observation)
    if query.agent == "ligand":
        action = trainer.agent_a.act(obs, deterministic=query.deterministic)
    elif query.agent == "receptor":
        action = trainer.agent_b.act(obs, deterministic=query.deterministic)
    else:
        raise HTTPException(400, f"Unknown agent: {query.agent}")
    return {"action": action.tolist()}


@app.get("/agents/state")
def agents_state() -> dict[str, Any]:
    with _lock:
        trainer = _state.get("phase1_trainer")
    if trainer is None:
        raise HTTPException(400, "Phase 1 trainer not initialized")
    return {
        "agent_a": {
            "sigma_mean": float(np.mean(trainer.agent_a.sigma)),
            "train_steps": trainer.agent_a._train_steps,
        },
        "agent_b": {
            "sigma_mean": float(np.mean(trainer.agent_b.sigma)),
            "train_episodes": trainer.agent_b._train_episodes,
        },
    }
