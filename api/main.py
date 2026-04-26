from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.trainer import AdversarialTrainer, BidirectionalTrainer


app = FastAPI(title="Bidirectional Adversarial RL API", version="1.0.0")

_state: dict[str, Any] = {"phase1_trainer": None, "phase1_summary": None, "phase2_trainer": None, "phase2_summary": None}


class Phase1Request(BaseModel):
    episodes: int = Field(default=4, ge=1, le=100)
    dimension: int = Field(default=8, ge=2, le=64)
    max_steps: int = Field(default=6, ge=1, le=50)
    llm_interval: int = Field(default=2, ge=1, le=50)
    seed: int = Field(default=0, ge=0)


class Phase2Request(BaseModel):
    episodes: int = Field(default=4, ge=1, le=100)
    llm_interval: int | None = Field(default=None, ge=1)


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "bidirectional-adversarial-rl", "phases": ["phase1", "phase2"], "status": "ready"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True}


@app.post("/train/phase1")
def train_phase1(payload: Phase1Request) -> dict[str, Any]:
    trainer = BidirectionalTrainer(dimension=payload.dimension, max_steps=payload.max_steps, llm_interval=payload.llm_interval, seed=payload.seed)
    summary = trainer.train_phase1(episodes=payload.episodes)
    _state["phase1_trainer"] = trainer
    _state["phase1_summary"] = summary.metrics
    return {"phase": summary.phase, "episodes": summary.episodes, "metrics": summary.metrics}


@app.post("/train/phase2")
def train_phase2(payload: Phase2Request) -> dict[str, Any]:
    trainer = _state.get("phase1_trainer")
    if trainer is None:
        trainer = BidirectionalTrainer()
        trainer.train_phase1(episodes=3)
        _state["phase1_trainer"] = trainer
        _state["phase1_summary"] = trainer.metrics_history[-1] if trainer.metrics_history else {}
    phase2_trainer = AdversarialTrainer(trainer, llm_interval=payload.llm_interval)
    summary = phase2_trainer.train_phase2(episodes=payload.episodes)
    _state["phase2_trainer"] = phase2_trainer
    _state["phase2_summary"] = summary.metrics
    return {"phase": summary.phase, "episodes": summary.episodes, "metrics": summary.metrics}


@app.get("/state")
def state() -> dict[str, Any]:
    trainer = _state.get("phase1_trainer")
    receptor = trainer.receptor.as_dict() if trainer is not None else None
    return {"phase1": _state.get("phase1_summary"), "phase2": _state.get("phase2_summary"), "receptor": receptor}


@app.get("/flow/{phase}")
def flow(phase: str) -> dict[str, Any]:
    if phase == "phase1" and _state.get("phase1_trainer") is not None:
        trainer = _state["phase1_trainer"]
        return {"phase": phase, "events": trainer.flow_history}
    if phase == "phase2" and _state.get("phase2_trainer") is not None:
        trainer = _state["phase2_trainer"]
        return {"phase": phase, "events": trainer.flow_history}
    return {"phase": phase, "events": []}
