---
title: Bidirectional Adversarial RL Drug Discovery
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: fastapi
app_file: api/main.py
pinned: false
---

# Bidirectional Adversarial Reinforcement Learning for Drug Discovery

Drug discovery usually assumes a fixed biological target. This project removes that assumption by introducing a **co-evolving adversary system**.

## Core Idea

Two agents are trained in a shared environment:

- **Ligand Designer** → generates molecules to maximize binding
- **Receptor Mutator** → modifies receptor structure to reduce binding

This creates a **dynamic evolutionary arms race**, not a static optimization problem.

## Why it matters

Traditional RL drug design:

- Fixed receptor → overfitting risk

This system:

- Evolving receptor → robustness pressure
- Forces generalization across mutations

## Key Mechanism

At each step:

1. Ligand proposes molecule
2. Receptor mutates immediately
3. Binding is recomputed on updated state

No frozen benchmark. No static target.

## Phase 2: Escape Attack

An additional **Escape Agent** discovers receptor mutations that break high-performing ligands and feeds them back into training.

This creates a full loop:

- Design → Resistance → Exploit → Retrain

## Outcome

Instead of a single “best molecule”, the system produces:

- Mutation-robust ligands
- Hard negative receptor cases
- Adaptive drug design policies

## Insight

Lower binding score in later training ≠ failure  
It indicates **robustness under adversarial pressure**

---

Built for research in:

- Adversarial RL
- Drug discovery
- Evolutionary optimization
