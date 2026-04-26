## Bidirectional Adversarial Reinforcement Learning for Drug Discovery

### When the target stops being passive

Most reinforcement learning systems in drug design assume a quiet truth: the target is fixed.  
The receptor is a static lock, and the ligand is just trying different keys until one fits.

This project breaks that assumption completely.

Here, the receptor fights back.

---

## The core idea (and why it feels different instantly)

Instead of training a single agent in a stable environment, we train **two adaptive systems that continuously reshape each other**:

- A **Ligand Designer** that tries to maximize binding
- A **Receptor Mutator** that actively changes the receptor to _escape binding_

But here’s the key distinction:

> This is not “two agents in one environment.”  
> This is **one environment that is being rewritten live by the opponent.**

The receptor is not just an opponent — it is the environment itself.

That subtle shift changes everything.

---

## Why traditional RL fails here

Standard molecular RL works like this:

- Fix receptor structure
- Train ligand policy
- Optimize binding score
- Done

This breaks in real biology because:

- Receptors mutate under pressure
- Binding pockets evolve
- Static optimization overfits instantly
- Performance collapses under even small perturbations

In short: you train for a world that never stays the same.

---

## What changes in the bidirectional setup

Instead of a frozen target, we introduce **continuous adversarial evolution**.

### Phase 1: Co-evolution loop

The system becomes a closed loop:

- Ligand proposes a molecule
- Receptor mutates in response
- The ligand now faces a _new target instantly_
- Both agents learn from the same evolving state

The key mechanism is shared state:

> The receptor mutation is not an external event — it directly alters the ligand’s environment mid-training.

So the ligand is never optimizing against a stable distribution.  
It is optimizing against a moving target that adapts to it.

---

## Why this is fundamentally different from “two separate agents”

A common misconception is:  
“Why not just train a ligand agent and a receptor agent separately, then pit them together?”

That fails for three reasons:

### 1. No co-adaptation

Separate training produces agents that specialize against _old versions of each other_.  
When combined, both collapse because neither has learned real-time adaptation.

---

### 2. No shared state pressure

In this system:

- A receptor mutation instantly changes the ligand’s environment
- The ligand must react _within the same episode_

There is no reset boundary where adaptation can be delayed.

This creates continuous pressure — not staged evaluation.

---

### 3. No curriculum emergence

The receptor mutator naturally generates difficulty progression:

- Early: random perturbations
- Mid: structural drift
- Late: targeted escape mutations

This becomes an **automatic curriculum generator** — no handcrafted difficulty schedule needed.

---

## The hidden advantage: adversarial curriculum learning

As training progresses:

- Ligand improves binding → receptor responds
- Receptor improves escape → ligand is forced into harder regions
- Both agents expand their strategy space

The result is not convergence.

It is escalation.

---

## Phase 2: The escape amplifier

After co-training, we freeze the best ligand and introduce a third pressure:

- The **Escape Agent**
- Its goal: discover receptor mutations that break the strongest ligand

This does something important:

> It exposes failure modes that never appear during normal training.

Those failure cases are then fed back as hard negatives, forcing robustness rather than narrow optimization.

---

## What actually gets learned (and why it matters)

The surprising outcome:

- Peak performance happens early
- Later performance may drop
- But robustness increases significantly

This is not instability — it is adaptation pressure increasing.

The ligand stops being:

> “best for one receptor”

and becomes:

> “stable across a landscape of receptor variants”

That is the real goal in drug discovery under resistance pressure.

---

## Key results

- Strong binding improvement over baseline
- Significant selectivity gains
- Discovery of escape-sensitive receptor motifs
- Robust ligand behavior across mutations
- Emergent adversarial curriculum without manual design

---

## Why this approach matters

This system shifts the paradigm in three important ways:

### 1. From optimization → evolution

Not searching for a single best molecule, but evolving against changing constraints.

### 2. From static targets → adaptive systems

The target is no longer an input. It is part of the learning loop.

### 3. From robustness by design → robustness by pressure

Instead of engineering robustness explicitly, it emerges from adversarial stress.

---

## Final insight

Most RL systems ask:

> “What is the best action for this state?”

This system asks something harder:

> “What is the best action when the state is actively trying to defeat you?”

And that single shift is why the learned molecules are not just optimized — they are _resilient under attack_.
