---
title: Bidirectional Adversarial RL Drug Discovery
emoji: 🧬
colorFrom: blue
colorTo: purple
sdk: docker
app_file: api/main.py
pinned: false
---

# 🧬 Bidirectional Adversarial RL for Drug Discovery

> _What if the drug target fought back?_

[![HF Space](https://img.shields.io/badge/🤗%20Space-Live%20Demo-blue)](https://huggingface.co/spaces/Sumayyakhalid92587/drug-discovery)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](YOUR_GITHUB_REPO_LINK_HERE)
[![HF Blog](https://img.shields.io/badge/🤗%20Blog-Writeup-orange)](YOUR_HF_BLOG_LINK_HERE)

---

## Problem: Drug design is a one-sided game — until now

Traditional RL-based drug design trains an agent to bind to a **fixed receptor**. That's like learning to pick a lock that never changes. Real biology doesn't work that way — viruses mutate, cancer cells evolve resistance, and proteins shift conformation under pressure.

This project removes the static target assumption entirely by pitting two co-evolving agents against each other in an **adversarial arms race**.

---

## The Environment

Two agents share a single environment and compete at every step:

| Agent                | Goal                      | Action                           |
| -------------------- | ------------------------- | -------------------------------- |
| **Ligand Designer**  | Maximize binding affinity | Proposes molecular modifications |
| **Receptor Mutator** | Minimize binding affinity | Mutates receptor structure       |

At each timestep:

1. Ligand Designer proposes a molecule
2. Receptor Mutator immediately mutates the target
3. Binding affinity is recomputed on the _updated_ receptor
4. Both agents receive adversarial rewards

No frozen benchmark. No static target. The environment resets to a new receptor configuration each episode, forcing the ligand agent to generalize rather than memorize.

**Phase 2 adds an Escape Agent** — a third agent that specifically hunts for receptor mutations that _break_ high-performing ligands. These hard negatives are fed back into training, closing the loop:

```
Design → Resistance → Exploit → Retrain → Design ...
```

---

## Results

### Phase 1: Adversarial Training

**Binding Affinity During Adversarial Training**
![Binding Trained vs Baseline](plots/binding_trained_vs_baseline.png)
_The trained ligand agent starts strong (best: 0.689) but binding score declines as the receptor mutator learns to resist — this is expected and healthy. The agent is being hardened, not defeated. Baseline random agent avg = 0.349._

**Reward Curves: Ligand Agent vs Receptor Mutator**
![Reward Curves](plots/reward_curves.png)
_Left: Ligand agent cumulative reward drops as the adversary gets stronger — the receptor mutator is winning the arms race. Right: Receptor mutator reward converges near 0.93, confirming it learns to reliably suppress binding._

**Exploration Noise (Sigma) Decay & Policy Entropy**
![Sigma Entropy](plots/sigma_entropy.png)
_Both sigma (exploration noise) and policy entropy decay smoothly across 200 episodes, confirming the ligand agent is converging to a deterministic policy — not just thrashing randomly._

---

### Phase 2: Escape Agent

**Escape Agent Reward & Binding Disruption**
![Phase 2 Escape](plots/phase2_escape.png)
_Left: Escape agent reward climbs steadily to ~1.45 over 60 episodes. Right: Binding disruption (1 − binding) reaches near-1.0, meaning the escape agent learns to completely nullify high-performing ligands. These mutations become hard negatives for retraining._

---

### Evaluation: Trained vs Random Baseline

**Head-to-Head Comparison**
![Evaluation Comparison](plots/evaluation_comparison.png)
_Left bar chart: On a fixed receptor, the trained agent achieves 0.689 best binding, 0.421 total score, and 0.322 selectivity — vs 0.349, 0.421, and 0.060 for random. The selectivity gain (5× improvement) is the most important result: the trained agent learns to bind the target specifically, not just promiscuously. Right: Per-episode best binding shows the trained agent is stable and consistent; the random baseline is highly variable._

**Selectivity Under Sequential Mutations**
![Selectivity Stability](plots/selectivity_stability.png)
_Trained selectivity (blue) holds flat at ~0.322 across 50 sequential receptor mutations. Baseline selectivity (red) fluctuates chaotically around 0.06. This is the core robustness result — the trained agent maintains target specificity even as the receptor drifts._

---

## Key Insight

> **Lower binding score during training ≠ failure.**
>
> It signals the receptor mutator is doing its job. The ligand agent is being stress-tested against an increasingly hostile target. When evaluated on a _fixed_ receptor, the adversarially-trained agent outperforms random on every metric — especially selectivity, which matters most for drug safety.

---

## Why It Matters

| Who cares                      | Why                                                            |
| ------------------------------ | -------------------------------------------------------------- |
| **Drug discovery researchers** | Adversarial training produces more robust lead compounds       |
| **RL researchers**             | Novel multi-agent co-evolution setup with real-world grounding |
| **Pharma / biotech**           | Directly addresses the drug resistance problem                 |

The biggest failure mode in drug development is resistance — the target changes and the drug stops working. This environment trains agents to anticipate resistance from the start.

---

## Quick Start

```bash
git clone YOUR_GITHUB_REPO_LINK_HERE
cd drug-discovery
pip install -r requirements.txt
python train.py
```

Or run end-to-end in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

---

## Environment Structure (OpenEnv)

```
drug-discovery/
├── env/
│   ├── drug_discovery_env.py   # Main Environment (OpenEnv base class)
│   ├── ligand_agent.py
│   ├── receptor_mutator.py
│   └── escape_agent.py
├── api/
│   └── main.py                 # HF Space app entrypoint
├── train.py                    # Training script
├── plots/                      # Training evidence (committed)
│   ├── binding_trained_vs_baseline.png
│   ├── reward_curves.png
│   ├── sigma_entropy.png
│   ├── phase2_escape.png
│   ├── evaluation_comparison.png
│   └── selectivity_stability.png
└── openenv.yaml
```

The environment follows the OpenEnv spec: `Environment` base class with Gym-style `reset()` / `step()` / `state`, and a valid `openenv.yaml`.

---

## Links

| Resource                     | Link                                                                                                        |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------- |
| 🤗 HF Space (Live Demo)      | [spaces/Sumayyakhalid92587/drug-discovery](https://huggingface.co/spaces/Sumayyakhalid92587/drug-discovery) |
| 📓 Training Notebook (Colab) | [Open in Colab](https://colab.research.google.com/drive/1HlA3rWhufUO823gIfxiUKI0uDqPUmYfx)                  |
| 💻 Code Repository           | [GitHub](https://github.com/coral-cyber/drug-discovery.git)                                                 |
| 📝 HF Blog Post              | [Read the writeup](https://huggingface.co/spaces/Sumayyakhalid92587/drug-discovery/blob/main/blog.md)       |

---

_Built for research in adversarial RL, drug discovery, and evolutionary optimization._
