#!/usr/bin/env python3
"""CLI entrypoint for adversarial RL training.

Usage:
    python run_training.py --phase1 --episodes 200
    python run_training.py --phase2 --episodes 100
    python run_training.py --phase1 --phase2  # both sequentially
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from core.trainer import AdversarialTrainer, BidirectionalTrainer


def phase1_callback(ep: int, m: dict) -> None:
    if ep % 10 == 0 or ep == 1:
        print(
            f"[Phase1] ep={ep:4d}  "
            f"bind={m['avg_binding']:.4f}  "
            f"R_a={m['reward_a']:+.3f}  "
            f"R_b={m['reward_b']:+.3f}  "
            f"Δ(wt)={m['receptor_displacement']:.3f}  "
            f"steps={m['steps']}"
        )


def phase2_callback(ep: int, m: dict) -> None:
    if ep % 10 == 0 or ep == 1:
        print(
            f"[Phase2] ep={ep:4d}  "
            f"disrupt={m['disruption']:.4f}  "
            f"Δ(wt)={m['receptor_displacement']:.3f}  "
            f"σ={m['sigma_mean']:.4f}  "
            f"div={m['diversity_bonus']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adversarial RL — Ligand-Receptor Co-Training"
    )
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1 (bidirectional)")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 (adversarial)")
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episodes2", type=int, default=100, help="Phase 2 episodes")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--binding-threshold", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None, help="Save metrics JSON")

    args = parser.parse_args()

    if not args.phase1 and not args.phase2:
        parser.print_help()
        sys.exit(1)

    all_metrics: dict[str, list] = {}

    if args.phase1:
        print("=" * 60)
        print("PHASE 1: Bidirectional Co-Training")
        print("=" * 60)
        trainer1 = BidirectionalTrainer(
            dim=args.dim,
            max_steps=args.max_steps,
            episodes=args.episodes,
            gamma=args.gamma,
            binding_threshold=args.binding_threshold,
            seed=args.seed,
        )
        metrics1 = trainer1.train(callback=phase1_callback)
        all_metrics["phase1"] = metrics1
        print(f"\nPhase 1 complete. Best binding: {trainer1._best_binding:.4f}")

    frozen_state = None
    if args.phase2:
        if args.phase1:
            frozen_state = trainer1.best_ligand_state  # type: ignore[possibly-undefined]

        print("\n" + "=" * 60)
        print("PHASE 2: Adversarial Escape Training")
        print("=" * 60)
        trainer2 = AdversarialTrainer(
            dim=args.dim,
            episodes=args.episodes2,
            seed=args.seed + 100,
            frozen_ligand_state=frozen_state,
        )
        metrics2 = trainer2.train(callback=phase2_callback)
        all_metrics["phase2"] = metrics2

        motifs = trainer2.escape_agent.escape_motifs
        print(f"\nPhase 2 complete. Escape motifs found: {len(motifs)}")
        if motifs:
            top = max(motifs, key=lambda m: m["disruption"])
            print(f"  Top disruption: {top['disruption']:.4f} at episode {top['episode']}")

    if args.output:

        def _convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(args.output, "w") as f:
            json.dump(all_metrics, f, indent=2, default=_convert)
        print(f"\nMetrics saved to {args.output}")


if __name__ == "__main__":
    main()
