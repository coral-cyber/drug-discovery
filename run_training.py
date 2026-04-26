from __future__ import annotations

import argparse
import json

from core.trainer import AdversarialTrainer, BidirectionalTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bidirectional adversarial RL training")
    parser.add_argument("--phase1", action="store_true", help="Run phase 1 co-training")
    parser.add_argument("--phase2", action="store_true", help="Run phase 2 adversarial training")
    parser.add_argument("--episodes", type=int, default=4, help="Episodes for the selected phase")
    parser.add_argument("--dimension", type=int, default=8, help="Vector dimension")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum ligand steps per episode")
    parser.add_argument("--llm-interval", type=int, default=2, help="LLM bias interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    run_phase1 = args.phase1 or not args.phase2
    phase1_trainer = None
    if run_phase1:
        phase1_trainer = BidirectionalTrainer(dimension=args.dimension, max_steps=args.max_steps, llm_interval=args.llm_interval, seed=args.seed)
        phase1_summary = phase1_trainer.train_phase1(episodes=args.episodes)
        print(json.dumps({"phase1": phase1_summary.metrics}, indent=2))
    if args.phase2:
        if phase1_trainer is None:
            phase1_trainer = BidirectionalTrainer(dimension=args.dimension, max_steps=args.max_steps, llm_interval=args.llm_interval, seed=args.seed)
            phase1_trainer.train_phase1(episodes=max(2, args.episodes))
        phase2_trainer = AdversarialTrainer(phase1_trainer, llm_interval=args.llm_interval)
        phase2_summary = phase2_trainer.train_phase2(episodes=args.episodes)
        print(json.dumps({"phase2": phase2_summary.metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
