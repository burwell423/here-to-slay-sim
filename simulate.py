#!/usr/bin/env python3
"""
Here to Slay - MVP simulator

Key features implemented:
- Separate draw deck / monster deck / party leader deck
- Setup: each player gets random party leader + 3 cards, reveal 3 monsters
- Turn structure: 3 actions; Draw (1), Play (1), Activate hero (1), Attack monster (2)
- Challenge cards: can challenge plays of hero/item/magic (not modifiers), cancelled on win
- Modifier cards: may be played by any player during any roll event; multi-option deltas supported
- Monsters:
  - on_attacked row defines success_condition / fail_condition + (optional) requirements
  - success_action / fail_action rows resolve via EffectSteps
  - captured monster passives via triggers like on_draw / on_challenge
"""

import argparse

from hts_sim.game import run_game
from hts_sim.models import Policy
from hts_sim.rl import evaluate_policies, load_transitions, save_transitions, train_policy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Here to Slay simulator")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run a single simulated game")
    run_parser.add_argument("--seed", type=int, default=7)
    run_parser.add_argument("--turns", type=int, default=100)
    run_parser.add_argument("--players", type=int, default=4)
    run_parser.add_argument("--weights", type=str, default=None)

    train_parser = subparsers.add_parser("train", help="Run RL training")
    train_parser.add_argument("--episodes", type=int, default=25)
    train_parser.add_argument("--turns", type=int, default=100)
    train_parser.add_argument("--players", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=1)
    train_parser.add_argument("--epsilon", type=float, default=0.15)
    train_parser.add_argument("--alpha", type=float, default=0.05)
    train_parser.add_argument("--gamma", type=float, default=0.9)
    train_parser.add_argument("--output", type=str, default="policy_weights.json")
    train_parser.add_argument("--transitions-in", type=str, default=None)
    train_parser.add_argument("--transitions-out", type=str, default=None)
    train_parser.add_argument("--replay-epochs", type=int, default=1)

    eval_parser = subparsers.add_parser("evaluate", help="Compare baseline vs tuned policy")
    eval_parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    eval_parser.add_argument("--turns", type=int, default=100)
    eval_parser.add_argument("--players", type=int, default=4)
    eval_parser.add_argument("--weights", type=str, default="policy_weights.json")

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "train":
        existing_transitions = load_transitions(args.transitions_in) if args.transitions_in else []
        _, transitions = train_policy(
            episodes=args.episodes,
            turns=args.turns,
            n_players=args.players,
            seed=args.seed,
            epsilon=args.epsilon,
            alpha=args.alpha,
            gamma=args.gamma,
            weights_path=args.output,
            replay_data=existing_transitions,
            replay_epochs=args.replay_epochs,
        )
        print(f"Saved weights to {args.output}. Collected {len(transitions)} transitions.")
        if args.transitions_out:
            combined = existing_transitions + transitions
            save_transitions(args.transitions_out, combined)
            print(f"Saved {len(combined)} transitions to {args.transitions_out}.")
        return

    if args.command == "evaluate":
        tuned = Policy(weights_path=args.weights)
        results = evaluate_policies(
            seeds=args.seeds,
            turns=args.turns,
            n_players=args.players,
            tuned_policy=tuned,
        )
        print("Evaluation results:", results)
        return

    policy = Policy(weights_path=getattr(args, "weights", None)) if args.command == "run" else None
    for line in run_game(
        seed=getattr(args, "seed", 7),
        turns=getattr(args, "turns", 8),
        n_players=getattr(args, "players", 4),
        policy=policy,
    ):
        print(line)


if __name__ == "__main__":
    main()
