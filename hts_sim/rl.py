from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .actions import apply_action_candidate, build_action_candidates
from .effects import resolve_effect
from .game_helpers import collect_party_classes
from .loaders import build_engine
from .models import GameState, PlayerState, Policy
from .setup import build_decks, setup_game


@dataclass(frozen=True)
class RewardConfig:
    win: float = 10.0
    loss: float = -8.0
    monster_capture: float = 2.5
    party_class_completion: float = 6.0
    party_class_progress: float = 1.5
    wasted_action: float = -1.0


@dataclass
class Transition:
    state: Dict[str, float]
    action: Dict[str, float]
    reward: float
    next_state: Dict[str, float]
    terminal: bool


def _required_hero_classes(engine) -> set:
    return {
        str(meta.get("subtype", "")).strip().lower()
        for meta in engine.card_meta.values()
        if str(meta.get("type", "")).strip().lower() in ("hero", "party_leader", "party leader", "leader", "party-leader")
        and str(meta.get("subtype", "")).strip()
    }


def _summarize_state(engine, player: PlayerState, required_classes: set) -> Dict[str, float]:
    party_classes = collect_party_classes(engine, player)
    progress = len(party_classes) / max(len(required_classes), 1) if required_classes else 0.0
    return {
        "hand_size": float(len(player.hand)),
        "party_size": float(len(player.party)),
        "monsters_captured": float(len(player.captured_monsters)),
        "party_class_progress": float(progress),
        "action_points": float(player.action_points),
    }


def _check_win_conditions(state: GameState, engine) -> Optional[int]:
    required_classes = _required_hero_classes(engine)
    for player in state.players:
        if len(player.captured_monsters) >= 3:
            return player.pid
        party_classes = collect_party_classes(engine, player)
        if required_classes and required_classes.issubset(party_classes):
            return player.pid
    return None


def _compute_reward_delta(
    engine,
    player_before: PlayerState,
    player_after: PlayerState,
    required_classes: set,
    config: RewardConfig,
    action_taken: bool,
) -> float:
    reward = 0.0
    captured_before = len(player_before.captured_monsters)
    captured_after = len(player_after.captured_monsters)
    reward += (captured_after - captured_before) * config.monster_capture

    classes_before = collect_party_classes(engine, player_before)
    classes_after = collect_party_classes(engine, player_after)
    progress_before = len(classes_before) / max(len(required_classes), 1) if required_classes else 0.0
    progress_after = len(classes_after) / max(len(required_classes), 1) if required_classes else 0.0
    reward += max(progress_after - progress_before, 0.0) * config.party_class_progress

    if required_classes and required_classes.issubset(classes_after) and not required_classes.issubset(classes_before):
        reward += config.party_class_completion

    if not action_taken:
        reward += config.wasted_action
    return reward


def _action_features(policy: Policy, action, state: GameState, engine, pid: int) -> Dict[str, float]:
    if action.kind == "attack_monster":
        return policy.extract_attack_features(action, state, engine, pid)
    if action.kind == "activate_hero":
        return policy.extract_activate_features(action, state, engine, pid)
    if action.kind == "play_card":
        return policy.extract_play_features(action, state, engine, pid)
    if action.kind == "draw":
        return policy.extract_draw_features(action, state, engine, pid)
    return {}


def train_policy(
    episodes: int = 25,
    turns: int = 12,
    n_players: int = 4,
    seed: int = 1,
    epsilon: float = 0.15,
    alpha: float = 0.05,
    gamma: float = 0.9,
    reward_config: RewardConfig = RewardConfig(),
    weights_path: Optional[str] = None,
) -> Tuple[Policy, List[Transition]]:
    rng = random.Random(seed)
    engine = build_engine()
    policy = Policy(weights_path=weights_path)
    transitions: List[Transition] = []

    for episode in range(episodes):
        draw_deck, monster_deck, leader_deck = build_decks(engine.card_meta)
        rng.shuffle(draw_deck)
        rng.shuffle(monster_deck)
        rng.shuffle(leader_deck)

        players = [PlayerState(pid=i) for i in range(n_players)]
        state = GameState(
            players=players,
            draw_pile=draw_deck,
            monster_deck=monster_deck,
            party_leader_deck=leader_deck,
        )
        setup_game(state, engine, rng, [])
        required_classes = _required_hero_classes(engine)

        winner_pid: Optional[int] = None
        for t in range(turns):
            state.turn = t + 1
            for player in state.players:
                if player.roll_modifiers:
                    player.roll_modifiers = [
                        entry for entry in player.roll_modifiers if entry[2] is None or entry[2] >= state.turn
                    ]
            pid = t % len(state.players)
            state.active_pid = pid
            active = state.players[pid]
            active.activated_heroes_this_turn.clear()
            active.actions_per_turn = 3
            for mid in active.captured_monsters:
                for step in engine.monster_effects.get(mid, []):
                    if "passive" in step.triggers():
                        resolve_effect(step, state, engine, pid, {}, rng, policy, [])
            active.action_points = active.actions_per_turn

            safety = 30
            while active.action_points > 0 and safety > 0:
                candidates = build_action_candidates(state, engine, pid)
                if not candidates:
                    break

                if rng.random() < epsilon:
                    action = rng.choice(candidates)
                else:
                    scored = [(policy.score_action(c, state, engine, pid), c) for c in candidates]
                    scored.sort(key=lambda pair: (-pair[0], pair[1].kind))
                    action = scored[0][1]

                player_snapshot = PlayerState(
                    pid=active.pid,
                    hand=list(active.hand),
                    party=list(active.party),
                    captured_monsters=list(active.captured_monsters),
                    party_leader=active.party_leader,
                    hero_items={k: list(v) for k, v in active.hero_items.items()},
                    hero_class_overrides={k: list(v) for k, v in active.hero_class_overrides.items()},
                    actions_per_turn=active.actions_per_turn,
                    action_points=active.action_points,
                    roll_modifiers=list(active.roll_modifiers),
                    activated_heroes_this_turn=set(active.activated_heroes_this_turn),
                )

                features = _action_features(policy, action, state, engine, pid)
                current_q = sum(policy.feature_weights.get(name, 0.0) * value for name, value in features.items())
                action_taken = apply_action_candidate(action, state, engine, pid, rng, policy, [])

                reward = _compute_reward_delta(
                    engine,
                    player_snapshot,
                    active,
                    required_classes,
                    reward_config,
                    action_taken,
                )
                winner_pid = _check_win_conditions(state, engine)
                terminal = winner_pid is not None
                if terminal:
                    reward += reward_config.win if winner_pid == pid else reward_config.loss

                next_candidates = build_action_candidates(state, engine, pid) if not terminal else []
                next_q = max(
                    (policy.score_action(c, state, engine, pid) for c in next_candidates),
                    default=0.0,
                )
                td_target = reward + gamma * next_q
                td_error = td_target - current_q
                for name, value in features.items():
                    policy.feature_weights[name] = policy.feature_weights.get(name, 0.0) + alpha * td_error * value

                transitions.append(
                    Transition(
                        state=_summarize_state(engine, player_snapshot, required_classes),
                        action={"kind": action.kind, "score": current_q},
                        reward=reward,
                        next_state=_summarize_state(engine, active, required_classes),
                        terminal=terminal,
                    )
                )

                if terminal:
                    break
                safety -= 1
            if winner_pid is not None:
                break

    if weights_path:
        policy.save_feature_weights(weights_path)
    return policy, transitions


def evaluate_policies(
    seeds: List[int],
    turns: int = 12,
    n_players: int = 4,
    baseline_policy: Optional[Policy] = None,
    tuned_policy: Optional[Policy] = None,
) -> Dict[str, int]:
    engine = build_engine()
    baseline_policy = baseline_policy or Policy(feature_weights=Policy.default_feature_weights())
    tuned_policy = tuned_policy or Policy(weights_path=None)

    results = {"baseline_wins": 0, "tuned_wins": 0, "ties": 0}
    for seed in seeds:
        rng = random.Random(seed)
        draw_deck, monster_deck, leader_deck = build_decks(engine.card_meta)
        rng.shuffle(draw_deck)
        rng.shuffle(monster_deck)
        rng.shuffle(leader_deck)

        players = [PlayerState(pid=i) for i in range(n_players)]
        state = GameState(
            players=players,
            draw_pile=draw_deck,
            monster_deck=monster_deck,
            party_leader_deck=leader_deck,
        )
        setup_game(state, engine, rng, [])

        winner: Optional[int] = None
        for t in range(turns):
            state.turn = t + 1
            pid = t % len(state.players)
            state.active_pid = pid
            p = state.players[pid]
            p.activated_heroes_this_turn.clear()
            p.actions_per_turn = 3
            p.action_points = p.actions_per_turn

            policy = baseline_policy if pid % 2 == 0 else tuned_policy
            while p.action_points > 0:
                candidates = build_action_candidates(state, engine, pid)
                if not candidates:
                    break
                scored = [(policy.score_action(c, state, engine, pid), c) for c in candidates]
                scored.sort(key=lambda pair: (-pair[0], pair[1].kind))
                apply_action_candidate(scored[0][1], state, engine, pid, rng, policy, [])
                winner = _check_win_conditions(state, engine)
                if winner is not None:
                    break
            if winner is not None:
                break

        if winner is None:
            results["ties"] += 1
        elif winner % 2 == 0:
            results["baseline_wins"] += 1
        else:
            results["tuned_wins"] += 1

    return results
