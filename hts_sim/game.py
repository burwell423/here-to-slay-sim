import random
from typing import List, Optional

from .actions import choose_and_take_action
from .effects import resolve_effect
from .loaders import build_engine
from .models import GameState, PlayerState, Policy
from .game_helpers import collect_party_classes
from .setup import build_decks, log_turn_state, setup_game


def _required_hero_classes(engine) -> set:
    return {
        str(meta.get("subtype", "")).strip().lower()
        for meta in engine.card_meta.values()
        if str(meta.get("type", "")).strip().lower() in ("hero", "party_leader", "party leader", "leader", "party-leader")
        and str(meta.get("subtype", "")).strip()
    }


def _check_win_conditions(state: GameState, engine, log: List[str]) -> Optional[int]:
    required_classes = _required_hero_classes(engine)
    for player in state.players:
        if len(player.captured_monsters) >= 3:
            log.append(f"[WIN] P{player.pid} captured {len(player.captured_monsters)} monsters")
            return player.pid
        party_classes = collect_party_classes(engine, player)
        if required_classes and required_classes.issubset(party_classes):
            log.append(
                f"[WIN] P{player.pid} assembled party classes: {', '.join(sorted(party_classes))}"
            )
            return player.pid
    return None


def run_game(seed: int = 1, turns: int = 10, n_players: int = 4, policy: Optional[Policy] = None) -> List[str]:
    rng = random.Random(seed)
    engine = build_engine()
    policy = policy or Policy()

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

    log: List[str] = []
    setup_game(state, engine, rng, log)

    winner_pid: Optional[int] = None
    for t in range(turns):
        state.turn = t + 1
        for player in state.players:
            if player.roll_modifiers:
                player.roll_modifiers = [
                    entry
                    for entry in player.roll_modifiers
                    if entry[2] is None or entry[2] >= state.turn
                ]
        pid = t % len(state.players)
        state.active_pid = pid
        p = state.players[pid]
        p.actions_per_turn = 3
        for mid in p.captured_monsters:
            for step in engine.monster_effects.get(mid, []):
                if "passive" in step.triggers():
                    resolve_effect(step, state, engine, pid, {}, rng, policy, log)
        p.action_points = p.actions_per_turn

        log_turn_state(state, engine, pid, log)

        safety = 30
        while p.action_points > 0 and safety > 0:
            acted = choose_and_take_action(state, engine, pid, rng, policy, log)
            if not acted:
                break
            winner_pid = _check_win_conditions(state, engine, log)
            if winner_pid is not None:
                break
            safety -= 1
        if winner_pid is not None:
            break

    return log
