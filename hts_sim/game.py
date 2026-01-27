import random
from typing import List, Optional

from .actions import choose_and_take_action
from .effects import resolve_effect
from .loaders import build_engine
from .models import GameState, PlayerState, Policy
from .setup import build_decks, log_turn_state, setup_game


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

    for t in range(turns):
        state.turn = t + 1
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
            safety -= 1

    return log
