#!/usr/bin/env python3
import random
import sys
from collections import defaultdict
from typing import Dict, List

from hts_sim.actions import play_card_from_hand
from hts_sim.loaders import build_engine
from hts_sim.models import GameState, PlayerState, Policy


def _build_state(hero_ids: List[int]) -> GameState:
    players = [PlayerState(pid=0), PlayerState(pid=1)]
    if hero_ids:
        players[0].party.append(hero_ids[0])
        if len(hero_ids) > 1:
            players[1].party.append(hero_ids[1])
        else:
            players[1].party.append(hero_ids[0])
    for player in players:
        player.action_points = 3
    return GameState(players=players, draw_pile=[])


def _collect_unimplemented_warnings(log: List[str]) -> List[str]:
    warnings: List[str] = []
    for line in log:
        if " WARN " not in line:
            continue
        _, warning = line.split(" WARN ", 1)
        if warning.startswith("UNIMPLEMENTED_EFFECT_KIND"):
            warnings.append(warning)
    return warnings


def main() -> int:
    rng = random.Random(1)
    engine = build_engine()
    policy = Policy()

    hero_ids = sorted(
        cid for cid, meta in engine.card_meta.items() if str(meta.get("type", "")).lower() == "hero"
    )

    warnings_by_card: Dict[int, List[str]] = defaultdict(list)

    for card_id in sorted(engine.card_meta.keys()):
        state = _build_state(hero_ids)
        state.active_pid = 0
        player = state.players[0]
        player.hand.append(card_id)

        log: List[str] = []
        play_card_from_hand(
            state=state,
            engine=engine,
            pid=0,
            card_id=card_id,
            rng=rng,
            policy=policy,
            log=log,
            cost_override=0,
            allow_challenge=False,
        )
        warnings = _collect_unimplemented_warnings(log)
        if warnings:
            warnings_by_card[card_id].extend(warnings)

    print("=== Play All Cards Audit ===\n")
    if not warnings_by_card:
        print("No UNIMPLEMENTED_EFFECT_KIND warnings found.")
        return 0

    total_warnings = sum(len(v) for v in warnings_by_card.values())
    print(
        f"Cards with unimplemented effect warnings: {len(warnings_by_card)} "
        f"(total warnings: {total_warnings})"
    )
    for card_id, warnings in sorted(warnings_by_card.items(), key=lambda item: item[0]):
        meta = engine.card_meta.get(card_id, {})
        name = meta.get("name", "?")
        ctype = meta.get("type", "unknown")
        print(f"- {card_id} {name} ({ctype}):")
        for warning in sorted(set(warnings)):
            print(f"  - {warning}")

    return 2


if __name__ == "__main__":
    sys.exit(main())
