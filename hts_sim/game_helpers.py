from typing import List, Optional

from .models import Engine, GameState
from .utils import format_card_list


def get_zone(state: GameState, pid: int, zone: str) -> List[int]:
    p = state.players[pid]
    z = zone.strip()
    if z == "player.hand":
        return p.hand
    if z == "player.party":
        return p.party
    if z == "player.captured_monsters":
        return p.captured_monsters
    if z == "discard_pile":
        return state.discard_pile
    if z == "draw_pile":
        return state.draw_pile
    if z == "monster_row":
        return state.monster_row
    raise KeyError(f"Unknown zone: {zone}")


def destroy_hero_card(state: GameState, engine: Engine, victim_pid: int, hero_id: int, log: List[str]) -> bool:
    p = state.players[victim_pid]
    if hero_id not in p.party:
        return False

    items = list(p.hero_items.get(hero_id, []))
    if items:
        for item_id in items:
            state.discard_pile.append(item_id)
        p.hero_items[hero_id] = []
        log.append(
            f"[P{victim_pid}] hero {hero_id} dies -> discarded items: {format_card_list(items, engine.card_meta)}"
        )
    p.hero_class_overrides.pop(hero_id, None)

    p.party.remove(hero_id)
    state.discard_pile.append(hero_id)
    log.append(
        f"[P{victim_pid}] hero destroyed/sacrificed -> {hero_id}:{engine.card_meta.get(hero_id,{}).get('name','?')}"
    )
    return True


def pick_opponent_pid(state: GameState, pid: int) -> Optional[int]:
    n = len(state.players)
    for off in range(1, n):
        op = (pid + off) % n
        if state.players[op].party:
            return op
    return None


def attacker_choose_hero_to_destroy(state: GameState, engine: Engine, victim_pid: int) -> Optional[int]:
    p = state.players[victim_pid]
    if not p.party:
        return None
    best = None
    best_score = -1
    for hid in p.party:
        score = len(p.hero_items.get(hid, []))
        if score > best_score:
            best_score = score
            best = hid
    return best


def victim_choose_hero_to_sacrifice(state: GameState, engine: Engine, victim_pid: int) -> Optional[int]:
    p = state.players[victim_pid]
    if not p.party:
        return None
    best = None
    best_score = 10**9
    for hid in p.party:
        score = len(p.hero_items.get(hid, []))
        if score < best_score:
            best_score = score
            best = hid
    return best
