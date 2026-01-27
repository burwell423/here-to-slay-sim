from typing import List, Tuple

from .models import Engine, GameState
from .utils import format_card_list


def build_decks(card_meta) -> Tuple[List[int], List[int], List[int]]:
    draw_deck: List[int] = []
    monster_deck: List[int] = []
    leader_deck: List[int] = []

    for cid, m in card_meta.items():
        ctype = str(m.get("type", "unknown")).strip().lower()
        copies = int(m.get("copies_in_deck", 0) or 0)
        if copies <= 0:
            continue

        if ctype in ("monster", "monsters"):
            monster_deck.extend([cid] * copies)
        elif ctype in ("party_leader", "party leader", "leader", "party-leader"):
            leader_deck.extend([cid] * copies)
        else:
            draw_deck.extend([cid] * copies)

    return draw_deck, monster_deck, leader_deck


def setup_game(state: GameState, engine: Engine, rng: "random.Random", log: List[str]):
    for p in state.players:
        if state.party_leader_deck:
            leader = state.party_leader_deck.pop()
            p.party_leader = leader
            log.append(f"[P{p.pid}] party leader = {leader} ({engine.card_meta.get(leader, {}).get('name','?')})")

    for p in state.players:
        for _ in range(3):
            if not state.draw_pile:
                break
            cid = state.draw_pile.pop()
            p.hand.append(cid)
            m = engine.card_meta.get(cid, {})
            log.append(f"[P{p.pid}] starting hand drew {cid} ({m.get('name','?')} / {m.get('type','?')})")

    for i in range(3):
        if not state.monster_deck:
            break
        mid = state.monster_deck.pop()
        state.monster_row.append(mid)
        log.append(f"[SETUP] monster_row[{i}] = {mid} ({engine.card_meta.get(mid, {}).get('name','?')})")


def log_turn_state(state: GameState, engine: Engine, pid: int, log: List[str]):
    p = state.players[pid]
    log.append("")
    log.append(f"--- TURN START: Player {pid} ---")
    log.append(f"Actions: {p.action_points}")
    log.append(
        f"Party Leader: {p.party_leader}:{engine.card_meta.get(p.party_leader, {}).get('name','?')}"
        if p.party_leader is not None
        else "Party Leader: —"
    )
    log.append(f"Hand ({len(p.hand)}): {format_card_list(p.hand, engine.card_meta)}")
    log.append(f"Party ({len(p.party)}): {format_card_list(p.party, engine.card_meta)}")
    if p.party:
        for hid in p.party:
            items = p.hero_items.get(hid, [])
            if items:
                log.append(
                    f"  Items on {hid}:{engine.card_meta.get(hid, {}).get('name','?')} -> "
                    f"{format_card_list(items, engine.card_meta)}"
                )
    log.append(
        f"Captured Monsters ({len(p.captured_monsters)}): {format_card_list(p.captured_monsters, engine.card_meta)}"
    )
    log.append(f"Monster Row ({len(state.monster_row)}): {format_card_list(state.monster_row, engine.card_meta)}")
    log.append("-" * 40)
