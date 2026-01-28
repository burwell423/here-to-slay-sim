import re
from typing import Dict, List, Optional, Set, Tuple, Union

from .models import Engine, GameState, PlayerState
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


def get_hero_class(engine: Engine, player: "PlayerState", hero_id: int) -> Optional[str]:
    overrides = player.hero_class_overrides.get(hero_id)
    if overrides:
        return overrides[-1][1].strip().lower() or None
    subtype = str(engine.card_meta.get(hero_id, {}).get("subtype", "")).strip().lower()
    return subtype or None


def collect_party_classes(engine: Engine, player: "PlayerState") -> Set[str]:
    classes: Set[str] = set()
    for hero_id in player.party:
        hero_class = get_hero_class(engine, player, hero_id)
        if hero_class:
            classes.add(hero_class)
    if player.party_leader is not None:
        leader_class = str(engine.card_meta.get(player.party_leader, {}).get("subtype", "")).strip().lower()
        if leader_class:
            classes.add(leader_class)
    return classes


def parse_attack_requirements(attack_requirements: Optional[Union[str, Dict[str, int]]]) -> Dict[str, int]:
    if not attack_requirements:
        return {}
    if isinstance(attack_requirements, dict):
        return {key.strip().lower(): int(value) for key, value in attack_requirements.items()}

    normalized = attack_requirements.strip()
    if not normalized:
        return {}

    requirements: Dict[str, int] = {}
    key_value_pairs = re.findall(r"([a-zA-Z][a-zA-Z\\s-]*)\\s*:\\s*(\\d+)", normalized)
    if key_value_pairs:
        for raw_key, raw_count in key_value_pairs:
            key = raw_key.strip().lower()
            if not key:
                continue
            requirements[key] = requirements.get(key, 0) + int(raw_count)
        return requirements

    pairs = re.findall(r"(\\d+)\\s*\\(([^)]+)\\)", normalized)
    for count, cls in pairs:
        key = cls.strip().lower()
        if not key:
            continue
        requirements[key] = requirements.get(key, 0) + int(count)
    return requirements


def collect_party_class_counts(engine: Engine, player: PlayerState) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for hero_id in player.party:
        hero_class = get_hero_class(engine, player, hero_id)
        if hero_class:
            counts[hero_class] = counts.get(hero_class, 0) + 1
    if player.party_leader is not None:
        leader_class = str(engine.card_meta.get(player.party_leader, {}).get("subtype", "")).strip().lower()
        if leader_class:
            counts[leader_class] = counts.get(leader_class, 0) + 1
    return counts


def can_player_attack_monster(player: PlayerState, engine: Engine, monster_id: int) -> bool:
    rule = engine.monster_attack_rules.get(monster_id)
    if not rule or not rule.attack_requirements:
        return True
    requirements = parse_attack_requirements(rule.attack_requirements)
    if not requirements:
        return True
    total_heroes = len(player.party) + (1 if player.party_leader is not None else 0)
    class_counts = collect_party_class_counts(engine, player)
    for req_class, count in requirements.items():
        if req_class == "any":
            if total_heroes < count:
                return False
        elif class_counts.get(req_class, 0) < count:
            return False
    return True


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
