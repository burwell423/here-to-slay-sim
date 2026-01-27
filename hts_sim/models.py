from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .utils import find_challenge_card_in_hand


@dataclass
class EffectStep:
    name: str
    card_id: int
    step: int
    trigger: str
    effect_kind: str
    source_zone: Optional[str]
    dest_zone: Optional[str]
    filter_expr: Optional[str]
    amount: Optional[int]
    amount_expr: Optional[str]
    requires_roll: bool
    roll_condition: Optional[str]
    condition: Optional[str]
    notes: Optional[str]

    def triggers(self) -> List[str]:
        return [t.strip() for t in (self.trigger or "").split(";") if t.strip()]


@dataclass
class PlayerState:
    pid: int
    hand: List[int] = field(default_factory=list)
    party: List[int] = field(default_factory=list)
    captured_monsters: List[int] = field(default_factory=list)
    party_leader: Optional[int] = None
    hero_items: Dict[int, List[int]] = field(default_factory=lambda: defaultdict(list))
    actions_per_turn: int = 3
    action_points: int = 3


@dataclass
class GameState:
    players: List[PlayerState]
    draw_pile: List[int]
    monster_deck: List[int] = field(default_factory=list)
    party_leader_deck: List[int] = field(default_factory=list)
    monster_row: List[int] = field(default_factory=list)
    discard_pile: List[int] = field(default_factory=list)
    turn: int = 0
    active_pid: int = 0


@dataclass
class MonsterRule:
    monster_id: int
    success_condition: Optional[str]
    fail_condition: Optional[str]
    success_action: Optional[str]
    fail_action: Optional[str]


@dataclass
class ChallengePolicy:
    challenge_probability: float = 0.35


@dataclass
class Policy:
    challenge: ChallengePolicy = field(default_factory=ChallengePolicy)

    def score_card_value(self, card_id: int, engine: "Engine") -> int:
        meta = engine.card_meta.get(card_id, {})
        ctype = str(meta.get("type", "unknown")).lower()
        cost = int(meta.get("action_cost", 1) or 1)
        base = {
            "hero": 60,
            "item": 45,
            "magic": 35,
            "challenge": 25,
            "modifier": 15,
            "monster": 5,
            "party_leader": 80,
        }.get(ctype, 20)
        return base + cost

    def choose_discard_card(self, hand: List[int], engine: "Engine") -> Optional[int]:
        if not hand:
            return None
        ranked = sorted(hand, key=lambda cid: (self.score_card_value(cid, engine), cid))
        return ranked[0]

    def choose_steal_card(self, opp_hand: List[int], engine: "Engine") -> Optional[int]:
        if not opp_hand:
            return None
        ranked = sorted(opp_hand, key=lambda cid: (-self.score_card_value(cid, engine), cid))
        return ranked[0]

    def choose_move_card(self, source: List[int], dest_zone: str, engine: "Engine") -> Optional[int]:
        if not source:
            return None
        if dest_zone in ("discard_pile",):
            ranked = sorted(source, key=lambda cid: (self.score_card_value(cid, engine), cid))
        else:
            ranked = sorted(source, key=lambda cid: (-self.score_card_value(cid, engine), cid))
        return ranked[0]

    def choose_card_to_play(self, hand: List[int], engine: "Engine") -> Optional[int]:
        candidates = [cid for cid in hand if str(engine.card_meta.get(cid, {}).get("type", "")).lower() != "modifier"]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda cid: (-self.score_card_value(cid, engine), cid))
        return ranked[0]

    def choose_item_attach_target(
        self,
        party: List[int],
        engine: "Engine",
        hero_items: Dict[int, List[int]],
    ) -> Optional[int]:
        if not party:
            return None
        ranked = sorted(
            party,
            key=lambda hid: (-len(hero_items.get(hid, [])), hid),
        )
        return ranked[0]

    def choose_item_to_destroy(self, items: List[int], engine: "Engine") -> Optional[int]:
        if not items:
            return None
        ranked = sorted(items, key=lambda cid: (-self.score_card_value(cid, engine), cid))
        return ranked[0]

    def choose_monster_to_attack(self, monster_row: List[int], engine: "Engine") -> Optional[int]:
        if not monster_row:
            return None
        ranked = sorted(monster_row, key=lambda mid: (-self.score_card_value(mid, engine), mid))
        return ranked[0]

    def choose_challenger(self, state: "GameState", engine: "Engine", pid_playing: int) -> Optional[Tuple[int, int]]:
        candidates: List[Tuple[int, int]] = []
        n = len(state.players)
        for offset in range(1, n):
            opid = (pid_playing + offset) % n
            ccid = find_challenge_card_in_hand(state.players[opid], engine.card_meta)
            if ccid is not None:
                candidates.append((opid, ccid))
        if not candidates:
            return None
        ranked = sorted(
            candidates,
            key=lambda pair: (-len(state.players[pair[0]].hand), pair[0]),
        )
        return ranked[0]

    def should_challenge(self, rng: random.Random) -> bool:
        return rng.random() < self.challenge.challenge_probability


@dataclass
class Engine:
    effects_by_card: Dict[int, List[EffectStep]]
    card_meta: Dict[int, Dict[str, Any]]
    monster_attack_rules: Dict[int, MonsterRule]
    monster_effects: Dict[int, List[EffectStep]]
    modifier_options_by_card_id: Dict[int, List[int]]
