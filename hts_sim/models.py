from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
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
    duration: Optional[str]

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
    hero_class_overrides: Dict[int, List[Tuple[int, str]]] = field(default_factory=lambda: defaultdict(list))
    actions_per_turn: int = 3
    action_points: int = 3
    roll_modifiers: List[Tuple[int, int, Optional[int]]] = field(default_factory=list)
    activated_heroes_this_turn: Set[int] = field(default_factory=set)


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
    attack_requirements: Optional[Dict[str, int]] = None


@dataclass
class ChallengePolicy:
    challenge_probability: float = 0.35


@dataclass
class Policy:
    challenge: ChallengePolicy = field(default_factory=ChallengePolicy)
    feature_weights: Dict[str, float] = field(default_factory=dict)
    weights_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.feature_weights:
            self.feature_weights = self.default_feature_weights()
        if self.weights_path:
            self.load_feature_weights(self.weights_path)

    @staticmethod
    def default_feature_weights() -> Dict[str, float]:
        return {
            "bias": 0.0,
            "action_cost": -1.5,
            "action_point_efficiency": 1.0,
            "monsters_captured": 1.0,
            "party_class_progress": 2.5,
            "hand_size": 0.2,
            "party_size": 0.4,
            "is_attack": 4.0,
            "monster_value": 0.08,
            "monster_capture_urgency": 2.0,
            "is_activate": 1.2,
            "activated_hero_value": 0.1,
            "remaining_activations": 0.6,
            "is_play": 2.0,
            "played_card_value": 0.15,
            "played_card_is_hero": 1.5,
            "played_card_is_item": 0.6,
            "played_card_is_magic": 0.3,
            "played_card_is_challenge": 0.1,
            "played_card_is_modifier": -2.0,
            "adds_party_class": 2.5,
            "is_draw": 0.4,
            "draw_pile_size": 0.01,
        }

    def load_feature_weights(self, path: str) -> None:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            self.feature_weights.update({k: float(v) for k, v in payload.items()})

    def save_feature_weights(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.feature_weights, f, indent=2, sort_keys=True)

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

    def choose_steal_hero(
        self,
        opp_party: List[int],
        engine: "Engine",
        opp_hero_items: Dict[int, List[int]],
    ) -> Optional[int]:
        if not opp_party:
            return None
        ranked = sorted(
            opp_party,
            key=lambda hid: (
                -(self.score_card_value(hid, engine) + len(opp_hero_items.get(hid, [])) * 5),
                hid,
            ),
        )
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
        available = [hid for hid in party if not hero_items.get(hid)]
        if not available:
            return None
        ranked = sorted(
            available,
            key=lambda hid: (-self.score_card_value(hid, engine), hid),
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

    def choose_trade_partner(self, state: "GameState", pid: int) -> Optional[int]:
        candidates = [p for p in state.players if p.pid != pid]
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda p: (-len(p.hand), p.pid))
        return ranked[0].pid

    def choose_reveal_opponent(self, candidates: List["PlayerState"]) -> Optional[int]:
        if not candidates:
            return None
        ranked = sorted(candidates, key=lambda p: (-len(p.hand), p.pid))
        return ranked[0].pid

    def choose_reveal_card(self, opp_hand: List[int], engine: "Engine") -> Optional[int]:
        return self.choose_steal_card(opp_hand, engine)

    def should_challenge(self, rng: random.Random) -> bool:
        return rng.random() < self.challenge.challenge_probability

    def _hero_class(self, engine: "Engine", player: "PlayerState", hero_id: int) -> Optional[str]:
        overrides = player.hero_class_overrides.get(hero_id)
        if overrides:
            return overrides[-1][1].strip().lower() or None
        subtype = str(engine.card_meta.get(hero_id, {}).get("subtype", "")).strip().lower()
        return subtype or None

    def _party_classes(self, engine: "Engine", player: "PlayerState") -> Set[str]:
        classes: Set[str] = set()
        for hero_id in player.party:
            hero_class = self._hero_class(engine, player, hero_id)
            if hero_class:
                classes.add(hero_class)
        if player.party_leader is not None:
            leader_class = str(engine.card_meta.get(player.party_leader, {}).get("subtype", "")).strip().lower()
            if leader_class:
                classes.add(leader_class)
        return classes

    def _required_hero_classes(self, engine: "Engine") -> Set[str]:
        return {
            str(meta.get("subtype", "")).strip().lower()
            for meta in engine.card_meta.values()
            if str(meta.get("type", "")).strip().lower()
            in ("hero", "party_leader", "party leader", "leader", "party-leader")
            and str(meta.get("subtype", "")).strip()
        }

    def _party_class_progress(self, engine: "Engine", player: "PlayerState") -> float:
        required = self._required_hero_classes(engine)
        if not required:
            return 0.0
        collected = self._party_classes(engine, player)
        return len(collected) / max(len(required), 1)

    def _action_point_efficiency(self, player: "PlayerState", cost: int) -> float:
        return (player.action_points - cost) / max(player.actions_per_turn, 1)

    def _base_action_features(
        self,
        state: "GameState",
        engine: "Engine",
        pid: int,
        cost: int,
    ) -> Dict[str, float]:
        player = state.players[pid]
        return {
            "bias": 1.0,
            "action_cost": float(cost),
            "action_point_efficiency": self._action_point_efficiency(player, cost),
            "monsters_captured": float(len(player.captured_monsters)),
            "party_class_progress": self._party_class_progress(engine, player),
            "hand_size": float(len(player.hand)),
            "party_size": float(len(player.party)),
        }

    def extract_attack_features(
        self,
        action: "ActionCandidate",
        state: "GameState",
        engine: "Engine",
        pid: int,
    ) -> Dict[str, float]:
        features = self._base_action_features(state, engine, pid, action.cost)
        monster_id = action.monster_id or 0
        remaining = max(0, 3 - len(state.players[pid].captured_monsters))
        features.update(
            {
                "is_attack": 1.0,
                "monster_value": float(self.score_card_value(monster_id, engine)),
                "monster_capture_urgency": remaining / 3.0,
            }
        )
        return features

    def extract_activate_features(
        self,
        action: "ActionCandidate",
        state: "GameState",
        engine: "Engine",
        pid: int,
    ) -> Dict[str, float]:
        features = self._base_action_features(state, engine, pid, action.cost)
        hero_id = action.hero_id or 0
        remaining = sum(
            1
            for hid in state.players[pid].party
            if hid not in state.players[pid].activated_heroes_this_turn
            and any("on_activation" in s.triggers() for s in engine.effects_by_card.get(hid, []))
        )
        features.update(
            {
                "is_activate": 1.0,
                "activated_hero_value": float(self.score_card_value(hero_id, engine)),
                "remaining_activations": float(remaining),
            }
        )
        return features

    def extract_play_features(
        self,
        action: "ActionCandidate",
        state: "GameState",
        engine: "Engine",
        pid: int,
    ) -> Dict[str, float]:
        features = self._base_action_features(state, engine, pid, action.cost)
        card_id = action.card_id or 0
        meta = engine.card_meta.get(card_id, {})
        ctype = str(meta.get("type", "")).strip().lower()
        subtype = str(meta.get("subtype", "")).strip().lower()
        adds_class = 0.0
        if ctype == "hero" and subtype:
            existing = self._party_classes(engine, state.players[pid])
            if subtype not in existing:
                adds_class = 1.0
        features.update(
            {
                "is_play": 1.0,
                "played_card_value": float(self.score_card_value(card_id, engine)),
                "played_card_is_hero": 1.0 if ctype == "hero" else 0.0,
                "played_card_is_item": 1.0 if ctype == "item" else 0.0,
                "played_card_is_magic": 1.0 if ctype == "magic" else 0.0,
                "played_card_is_challenge": 1.0 if ctype == "challenge" else 0.0,
                "played_card_is_modifier": 1.0 if ctype == "modifier" else 0.0,
                "adds_party_class": adds_class,
            }
        )
        return features

    def extract_draw_features(
        self,
        action: "ActionCandidate",
        state: "GameState",
        engine: "Engine",
        pid: int,
    ) -> Dict[str, float]:
        features = self._base_action_features(state, engine, pid, action.cost)
        features.update({"is_draw": 1.0, "draw_pile_size": float(len(state.draw_pile))})
        return features

    def score_action(
        self,
        action: "ActionCandidate",
        state: "GameState",
        engine: "Engine",
        pid: int,
    ) -> float:
        if action.kind == "attack_monster":
            features = self.extract_attack_features(action, state, engine, pid)
        elif action.kind == "activate_hero":
            features = self.extract_activate_features(action, state, engine, pid)
        elif action.kind == "play_card":
            features = self.extract_play_features(action, state, engine, pid)
        elif action.kind == "draw":
            features = self.extract_draw_features(action, state, engine, pid)
        else:
            features = self._base_action_features(state, engine, pid, action.cost)
        return sum(self.feature_weights.get(name, 0.0) * value for name, value in features.items())


@dataclass
class Engine:
    effects_by_card: Dict[int, List[EffectStep]]
    card_meta: Dict[int, Dict[str, Any]]
    monster_attack_rules: Dict[int, MonsterRule]
    monster_effects: Dict[int, List[EffectStep]]
    modifier_options_by_card_id: Dict[int, List[int]]


@dataclass(frozen=True)
class ActionCandidate:
    kind: str
    cost: int
    card_id: Optional[int] = None
    hero_id: Optional[int] = None
    monster_id: Optional[int] = None
