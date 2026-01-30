from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from .models import EffectStep, MonsterRule


EFFECT_KIND_VALUES: Dict[str, float] = {
    "draw_card": 6.0,
    "search_and_draw": 7.0,
    "play_immediately": 5.0,
    "play_card": 4.0,
    "play_drawn_immediately": 5.0,
    "steal_card": 9.0,
    "steal_hero": 12.0,
    "discard_card": 4.0,
    "destroy_hero": 10.0,
    "sacrifice_hero": 9.0,
    "return_to_hand": 5.0,
    "modify_roll": 3.0,
    "modify_hero_roll": 3.0,
    "modify_action_total": 4.0,
    "modify_hero_class": 6.0,
    "deny": 6.0,
    "deny_challenge": 6.0,
    "protection_from_destroy": 4.0,
    "protection_from_challenge": 4.0,
    "protection_from_steal": 4.0,
    "trade_hands": 5.0,
    "use_hero": 5.0,
    "move_card": 4.0,
    "reveal_card": 2.0,
    "do_nothing": 0.0,
}

CARD_BASE_VALUES: Dict[str, int] = {
    "hero": 60,
    "item": 45,
    "magic": 35,
    "challenge": 25,
    "modifier": 15,
    "monster": 20,
    "party_leader": 80,
    "party leader": 80,
    "party-leader": 80,
}


def _effect_value(step: EffectStep) -> float:
    kind = str(step.effect_kind or "").strip().lower()
    base = EFFECT_KIND_VALUES.get(kind, 1.5)

    amount = step.amount if step.amount is not None else 0
    if kind in ("modify_roll", "modify_hero_roll"):
        base += min(6, abs(amount)) * 2.5 if amount else 1.5
    elif amount:
        base += min(8, abs(amount)) * 0.8

    triggers = step.triggers()
    if any(t in ("on_activation", "on_play") for t in triggers):
        base += 1.5
    if "passive" in triggers:
        base += 1.0

    if step.requires_roll:
        base *= 0.85
    if step.condition:
        base *= 0.9
    return base


def _sum_effect_values(steps: Iterable[EffectStep]) -> float:
    return sum(_effect_value(step) for step in steps)


def compute_card_tuning_value(
    card_id: int,
    meta: Dict[str, Any],
    effects_by_card: Dict[int, Iterable[EffectStep]],
    monster_attack_rules: Dict[int, MonsterRule],
    monster_effects: Dict[int, Iterable[EffectStep]],
    overrides: Optional[Dict[int, float]] = None,
) -> int:
    if overrides and card_id in overrides:
        return int(round(overrides[card_id]))

    ctype = str(meta.get("type", "unknown")).strip().lower()
    action_cost = int(meta.get("action_cost", 1) or 1)
    base = CARD_BASE_VALUES.get(ctype, 20) + action_cost * 2

    effect_bonus = _sum_effect_values(effects_by_card.get(card_id, []))

    monster_rule = monster_attack_rules.get(card_id)
    if ctype == "monster" or monster_rule is not None:
        requirements = monster_rule.attack_requirements if monster_rule else None
        difficulty = sum(requirements.values()) if requirements else 0
        monster_bonus = difficulty * 3.0
        monster_bonus += _sum_effect_values(monster_effects.get(card_id, []))
        effect_bonus += monster_bonus
        base = CARD_BASE_VALUES.get("monster", 20) + action_cost * 2

    total = base + effect_bonus
    return max(1, int(round(total)))
