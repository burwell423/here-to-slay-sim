from typing import List, Optional, Tuple

from .conditions import goal_satisfied, roll_2d6
from .models import Engine, GameState
from .utils import find_modifier_cards


def _collect_hero_roll_modifiers(
    state: GameState,
    engine: Engine,
    hero_id: Optional[int],
) -> Tuple[int, List[Tuple[int, int]]]:
    if hero_id is None:
        return 0, []

    for player in state.players:
        items = player.hero_items.get(hero_id, [])
        if not items:
            continue

        total = 0
        details: List[Tuple[int, int]] = []
        for item_id in items:
            for step in engine.effects_by_card.get(item_id, []):
                if (step.effect_kind or "").strip().lower() != "modify_hero_roll":
                    continue
                if "passive" not in step.triggers():
                    continue
                if step.amount is None:
                    continue
                total += step.amount
                details.append((item_id, step.amount))
        if details:
            return total, details

    return 0, []


def resolve_roll_event(
    state: GameState,
    engine: Engine,
    roller_pid: int,
    roll_reason: str,
    rng: "random.Random",
    log: List[str],
    goal: Optional[Tuple[str, int]] = None,
    mode: str = "threshold",
    hero_id: Optional[int] = None,
) -> int:
    """
    mode:
      - 'threshold': use goal ('>=',X) or ('<=',X) to decide whether to play mods
      - 'maximize': roller wants high, others want low (challenge-style)
    """
    base = roll_2d6(rng)
    total = base
    log.append(f"[ROLL:{roll_reason}] P{roller_pid} base 2d6 = {base}")

    hero_mod, hero_mod_details = _collect_hero_roll_modifiers(state, engine, hero_id)
    if hero_mod:
        total += hero_mod
        parts = ", ".join(
            f"{item_id}:{engine.card_meta.get(item_id, {}).get('name', '?')} {delta:+d}"
            for item_id, delta in hero_mod_details
        )
        log.append(
            f"[ROLL:{roll_reason}] hero {hero_id} modifiers {hero_mod:+d} from {parts} -> total={total}"
        )

    used_by_player = set()
    ordered = [pid for pid in range(len(state.players)) if pid != roller_pid] + [roller_pid]

    def improvement_score_before_after(before: int, after: int, pid: int) -> int:
        if mode == "maximize":
            return (after - before) if pid == roller_pid else (before - after)

        if not goal:
            return 0

        op, target = goal
        before_ok = goal_satisfied(before, op, target)
        after_ok = goal_satisfied(after, op, target)

        if pid == roller_pid:
            if (not before_ok) and after_ok:
                return 1000 + abs(after - before)
            if before_ok and after_ok:
                return 10 + (before - after if op == "<=" else after - before)
            return (after - before if op == ">=" else before - after)
        else:
            if before_ok and (not after_ok):
                return 1000 + abs(after - before)
            if (not before_ok) and (not after_ok):
                return 10 + (after - before if op == "<=" else before - after)
            return (before - after if op == ">=" else after - before)

    for pid in ordered:
        if pid in used_by_player:
            continue

        mods = find_modifier_cards(state.players[pid], engine.card_meta)
        if not mods:
            continue

        best = None  # (score, mod_card_id, chosen_delta)
        for mid in mods:
            deltas = engine.modifier_options_by_card_id.get(mid, [])
            if not deltas:
                continue
            for d in deltas:
                score = improvement_score_before_after(total, total + d, pid)
                if best is None or score > best[0]:
                    best = (score, mid, d)

        if not best:
            continue

        score, chosen_card, chosen_delta = best
        if score <= 0:
            continue

        state.players[pid].hand.remove(chosen_card)
        state.discard_pile.append(chosen_card)
        total += chosen_delta
        used_by_player.add(pid)

        log.append(
            f"[ROLL:{roll_reason}] P{pid} plays modifier {chosen_card} "
            f"({engine.card_meta.get(chosen_card,{}).get('name','?')}) choose {chosen_delta:+d} -> total={total}"
        )

    log.append(f"[ROLL:{roll_reason}] FINAL total = {total}")
    return total
