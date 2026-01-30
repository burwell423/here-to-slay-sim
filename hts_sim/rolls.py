from typing import List, Optional, Tuple

from .conditions import goal_satisfied, roll_2d6_detail
from .models import Engine, GameState, Policy
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
    policy: Policy,
    goal: Optional[Tuple[str, int]] = None,
    mode: str = "threshold",
    hero_id: Optional[int] = None,
) -> int:
    """
    mode:
      - 'threshold': use goal ('>=',X) or ('<=',X) to decide whether to play mods
      - 'maximize': roller wants high, others want low (challenge-style)
    """
    ctx = {
        "roll_player": roller_pid,
        "roll_reason": roll_reason,
        "roll_modifiers": [],
    }
    if roll_reason.startswith("challenge:"):
        for mid in state.players[roller_pid].captured_monsters:
            for step in engine.monster_effects.get(mid, []):
                if "on_challenge_roll" in step.triggers():
                    from .effects import resolve_effect

                    resolve_effect(step, state, engine, roller_pid, ctx, rng, policy, log)
    for mid in state.players[roller_pid].captured_monsters:
        for step in engine.monster_effects.get(mid, []):
            if "on_roll" in step.triggers():
                from .effects import resolve_effect

                resolve_effect(step, state, engine, roller_pid, ctx, rng, policy, log)

    die_one, die_two, base = roll_2d6_detail(rng)
    total = base
    log.append(
        f"[ROLL:{roll_reason}] P{roller_pid} base 2d6 = {base} ({die_one}+{die_two})"
    )

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

    ctx_roll_mods = ctx.get("roll_modifiers") or []
    if ctx_roll_mods:
        total += sum(entry[1] for entry in ctx_roll_mods)
        parts = ", ".join(
            f"{entry[0]}:{engine.card_meta.get(entry[0], {}).get('name', '?')} {entry[1]:+d}"
            for entry in ctx_roll_mods
        )
        log.append(
            f"[ROLL:{roll_reason}] P{roller_pid} on_roll modifiers "
            f"{sum(entry[1] for entry in ctx_roll_mods):+d} from {parts} -> total={total}"
        )

    roller = state.players[roller_pid]
    if roller.roll_modifiers:
        roller.roll_modifiers = [
            entry
            for entry in roller.roll_modifiers
            if entry[2] is None or entry[2] >= state.turn
        ]
    if roller.roll_modifiers:
        total += sum(entry[1] for entry in roller.roll_modifiers)
        parts = ", ".join(
            f"{entry[0]}:{engine.card_meta.get(entry[0], {}).get('name', '?')} {entry[1]:+d}"
            for entry in roller.roll_modifiers
        )
        log.append(
            f"[ROLL:{roll_reason}] P{roller_pid} passive modifiers "
            f"{sum(entry[1] for entry in roller.roll_modifiers):+d} from {parts} -> total={total}"
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

        if before_ok == after_ok:
            return 0

        if pid == roller_pid:
            if (not before_ok) and after_ok:
                return 1000 + abs(after - before)
            return (after - before if op == ">=" else before - after)
        else:
            if before_ok and (not after_ok):
                return 1000 + abs(after - before)
            return (before - after if op == ">=" else after - before)

    for pid in ordered:
        if pid in used_by_player:
            continue

        player = state.players[pid]
        mods = find_modifier_cards(player, engine.card_meta)
        if not mods:
            continue

        sources: List[Tuple[str, int, Optional[int], List[int]]] = []
        for mid in mods:
            deltas = engine.modifier_options_by_card_id.get(mid, [])
            if deltas:
                sources.append(("card", mid, None, deltas))

        states: dict[int, List[Tuple[str, int, Optional[int], int]]] = {0: []}
        for source_type, source_id, source_card_id, deltas in sources:
            updated = dict(states)
            for current_delta, choices in states.items():
                for d in deltas:
                    next_delta = current_delta + d
                    next_choices = choices + [(source_type, source_id, source_card_id, d)]
                    existing = updated.get(next_delta)
                    if existing is None or len(next_choices) < len(existing):
                        updated[next_delta] = next_choices
            states = updated

        best_score = 0.0
        best_choices: Optional[List[Tuple[str, int, Optional[int], int]]] = None
        for delta_total, choices in states.items():
            if not choices:
                continue
            score = improvement_score_before_after(total, total + delta_total, pid)
            score -= policy.modifier_choice_cost(choices, engine)
            if score <= 0:
                continue
            if best_choices is None or (score, -len(choices)) > (best_score, -len(best_choices)):
                best_score = score
                best_choices = choices

        if not best_choices:
            continue

        for source_type, source_id, source_card_id, chosen_delta in best_choices:
            if source_type == "card":
                player.hand.remove(source_id)
                state.discard_pile.append(source_id)
                played_name = engine.card_meta.get(source_id, {}).get("name", "?")
                log.append(
                    f"[ROLL:{roll_reason}] P{pid} plays modifier {source_id} "
                    f"({played_name}) choose {chosen_delta:+d} -> total={total + chosen_delta}"
                )
                ctx = {
                    "roll_player": roller_pid,
                    "modifier_player": pid,
                    "modifier_card": engine.card_meta.get(source_id, {"id": source_id}),
                }
                for owner in state.players:
                    for mid in owner.captured_monsters:
                        for step in engine.monster_effects.get(mid, []):
                            if "on_modifier_played" in step.triggers():
                                from .effects import resolve_effect

                                resolve_effect(step, state, engine, owner.pid, ctx, rng, policy, log)
            total += chosen_delta

        used_by_player.add(pid)

    log.append(f"[ROLL:{roll_reason}] FINAL total = {total}")
    return total
