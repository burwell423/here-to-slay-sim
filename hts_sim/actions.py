from typing import Any, Dict, List, Optional

from .challenges import maybe_challenge_play
from .conditions import goal_satisfied, parse_simple_condition
from .effects import resolve_effect
from .game_helpers import can_player_attack_monster
from .models import Engine, GameState, Policy
from .rolls import resolve_roll_event


def play_card_from_hand(
    state: GameState,
    engine: Engine,
    pid: int,
    card_id: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
    cost_override: Optional[int] = None,
    allow_challenge: bool = True,
):
    p = state.players[pid]
    if card_id not in p.hand:
        return
    cost = 1 if cost_override is None else cost_override
    if p.action_points < cost:
        return

    meta = engine.card_meta.get(card_id, {})
    ctype = str(meta.get("type", "unknown")).strip().lower()

    p.hand.remove(card_id)
    log.append(f"[P{pid}] PLAY {card_id} ({meta.get('name','?')} / {ctype}) cost={cost}")

    if allow_challenge:
        cancelled = maybe_challenge_play(state, engine, pid, card_id, rng, policy, log)
        if cancelled:
            state.discard_pile.append(card_id)
            log.append(f"[P{pid}] play of {card_id} cancelled -> discard")
            p.action_points -= cost
            return

    attached_hero: Optional[int] = None

    if ctype == "hero":
        p.party.append(card_id)
        log.append(f"[P{pid}] -> entered party: {card_id}:{meta.get('name','?')}")

    elif ctype == "item":
        subtype = str(meta.get("subtype", "")).strip().lower()
        is_cursed = subtype == "cursed"
        if is_cursed:
            candidates: List[tuple[int, int]] = []
            for opp in state.players:
                if opp.pid == pid:
                    continue
                for hero_id in opp.party:
                    if opp.hero_items.get(hero_id):
                        continue
                    candidates.append((opp.pid, hero_id))
            if not candidates:
                state.discard_pile.append(card_id)
                log.append(f"[P{pid}] WARN played cursed item with no valid opponent hero; discarded {card_id}")
            else:
                target_pid, target_hero = sorted(
                    candidates,
                    key=lambda pair: (-policy.score_card_value(pair[1], engine), pair[0], pair[1]),
                )[0]
                target_player = state.players[target_pid]
                target_player.hero_items[target_hero].append(card_id)
                attached_hero = target_hero
                log.append(
                    f"[P{pid}] -> attached cursed item {card_id}:{meta.get('name','?')} "
                    f"to P{target_pid} hero {target_hero}:{engine.card_meta.get(target_hero, {}).get('name','?')}"
                )
        else:
            if not p.party:
                state.discard_pile.append(card_id)
                log.append(f"[P{pid}] WARN played item with no heroes; discarded {card_id}")
            else:
                target_hero = policy.choose_item_attach_target(p.party, engine, p.hero_items)
                if target_hero is None:
                    state.discard_pile.append(card_id)
                    log.append(f"[P{pid}] WARN played item with no valid hero; discarded {card_id}")
                else:
                    p.hero_items[target_hero].append(card_id)
                    attached_hero = target_hero
                    log.append(
                        f"[P{pid}] -> attached item {card_id}:{meta.get('name','?')} "
                        f"to hero {target_hero}:{engine.card_meta.get(target_hero, {}).get('name','?')}"
                    )

    else:
        state.discard_pile.append(card_id)

    ctx: Dict[str, Any] = {
        "played_card_id": card_id,
        "played_card_type": ctype,
        "attached_to_hero": attached_hero if ctype == "item" else None,
    }
    if ctype == "item" and attached_hero is not None:
        for step in engine.effects_by_card.get(card_id, []):
            if "passive" in step.triggers() and step.effect_kind == "modify_hero_class":
                resolve_effect(step, state, engine, pid, ctx, rng, policy, log)
    for step in engine.effects_by_card.get(card_id, []):
        trig = step.triggers()
        if "on_play" in trig or "auto" in trig or "on_activation" in trig:
            resolve_effect(step, state, engine, pid, ctx, rng, policy, log)

    p.action_points -= cost
    for w in ctx.get("_warnings", []):
        log.append(f"[P{pid}] WARN {w}")


def action_draw(
    state: GameState,
    engine: Engine,
    pid: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
) -> bool:
    p = state.players[pid]
    if p.action_points < 1:
        return False
    if not state.draw_pile:
        return False

    cid = state.draw_pile.pop()
    p.hand.append(cid)

    drawn = engine.card_meta.get(cid, {"id": cid, "type": "unknown"})
    ctx = {"drawn_card": drawn}

    log.append(f"[P{pid}] ACTION draw (cost 1) -> {cid} ({drawn.get('name','?')} / {drawn.get('type','?')})")
    p.action_points -= 1

    for mid in p.captured_monsters:
        for step in engine.monster_effects.get(mid, []):
            if "on_draw" in step.triggers():
                resolve_effect(step, state, engine, pid, ctx, rng, policy, log)

    for w in ctx.get("_warnings", []):
        log.append(f"[P{pid}] WARN {w}")

    return True


def action_activate_hero(
    state: GameState,
    engine: Engine,
    pid: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
) -> bool:
    p = state.players[pid]
    if p.action_points < 1:
        return False
    if not p.party:
        return False

    for hero_id in p.party:
        steps = engine.effects_by_card.get(hero_id, [])
        if any("on_activation" in s.triggers() for s in steps):
            log.append(
                f"[P{pid}] ACTION activate hero (cost 1) -> {hero_id} "
                f"({engine.card_meta.get(hero_id,{}).get('name','?')})"
            )
            p.action_points -= 1

            ctx = {"activated_hero_id": hero_id}
            for step in steps:
                if "on_activation" in step.triggers() or "auto" in step.triggers():
                    resolve_effect(step, state, engine, pid, ctx, rng, policy, log)

            for w in ctx.get("_warnings", []):
                log.append(f"[P{pid}] WARN {w}")
            return True

    return False


def action_attack_monster(
    state: GameState,
    engine: Engine,
    pid: int,
    monster_id: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
) -> bool:
    p = state.players[pid]
    if p.action_points < 2:
        return False
    if monster_id not in state.monster_row:
        return False
    if not can_player_attack_monster(p, engine, monster_id):
        log.append(f"[P{pid}] WARN cannot attack monster {monster_id} (requirements unmet)")
        return False

    rule = engine.monster_attack_rules.get(monster_id)
    if not rule or not rule.success_condition:
        log.append(f"[P{pid}] WARN monster {monster_id} has no on_attacked rule/success_condition")
        return False

    p.action_points -= 2
    log.append(
        f"[P{pid}] ACTION attack monster (cost 2) -> {monster_id} "
        f"({engine.card_meta.get(monster_id,{}).get('name','?')})"
    )

    op, target = parse_simple_condition(rule.success_condition)
    fail_op = None
    fail_target = None
    if rule.fail_condition:
        fail_op, fail_target = parse_simple_condition(rule.fail_condition)
    final = resolve_roll_event(
        state=state,
        engine=engine,
        roller_pid=pid,
        roll_reason=f"monster:{engine.card_meta.get(monster_id,{}).get('name','?')}",
        rng=rng,
        log=log,
        policy=policy,
        goal=(op, target),
        mode="threshold",
    )
    success = goal_satisfied(final, op, target)
    if fail_op and fail_target is not None:
        fail = goal_satisfied(final, fail_op, fail_target)
    else:
        fail = not success
    if success:
        outcome = "SUCCESS"
    elif fail:
        outcome = "FAIL"
    else:
        outcome = "NO_EFFECT"
    log.append(
        f"[P{pid}] monster attack roll 2d6={final} -> {outcome} "
        f"(success:{rule.success_condition} fail:{rule.fail_condition})"
    )

    ctx = {
        "attack_roll": final,
        "attack.success": success,
        "attack.fail": fail,
        "attack.no_effect": not success and not fail,
        "target_monster_id": monster_id,
    }

    for step in engine.monster_effects.get(monster_id, []):
        resolve_effect(step, state, engine, pid, ctx, rng, policy, log)

    if success:
        if monster_id in state.monster_row:
            state.monster_row.remove(monster_id)
            p.captured_monsters.append(monster_id)
            log.append(f"[P{pid}] captured monster -> {monster_id}")

        if state.monster_deck:
            new_mid = state.monster_deck.pop()
            state.monster_row.append(new_mid)
            log.append(
                f"[SETUP] refill monster_row -> {new_mid} "
                f"({engine.card_meta.get(new_mid,{}).get('name','?')})"
            )

    for w in ctx.get("_warnings", []):
        log.append(f"[P{pid}] WARN {w}")

    return True


def choose_and_take_action(
    state: GameState,
    engine: Engine,
    pid: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
) -> bool:
    p = state.players[pid]
    if p.action_points <= 0:
        return False

    if p.action_points >= 2 and state.monster_row:
        eligible_monsters = [
            mid for mid in state.monster_row if can_player_attack_monster(p, engine, mid)
        ]
        target_monster = policy.choose_monster_to_attack(eligible_monsters, engine)
        if target_monster is not None:
            return action_attack_monster(state, engine, pid, target_monster, rng, policy, log)

    if action_activate_hero(state, engine, pid, rng, policy, log):
        return True

    chosen = policy.choose_card_to_play(p.hand, engine)
    if chosen is not None:
        play_card_from_hand(state, engine, pid, chosen, rng, policy, log)
        return True

    return action_draw(state, engine, pid, rng, policy, log)
