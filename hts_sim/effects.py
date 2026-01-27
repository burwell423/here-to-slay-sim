import re
from typing import Any, Dict, List

from .conditions import eval_condition, goal_satisfied, parse_simple_condition
from .game_helpers import (
    attacker_choose_hero_to_destroy,
    destroy_hero_card,
    get_zone,
    pick_opponent_pid,
    victim_choose_hero_to_sacrifice,
)
from .models import EffectStep, Engine, GameState, Policy
from .rolls import resolve_roll_event


def _handle_draw(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    n = step.amount if step.amount is not None else 1
    for _ in range(n):
        if not state.draw_pile:
            break
        cid = state.draw_pile.pop()
        state.players[pid].hand.append(cid)
        ctx["drawn_card"] = engine.card_meta.get(cid, {"id": cid, "type": "unknown"})
        log.append(f"[P{pid}] drew card_id={cid} ({ctx['drawn_card'].get('name','?')})")


def _handle_discard(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    n = step.amount if step.amount is not None else 1
    hand = state.players[pid].hand
    for _ in range(min(n, len(hand))):
        chosen = policy.choose_discard_card(hand, engine)
        if chosen is None:
            return
        hand.remove(chosen)
        cid = chosen
        state.discard_pile.append(cid)
        log.append(f"[P{pid}] discarded card_id={cid}")


def _handle_move(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    if not step.source_zone or not step.dest_zone:
        ctx.setdefault("_warnings", []).append(f"MOVE_CARD_MISSING_ZONES: {step.name}")
        return
    src = get_zone(state, pid, step.source_zone)
    dst = get_zone(state, pid, step.dest_zone)
    if not src:
        return
    chosen = policy.choose_move_card(src, step.dest_zone.strip(), engine)
    if chosen is None:
        return
    src.remove(chosen)
    cid = chosen
    dst.append(cid)
    log.append(f"[P{pid}] move_card {cid} {step.source_zone} -> {step.dest_zone}")


def _handle_steal(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    opp = (pid + 1) % len(state.players)
    opp_hand = state.players[opp].hand
    if not opp_hand:
        return
    chosen = policy.choose_steal_card(opp_hand, engine)
    if chosen is None:
        return
    opp_hand.remove(chosen)
    cid = chosen
    state.players[pid].hand.append(cid)
    ctx["stolen_card"] = engine.card_meta.get(cid, {"id": cid, "type": "unknown"})
    log.append(f"[P{pid}] stole card_id={cid} from P{opp}")


def _handle_play_immediately(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    stolen = ctx.get("stolen_card")
    if not isinstance(stolen, dict):
        return
    cid = int(stolen["id"])
    if cid in state.players[pid].hand:
        state.players[pid].hand.remove(cid)
        state.discard_pile.append(cid)
        log.append(f"[P{pid}] played immediately card_id={cid} (MVP: moved to discard)")


def _handle_play_drawn_immediately(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    from .actions import play_card_from_hand

    drawn = ctx.get("drawn_card")
    if not isinstance(drawn, dict):
        ctx.setdefault("_warnings", []).append("play_drawn_immediately missing ctx.drawn_card")
        return
    cid = int(drawn["id"])
    if cid not in state.players[pid].hand:
        ctx.setdefault("_warnings", []).append(f"play_drawn_immediately: card {cid} not in hand")
        return
    play_card_from_hand(
        state=state,
        engine=engine,
        pid=pid,
        card_id=cid,
        rng=rng,
        policy=policy,
        log=log,
        cost_override=0,
        allow_challenge=True,
    )


def _handle_deny_challenge(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    ctx["challenge.denied"] = True
    log.append(f"[P{pid}] deny_challenge triggered ({step.name})")


def _handle_search_and_draw(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    if not step.source_zone or not step.dest_zone:
        ctx.setdefault("_warnings", []).append(f"SEARCH_MISSING_ZONES: {step.name}")
        return
    src = get_zone(state, pid, step.source_zone)
    dst = get_zone(state, pid, step.dest_zone)

    want_type = None
    if step.filter_expr:
        m = re.match(r"^type==([a-zA-Z_]\w*)$", str(step.filter_expr).strip())
        if m:
            want_type = m.group(1).lower()

    found_idx = None
    for i, cid in enumerate(src):
        typ = str(engine.card_meta.get(cid, {}).get("type", "unknown")).lower()
        if want_type is None or typ == want_type:
            found_idx = i
            break
    if found_idx is None:
        return
    cid = src.pop(found_idx)
    dst.append(cid)
    log.append(f"[P{pid}] searched {step.source_zone} and took card_id={cid} to {step.dest_zone}")


def _handle_destroy_hero(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    victim_pid = ctx.get("target_pid")
    if victim_pid is None:
        victim_pid = pick_opponent_pid(state, pid)
    if victim_pid is None:
        ctx.setdefault("_warnings", []).append("destroy_hero: no valid opponent with heroes")
        return

    hid = attacker_choose_hero_to_destroy(state, engine, victim_pid)
    if hid is None:
        ctx.setdefault("_warnings", []).append("destroy_hero: victim has no heroes")
        return

    log.append(f"[P{pid}] destroy_hero targets P{victim_pid} hero {hid}")
    destroy_hero_card(state, engine, victim_pid, hid, log)


def _handle_sacrifice_hero(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    victim_pid = ctx.get("target_pid", pid)
    hid = victim_choose_hero_to_sacrifice(state, engine, victim_pid)
    if hid is None:
        ctx.setdefault("_warnings", []).append("sacrifice_hero: no heroes to sacrifice")
        return

    log.append(f"[P{pid}] sacrifice_hero by P{victim_pid} chooses hero {hid}")
    destroy_hero_card(state, engine, victim_pid, hid, log)


def _handle_do_nothing(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    return


def _handle_deny(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    if "challenge.denied" in ctx:
        ctx["challenge.denied"] = True
        log.append(f"[P{pid}] deny (challenge) via {step.name}")
    else:
        ctx["denied"] = True
        log.append(f"[P{pid}] deny via {step.name}")


def _handle_protection_from_steal(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    ctx["protect.steal"] = True


def _handle_protection_from_destroy(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    ctx["protect.destroy"] = True


def _handle_protection_from_challenge(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    ctx["protect.challenge"] = True


EFFECT_HANDLERS = {
    "draw_card": _handle_draw,
    "draw_cards": _handle_draw,
    "discard_card": _handle_discard,
    "discard_cards": _handle_discard,
    "move_card": _handle_move,
    "steal_card": _handle_steal,
    "play_immediately": _handle_play_immediately,
    "play_drawn_immediately": _handle_play_drawn_immediately,
    "deny_challenge": _handle_deny_challenge,
    "search_and_draw": _handle_search_and_draw,
    "destroy_hero": _handle_destroy_hero,
    "sacrifice_hero": _handle_sacrifice_hero,
    "do_nothing": _handle_do_nothing,
    "deny": _handle_deny,
    "protection_from_steal": _handle_protection_from_steal,
    "protection_from_destroy": _handle_protection_from_destroy,
    "protection_from_challenge": _handle_protection_from_challenge,
}

SUPPORTED_EFFECT_KINDS = set(EFFECT_HANDLERS.keys())


def resolve_effect(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    if not eval_condition(step.condition, ctx):
        return

    if step.dest_zone:
        dz = step.dest_zone.strip().lower()
        if dz == "opponent" and "target_pid" not in ctx:
            ctx["target_pid"] = pick_opponent_pid(state, pid)
        if dz == "self" and "target_pid" not in ctx:
            ctx["target_pid"] = pid

    if step.requires_roll:
        op, target = parse_simple_condition(step.roll_condition)
        final = resolve_roll_event(
            state=state,
            engine=engine,
            roller_pid=pid,
            roll_reason=f"hero:{step.name}",
            rng=rng,
            log=log,
            goal=(op, target),
            mode="threshold",
        )
        ok = goal_satisfied(final, op, target)
        log.append(
            f"[P{pid}] roll 2d6={final} vs {step.roll_condition} -> {'PASS' if ok else 'FAIL'} ({step.name})"
        )
        if not ok:
            return

    ek = (step.effect_kind or "").strip()
    handler = EFFECT_HANDLERS.get(ek)
    if handler is None:
        ctx.setdefault("_warnings", []).append(f"UNIMPLEMENTED_EFFECT_KIND: {ek} ({step.name})")
        return

    handler(step, state, engine, pid, ctx, rng, policy, log)
