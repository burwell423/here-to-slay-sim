import re
from typing import Any, Dict, List, Optional

from .conditions import eval_condition, goal_satisfied, parse_simple_condition
from .game_helpers import (
    attacker_choose_hero_to_destroy,
    collect_party_classes,
    destroy_hero_card,
    get_zone,
    pick_opponent_pid,
    victim_choose_hero_to_sacrifice,
)
from .models import EffectStep, Engine, GameState, Policy
from .rolls import resolve_roll_event
from .utils import format_card_list


def _parse_hero_class_from_notes(notes: str) -> Optional[str]:
    if not notes:
        return None
    m = re.search(r"hero\\.type\\s*==\\s*([a-zA-Z_][\\w-]*)", notes)
    if not m:
        return None
    return m.group(1).strip().lower()


def _find_hero_owner(state: GameState, hero_id: int) -> Optional[int]:
    for p in state.players:
        if hero_id in p.party:
            return p.pid
    return None


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
    target_pid = pid
    if step.source_zone:
        zone = step.source_zone.strip().lower()
        if zone == "challenge.source":
            challenger = ctx.get("challenge", {}).get("challenger_pid")
            if challenger is None:
                ctx.setdefault("_warnings", []).append("discard_card: missing challenge.challenger_pid")
            else:
                target_pid = challenger
        elif zone == "challenge.target":
            target = ctx.get("challenge", {}).get("target_pid")
            if target is None:
                ctx.setdefault("_warnings", []).append("discard_card: missing challenge.target_pid")
            else:
                target_pid = target
    hand = state.players[target_pid].hand
    for _ in range(min(n, len(hand))):
        chosen = policy.choose_discard_card(hand, engine)
        if chosen is None:
            return
        hand.remove(chosen)
        cid = chosen
        state.discard_pile.append(cid)
        log.append(f"[P{target_pid}] discarded card_id={cid}")


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
    chosen = None
    if step.source_zone.strip().lower() == "monster_row":
        target_monster_id = ctx.get("target_monster_id")
        if target_monster_id in src:
            chosen = target_monster_id
    if chosen is None:
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
    if ctx.get("protect.steal"):
        log.append(f"[P{pid}] steal_card blocked by protection ({step.name})")
        return
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


def _resolve_party_pid(state: GameState, pid: int, zone: Optional[str]) -> Optional[int]:
    if not zone:
        return None
    z = zone.strip().lower()
    if z.startswith("player."):
        return pid
    if z.startswith("opponent."):
        return pick_opponent_pid(state, pid)
    if z.startswith("opponents."):
        return pick_opponent_pid(state, pid)
    return None


def _handle_steal_hero(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    if ctx.get("protect.steal"):
        log.append(f"[P{pid}] steal_hero blocked by protection ({step.name})")
        return
    source_pid = _resolve_party_pid(state, pid, step.source_zone)
    dest_pid = _resolve_party_pid(state, pid, step.dest_zone)
    if source_pid is None or dest_pid is None:
        ctx.setdefault("_warnings", []).append(f"steal_hero: unresolved source/dest {step.source_zone}->{step.dest_zone}")
        return
    source_party = state.players[source_pid].party
    if not source_party:
        return
    amount = step.amount if step.amount is not None else 1
    for _ in range(min(amount, len(source_party))):
        chosen: Optional[int] = None
        if step.filter_expr:
            filter_expr = str(step.filter_expr).strip().lower()
            if filter_expr == "hero==active":
                active = ctx.get("activated_hero_id")
                if isinstance(active, int) and active in source_party:
                    chosen = active
            elif filter_expr == "hero==destroyed":
                destroyed = ctx.get("destroyed_hero_id")
                if destroyed is None and isinstance(ctx.get("hero_destroyed"), dict):
                    destroyed = ctx["hero_destroyed"].get("id")
                if isinstance(destroyed, int) and destroyed in source_party:
                    chosen = destroyed
        if chosen is None:
            chosen = policy.choose_steal_hero(source_party, engine, state.players[source_pid].hero_items)
        if chosen is None:
            return
        if chosen not in source_party:
            return
        source_party.remove(chosen)
        state.players[dest_pid].party.append(chosen)
        items = list(state.players[source_pid].hero_items.get(chosen, []))
        if items:
            state.players[source_pid].hero_items[chosen] = []
            state.players[dest_pid].hero_items[chosen].extend(items)
        overrides = state.players[source_pid].hero_class_overrides.pop(chosen, None)
        if overrides:
            state.players[dest_pid].hero_class_overrides[chosen] = overrides
        ctx["stolen_hero"] = engine.card_meta.get(chosen, {"id": chosen, "type": "hero"})
        log.append(
            f"[P{pid}] stole hero {chosen} from P{source_pid} -> P{dest_pid}"
            f"{' (with items)' if items else ''}"
        )


def _transfer_hero(
    state: GameState,
    engine: Engine,
    source_pid: int,
    dest_pid: int,
    hero_id: int,
    log: List[str],
    label: str,
) -> bool:
    source = state.players[source_pid]
    dest = state.players[dest_pid]
    if hero_id not in source.party:
        return False
    source.party.remove(hero_id)
    dest.party.append(hero_id)
    items = list(source.hero_items.get(hero_id, []))
    if items:
        source.hero_items[hero_id] = []
        dest.hero_items[hero_id].extend(items)
    overrides = source.hero_class_overrides.pop(hero_id, None)
    if overrides:
        dest.hero_class_overrides[hero_id] = overrides
    log.append(
        f"[P{source_pid}] {label} hero {hero_id}:{engine.card_meta.get(hero_id,{}).get('name','?')} -> P{dest_pid}"
        f"{' (with items)' if items else ''}"
    )
    return True


def _handle_swap_hero(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    if ctx.get("protect.steal"):
        log.append(f"[P{pid}] swap_hero blocked by protection ({step.name})")
        return
    source_pid = _resolve_party_pid(state, pid, step.source_zone)
    dest_pid = _resolve_party_pid(state, pid, step.dest_zone)
    if source_pid is None or dest_pid is None:
        ctx.setdefault("_warnings", []).append(f"swap_hero: unresolved source/dest {step.source_zone}->{step.dest_zone}")
        return
    active_hero_id = ctx.get("activated_hero_id")
    if not isinstance(active_hero_id, int):
        ctx.setdefault("_warnings", []).append("swap_hero: missing activated hero")
        return
    active_owner = _find_hero_owner(state, active_hero_id)
    if active_owner is None:
        ctx.setdefault("_warnings", []).append("swap_hero: activated hero not found")
        return

    source_party = state.players[source_pid].party
    if not source_party:
        return
    amount = step.amount if step.amount is not None else 1
    for _ in range(min(amount, len(source_party))):
        chosen: Optional[int] = None
        if step.filter_expr and str(step.filter_expr).strip().lower() == "hero==active":
            if active_hero_id in source_party:
                chosen = active_hero_id
        if chosen is None:
            chosen = policy.choose_steal_hero(source_party, engine, state.players[source_pid].hero_items)
        if chosen is None:
            return
        if chosen not in source_party:
            return
        if not _transfer_hero(state, engine, source_pid, dest_pid, chosen, log, "swap_hero stole"):
            return
        if not _transfer_hero(state, engine, active_owner, source_pid, active_hero_id, log, "swap_hero sent"):
            return
        ctx["stolen_hero"] = engine.card_meta.get(chosen, {"id": chosen, "type": "hero"})


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


def _handle_play_card(
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

    hand = state.players[pid].hand
    if not hand:
        return

    candidates = [cid for cid in hand if _filter_matches_card(engine, cid, step.filter_expr)]
    if not candidates:
        return

    amount = step.amount if step.amount is not None else 1
    for _ in range(min(amount, len(candidates))):
        chosen = policy.choose_card_to_play(candidates, engine)
        if chosen is None:
            return
        if chosen not in candidates:
            return
        candidates.remove(chosen)
        play_card_from_hand(
            state=state,
            engine=engine,
            pid=pid,
            card_id=chosen,
            rng=rng,
            policy=policy,
            log=log,
            cost_override=0,
            allow_challenge=True,
        )

    state.players[pid].action_points += 1
    log.append(f"[P{pid}] play_card granted +1 action (now {state.players[pid].action_points})")


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


def _handle_trade_hands(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    target_pid = ctx.get("target_pid")
    if target_pid is None:
        target_pid = policy.choose_trade_partner(state, pid)
    if target_pid is None:
        ctx.setdefault("_warnings", []).append("trade_hands: no opponent available")
        return
    if target_pid == pid:
        ctx.setdefault("_warnings", []).append("trade_hands: target is self")
        return
    player_hand = state.players[pid].hand
    target_hand = state.players[target_pid].hand
    state.players[pid].hand = list(target_hand)
    state.players[target_pid].hand = list(player_hand)
    log.append(
        f"[P{pid}] trade_hands with P{target_pid} "
        f"({len(player_hand)} cards -> {len(state.players[pid].hand)}, "
        f"{len(target_hand)} cards -> {len(state.players[target_pid].hand)})"
    )


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


def _resolve_hero_destruction(
    state: GameState,
    engine: Engine,
    source_pid: int,
    victim_pid: int,
    hero_id: int,
    rng: "random.Random",
    policy: Policy,
    log: List[str],
) -> bool:
    ctx: Dict[str, Any] = {
        "hero_destroyed": {"id": hero_id, "target": victim_pid, "source": source_pid},
        "hero": {"id": hero_id, "owner": victim_pid},
        "destroyed_hero_id": hero_id,
    }

    pre_effects = {"deny", "steal_hero", "protection_from_destroy"}
    for owner in state.players:
        for mid in owner.captured_monsters:
            for step in engine.monster_effects.get(mid, []):
                if "on_hero_destroyed" not in step.triggers():
                    continue
                if (step.effect_kind or "").strip() not in pre_effects:
                    continue
                resolve_effect(step, state, engine, owner.pid, ctx, rng, policy, log)

    if ctx.get("denied") or ctx.get("protect.destroy"):
        log.append(f"[P{victim_pid}] hero {hero_id} destruction prevented")
        return False

    if hero_id not in state.players[victim_pid].party:
        return False

    destroyed = destroy_hero_card(state, engine, victim_pid, hero_id, log)
    if not destroyed:
        return False

    for owner in state.players:
        for mid in owner.captured_monsters:
            for step in engine.monster_effects.get(mid, []):
                if "on_hero_destroyed" not in step.triggers():
                    continue
                if (step.effect_kind or "").strip() in pre_effects:
                    continue
                resolve_effect(step, state, engine, owner.pid, ctx, rng, policy, log)

    return True


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
    _resolve_hero_destruction(state, engine, pid, victim_pid, hid, rng, policy, log)


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
    _resolve_hero_destruction(state, engine, victim_pid, victim_pid, hid, rng, policy, log)


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


def _handle_destroy_item(
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
    hero_id = ctx.get("attached_to_hero") or ctx.get("activated_hero_id")
    if hero_id is None:
        p = state.players[victim_pid]
        candidates = [hid for hid in p.party if p.hero_items.get(hid)]
        if not candidates:
            ctx.setdefault("_warnings", []).append("destroy_item: no heroes with items")
            return
        hero_id = sorted(candidates, key=lambda hid: (-len(p.hero_items.get(hid, [])), hid))[0]

    p = state.players[victim_pid]
    items = p.hero_items.get(hero_id, [])
    if not items:
        ctx.setdefault("_warnings", []).append("destroy_item: hero has no items")
        return

    item_id = policy.choose_item_to_destroy(items, engine)
    if item_id is None:
        ctx.setdefault("_warnings", []).append("destroy_item: no item selected")
        return

    items.remove(item_id)
    state.discard_pile.append(item_id)
    overrides = p.hero_class_overrides.get(hero_id)
    if overrides:
        p.hero_class_overrides[hero_id] = [entry for entry in overrides if entry[0] != item_id]
        if not p.hero_class_overrides[hero_id]:
            p.hero_class_overrides.pop(hero_id, None)
    log.append(
        f"[P{pid}] destroy_item -> removed {item_id}:{engine.card_meta.get(item_id,{}).get('name','?')} "
        f"from P{victim_pid} hero {hero_id}"
    )


def _handle_look_at_hand(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    target_pid = ctx.get("target_pid")
    if target_pid is None:
        candidates = [p for p in state.players if p.pid != pid and p.hand]
        if not candidates:
            ctx.setdefault("_warnings", []).append("look_at_hand: no opponents with cards in hand")
            return
        target = max(candidates, key=lambda p: (len(p.hand), -p.pid))
        target_pid = target.pid
    hand = state.players[target_pid].hand
    log.append(
        f"[P{pid}] look_at_hand sees P{target_pid} hand: {format_card_list(hand, engine.card_meta)}"
    )


def _handle_reveal_card(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    source = (step.source_zone or "").strip().lower()
    dest = (step.dest_zone or "").strip().lower()

    if source == "drawn_card":
        drawn = ctx.get("drawn_card")
        if not drawn:
            ctx.setdefault("_warnings", []).append("reveal_card: missing ctx.drawn_card")
            return
        card_name = drawn.get("name", "?")
        card_id = drawn.get("id", "?")
        if dest == "all_opponents":
            opp_ids = [p.pid for p in state.players if p.pid != pid]
            log.append(
                f"[P{pid}] reveal_card shows drawn {card_id}:{card_name} to opponents {opp_ids}"
            )
        else:
            log.append(f"[P{pid}] reveal_card sees drawn {card_id}:{card_name}")
        return

    target_pid = ctx.get("target_pid")
    if target_pid is None:
        candidates = [p for p in state.players if p.pid != pid and p.hand]
        if not candidates:
            ctx.setdefault("_warnings", []).append("reveal_card: no opponents with cards in hand")
            return
        target_pid = policy.choose_reveal_opponent(candidates)
        if target_pid is None:
            ctx.setdefault("_warnings", []).append("reveal_card: no opponent selected")
            return

    hand = state.players[target_pid].hand
    if not hand:
        ctx.setdefault("_warnings", []).append("reveal_card: target has no cards in hand")
        return
    revealed = policy.choose_reveal_card(hand, engine)
    if revealed is None:
        ctx.setdefault("_warnings", []).append("reveal_card: no card selected")
        return
    log.append(
        f"[P{pid}] reveal_card sees P{target_pid} card {revealed}:{engine.card_meta.get(revealed,{}).get('name','?')}"
    )


def _handle_modify_action_total(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    delta = step.amount if step.amount is not None else 1
    state.players[pid].actions_per_turn += delta
    log.append(f"[P{pid}] modify_action_total {delta:+d} -> {state.players[pid].actions_per_turn}")


def _extract_modifier_deltas(step: EffectStep) -> List[int]:
    def parse_numbers(texts: List[str]) -> List[int]:
        values: List[int] = []
        seen = set()
        for text in texts:
            if not text:
                continue
            for raw in re.findall(r"[-+]?\d+", str(text)):
                try:
                    val = int(raw)
                except ValueError:
                    continue
                if val in seen:
                    continue
                seen.add(val)
                values.append(val)
        return values

    primary: List[str] = []
    if step.amount is not None:
        primary.append(str(step.amount))
    if step.amount_expr:
        primary.append(step.amount_expr)
    deltas = parse_numbers(primary)
    if deltas:
        return deltas

    fallback: List[str] = []
    if step.notes:
        fallback.append(step.notes)
    if step.filter_expr:
        fallback.append(step.filter_expr)
    return parse_numbers(fallback)


def _handle_modify_roll(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    deltas = _extract_modifier_deltas(step)
    if not deltas:
        ctx.setdefault("_warnings", []).append("modify_roll: missing delta")
        return

    expires_turn: Optional[int] = None
    if step.duration and step.duration.strip().lower() == "end_of_turn":
        expires_turn = state.turn
    if "passive" in step.triggers():
        expires_turn = state.turn

    for delta in deltas:
        state.players[pid].roll_modifiers.append((step.card_id, delta, expires_turn))
        log.append(
            f"[P{pid}] modify_roll adds {delta:+d} "
            f"({engine.card_meta.get(step.card_id,{}).get('name','?')})"
        )


def _handle_modify_hero_class(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    hero_id = ctx.get("attached_to_hero") or ctx.get("activated_hero_id")
    if hero_id is None:
        ctx.setdefault("_warnings", []).append("modify_hero_class: missing target hero")
        return

    hero_class = _parse_hero_class_from_notes(step.notes or "")
    if not hero_class:
        ctx.setdefault("_warnings", []).append("modify_hero_class: missing hero class")
        return

    owner_pid = _find_hero_owner(state, hero_id)
    if owner_pid is None:
        ctx.setdefault("_warnings", []).append("modify_hero_class: hero not found in any party")
        return

    owner = state.players[owner_pid]
    overrides = owner.hero_class_overrides.get(hero_id, [])
    overrides = [entry for entry in overrides if entry[0] != step.card_id]
    overrides.append((step.card_id, hero_class))
    owner.hero_class_overrides[hero_id] = overrides

    log.append(
        f"[P{pid}] modify_hero_class -> hero {hero_id}:{engine.card_meta.get(hero_id,{}).get('name','?')} "
        f"set to {hero_class} ({step.name})"
    )


def _filter_matches_card(engine: Engine, card_id: int, filter_expr: Optional[str]) -> bool:
    if not filter_expr:
        return True
    expr = str(filter_expr).strip().lower()
    meta = engine.card_meta.get(card_id, {})
    ctype = str(meta.get("type", "")).lower()
    subtype = str(meta.get("subtype", "")).lower()
    if expr.startswith("type=="):
        want = expr.split("==", 1)[1].strip()
        return ctype == want
    if expr.startswith("item.type=="):
        want = expr.split("==", 1)[1].strip()
        return ctype == "item" and subtype == want
    return True


def _remove_item_overrides(player, item_id: int) -> None:
    to_clear = []
    for hero_id, overrides in player.hero_class_overrides.items():
        filtered = [entry for entry in overrides if entry[0] != item_id]
        if filtered:
            player.hero_class_overrides[hero_id] = filtered
        else:
            to_clear.append(hero_id)
    for hero_id in to_clear:
        player.hero_class_overrides.pop(hero_id, None)


def _collect_return_candidates(
    state: GameState,
    engine: Engine,
    target_pid: int,
    filter_expr: Optional[str],
) -> List[Dict[str, Any]]:
    p = state.players[target_pid]
    candidates: List[Dict[str, Any]] = []
    for hero_id in p.party:
        if _filter_matches_card(engine, hero_id, filter_expr):
            candidates.append({"card_id": hero_id, "kind": "hero", "hero_id": hero_id})
        for item_id in p.hero_items.get(hero_id, []):
            if _filter_matches_card(engine, item_id, filter_expr):
                candidates.append({"card_id": item_id, "kind": "item", "hero_id": hero_id})
    return candidates


def _choose_target_pid_for_return(
    state: GameState,
    engine: Engine,
    pid: int,
    filter_expr: Optional[str],
    prefer_opponents: bool,
) -> Optional[int]:
    candidate_pids = [p.pid for p in state.players if _collect_return_candidates(state, engine, p.pid, filter_expr)]
    if not candidate_pids:
        return None
    if prefer_opponents:
        opponents = [opid for opid in candidate_pids if opid != pid]
        if opponents:
            return max(opponents, key=lambda opid: (len(state.players[opid].party), -opid))
    return max(candidate_pids, key=lambda opid: (len(state.players[opid].party), -opid))


def _move_return_candidate_to_hand(
    state: GameState,
    engine: Engine,
    target_pid: int,
    candidate: Dict[str, Any],
    log: List[str],
) -> None:
    p = state.players[target_pid]
    card_id = int(candidate["card_id"])
    if candidate["kind"] == "item":
        hero_id = int(candidate["hero_id"])
        items = p.hero_items.get(hero_id, [])
        if card_id in items:
            items.remove(card_id)
        _remove_item_overrides(p, card_id)
        p.hand.append(card_id)
        log.append(
            f"[P{target_pid}] return_to_hand item {card_id}:{engine.card_meta.get(card_id,{}).get('name','?')} "
            f"from hero {hero_id}"
        )
        return

    hero_id = int(candidate["hero_id"])
    if hero_id not in p.party:
        return
    p.party.remove(hero_id)
    p.hand.append(hero_id)
    items = list(p.hero_items.get(hero_id, []))
    if items:
        for item_id in items:
            p.hand.append(item_id)
            _remove_item_overrides(p, item_id)
        p.hero_items[hero_id] = []
        log.append(
            f"[P{target_pid}] return_to_hand hero {hero_id}:{engine.card_meta.get(hero_id,{}).get('name','?')} "
            f"with items {format_card_list(items, engine.card_meta)}"
        )
    else:
        log.append(
            f"[P{target_pid}] return_to_hand hero {hero_id}:{engine.card_meta.get(hero_id,{}).get('name','?')}"
        )
    p.hero_class_overrides.pop(hero_id, None)


def _handle_return_to_hand(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    src = (step.source_zone or "").strip().lower()
    dst = (step.dest_zone or "").strip().lower()
    amount = step.amount

    def resolve_for_pid(target_pid: int) -> None:
        candidates = _collect_return_candidates(state, engine, target_pid, step.filter_expr)
        if not candidates:
            return
        if amount is None:
            for cand in list(candidates):
                _move_return_candidate_to_hand(state, engine, target_pid, cand, log)
            return
        for _ in range(min(amount, len(candidates))):
            choice = policy.choose_move_card([c["card_id"] for c in candidates], "player.hand", engine)
            if choice is None:
                return
            selected = next((c for c in candidates if c["card_id"] == choice), None)
            if selected is None:
                return
            candidates.remove(selected)
            _move_return_candidate_to_hand(state, engine, target_pid, selected, log)

    if src.startswith("all_players."):
        for target in state.players:
            resolve_for_pid(target.pid)
        return

    if src.startswith("any_player."):
        target_pid = ctx.get("target_pid")
        if target_pid is None:
            target_pid = _choose_target_pid_for_return(state, engine, pid, step.filter_expr, prefer_opponents=True)
        if target_pid is None:
            ctx.setdefault("_warnings", []).append("return_to_hand: no valid target for any_player")
            return
        resolve_for_pid(target_pid)
        return

    if dst.endswith(".hand") and "target_pid" in ctx:
        resolve_for_pid(int(ctx["target_pid"]))
        return

    resolve_for_pid(pid)


def _handle_use_hero(
    step: EffectStep,
    state: GameState,
    engine: Engine,
    pid: int,
    ctx: Dict[str, Any],
    rng: "random.Random",
    policy: Policy,
    log: List[str],
):
    target_pid = _resolve_party_pid(state, pid, step.source_zone) or pid
    party = state.players[target_pid].party
    if not party:
        return

    hero_id: Optional[int] = None
    if step.filter_expr:
        filt = str(step.filter_expr).strip().lower()
        if filt == "hero==stolen_now":
            stolen = ctx.get("stolen_hero")
            if isinstance(stolen, dict):
                hero_id = stolen.get("id")
            elif isinstance(stolen, int):
                hero_id = stolen
        elif filt == "hero==active":
            hero_id = ctx.get("activated_hero_id")

    if hero_id is None:
        hero_id = policy.choose_steal_hero(party, engine, state.players[target_pid].hero_items)

    if hero_id is None or hero_id not in party:
        ctx.setdefault("_warnings", []).append("use_hero: missing target hero in party")
        return

    if hero_id in state.players[target_pid].activated_heroes_this_turn:
        log.append(
            f"[P{pid}] use_hero skipped -> {hero_id} "
            f"({engine.card_meta.get(hero_id,{}).get('name','?')}) already activated this turn"
        )
        return

    state.players[target_pid].activated_heroes_this_turn.add(hero_id)
    log.append(
        f"[P{pid}] use_hero -> {hero_id} "
        f"({engine.card_meta.get(hero_id,{}).get('name','?')})"
    )

    local_ctx = dict(ctx)
    local_ctx["activated_hero_id"] = hero_id
    local_ctx["used_hero_id"] = hero_id
    for hero_step in engine.effects_by_card.get(hero_id, []):
        trig = hero_step.triggers()
        if "on_activation" in trig or "auto" in trig:
            resolve_effect(hero_step, state, engine, pid, local_ctx, rng, policy, log)

    for w in local_ctx.get("_warnings", []):
        ctx.setdefault("_warnings", []).append(w)


EFFECT_HANDLERS = {
    "draw_card": _handle_draw,
    "draw_cards": _handle_draw,
    "discard_card": _handle_discard,
    "discard_cards": _handle_discard,
    "move_card": _handle_move,
    "steal_card": _handle_steal,
    "steal_hero": _handle_steal_hero,
    "swap_hero": _handle_swap_hero,
    "play_immediately": _handle_play_immediately,
    "play_drawn_immediately": _handle_play_drawn_immediately,
    "play_card": _handle_play_card,
    "deny_challenge": _handle_deny_challenge,
    "trade_hands": _handle_trade_hands,
    "search_and_draw": _handle_search_and_draw,
    "destroy_hero": _handle_destroy_hero,
    "sacrifice_hero": _handle_sacrifice_hero,
    "do_nothing": _handle_do_nothing,
    "deny": _handle_deny,
    "protection_from_steal": _handle_protection_from_steal,
    "protection_from_destroy": _handle_protection_from_destroy,
    "protection_from_challenge": _handle_protection_from_challenge,
    "destroy_item": _handle_destroy_item,
    "look_at_hand": _handle_look_at_hand,
    "reveal_card": _handle_reveal_card,
    "modify_action_total": _handle_modify_action_total,
    "modify_roll": _handle_modify_roll,
    "modify_hero_class": _handle_modify_hero_class,
    "return_to_hand": _handle_return_to_hand,
    "use_hero": _handle_use_hero,
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
    ctx.setdefault("player", pid)
    if step.step == 2 and not ctx.get("roll.success", False):
        return
    if "opponent" not in ctx:
        target_pid = ctx.get("target_pid")
        if target_pid is None:
            opponent_classes = set()
            for pstate in state.players:
                if pstate.pid == pid:
                    continue
                opponent_classes.update(collect_party_classes(engine, pstate))
        elif target_pid == pid:
            opponent_classes = set()
        else:
            opponent_classes = collect_party_classes(engine, state.players[target_pid])
        ctx["opponent"] = {"party": opponent_classes}

    if not eval_condition(step.condition, ctx):
        return

    if step.dest_zone:
        dz = step.dest_zone.strip().lower()
        if dz == "opponent" and "target_pid" not in ctx:
            ctx["target_pid"] = pick_opponent_pid(state, pid)
        if dz == "self" and "target_pid" not in ctx:
            ctx["target_pid"] = pid

    if step.requires_roll:
        for mid in state.players[pid].captured_monsters:
            for mstep in engine.monster_effects.get(mid, []):
                if "on_hero_roll" in mstep.triggers():
                    resolve_effect(mstep, state, engine, pid, ctx, rng, policy, log)

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
            hero_id=ctx.get("activated_hero_id"),
            policy=policy,
        )
        ok = goal_satisfied(final, op, target)
        log.append(
            f"[P{pid}] roll 2d6={final} vs {step.roll_condition} -> {'PASS' if ok else 'FAIL'} ({step.name})"
        )
        if not ok:
            return

        ctx.update({"roll.total": final, "roll.success": ok, "roll_player": pid})
        for mid in state.players[pid].captured_monsters:
            for mstep in engine.monster_effects.get(mid, []):
                if "on_hero_roll_success" in mstep.triggers():
                    resolve_effect(mstep, state, engine, pid, ctx, rng, policy, log)

    ek = (step.effect_kind or "").strip()
    handler = EFFECT_HANDLERS.get(ek)
    if handler is None:
        ctx.setdefault("_warnings", []).append(f"UNIMPLEMENTED_EFFECT_KIND: {ek} ({step.name})")
        return

    handler(step, state, engine, pid, ctx, rng, policy, log)
