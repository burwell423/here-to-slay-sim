#!/usr/bin/env python3
"""
Here to Slay - MVP simulator

Key features implemented:
- Separate draw deck / monster deck / party leader deck
- Setup: each player gets random party leader + 3 cards, reveal 3 monsters
- Turn structure: 3 actions; Draw (1), Play (1), Activate hero (1), Attack monster (2)
- Challenge cards: can challenge plays of hero/item/magic (not modifiers), cancelled on win
- Modifier cards: may be played by any player during any roll event; multi-option deltas supported
- Monsters:
  - on_attacked row defines success_condition / fail_condition + (optional) requirements
  - success_action / fail_action rows resolve via EffectSteps
  - captured monster passives via triggers like on_draw / on_challenge

Data files (CSV):
- cards - cards.csv
- cards - effects.csv
- cards - monsters.csv
"""

import ast
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import pandas as pd

EFFECTS_CSV = "cards - effects.csv"
MONSTERS_CSV = "cards - monsters.csv"
CARDS_CSV = "cards - cards.csv"


# -----------------------------
# Data structures
# -----------------------------

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
class Engine:
    effects_by_card: Dict[int, List[EffectStep]]
    card_meta: Dict[int, Dict[str, Any]]
    monster_attack_rules: Dict[int, MonsterRule]
    monster_effects: Dict[int, List[EffectStep]]
    modifier_options_by_card_id: Dict[int, List[int]]


# -----------------------------
# Utility: dice + conditions
# -----------------------------

ROLL_RE = re.compile(r"^\s*(?:2d6\s*)?(>=|<=|==|>|<)\s*(\d+)\s*$")
SUPPORTED_BOOL_NAMES = {"true": "True", "false": "False"}

def roll_2d6(rng: random.Random) -> int:
    return rng.randint(1, 6) + rng.randint(1, 6)

def check_roll(roll_value: int, cond: str) -> bool:
    """
    cond examples: '>=5', '<=7', '==9'
    """
    m = re.match(r"^\s*(>=|<=|==|>|<)\s*(\d+)\s*$", str(cond))
    if not m:
        raise ValueError(f"Unparseable roll_condition: {cond}")
    op, num_s = m.group(1), m.group(2)
    target = int(num_s)
    if op == ">=": return roll_value >= target
    if op == "<=": return roll_value <= target
    if op == "==": return roll_value == target
    if op == ">":  return roll_value > target
    if op == "<":  return roll_value < target
    raise ValueError(f"Unsupported operator: {op}")

def parse_roll_condition(cond: str) -> Optional[Tuple[str, int]]:
    """
    Accepts: '>=7', '<=5', '2d6>=9'
    Returns (op, target_int) or None if unparseable.
    """
    cond = str(cond).strip()
    m = ROLL_RE.match(cond)
    if not m:
        return None
    return m.group(1), int(m.group(2))

def parse_simple_condition(cond: str) -> Tuple[str, int]:
    parsed = parse_roll_condition(cond)
    if not parsed:
        raise ValueError(f"Unparseable condition: {cond}")
    return parsed

def goal_satisfied(total: int, op: str, target: int) -> bool:
    return check_roll(total, f"{op}{target}")

def is_challengeable_card_type(card_type: str) -> bool:
    t = (card_type or "").strip().lower()
    return t in ("hero", "item", "magic")

def find_challenge_card_in_hand(player: PlayerState, card_meta) -> Optional[int]:
    for cid in player.hand:
        if str(card_meta.get(cid, {}).get("type", "")).strip().lower() == "challenge":
            return cid
    return None

def find_modifier_cards(player: PlayerState, card_meta) -> List[int]:
    return [cid for cid in player.hand if str(card_meta.get(cid, {}).get("type","")).lower() == "modifier"]


# -----------------------------
# Condition evaluator (simple)
# -----------------------------

def _normalize_condition_text(cond: str) -> str:
    def replace_bool(match: re.Match[str]) -> str:
        word = match.group(0).lower()
        return SUPPORTED_BOOL_NAMES.get(word, word)

    return re.sub(r"\btrue\b|\bfalse\b", replace_bool, cond, flags=re.IGNORECASE)

def _eval_condition_node(node: ast.AST, ctx: Dict[str, Any]) -> Any:
    def dotted_name(n: ast.AST) -> Optional[str]:
        if isinstance(n, ast.Name):
            return n.id
        if isinstance(n, ast.Attribute):
            base = dotted_name(n.value)
            if base:
                return f"{base}.{n.attr}"
        return None

    if isinstance(node, ast.Expression):
        return _eval_condition_node(node.body, ctx)

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(bool(_eval_condition_node(v, ctx)) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(bool(_eval_condition_node(v, ctx)) for v in node.values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not bool(_eval_condition_node(node.operand, ctx))

    if isinstance(node, ast.Compare):
        left = _eval_condition_node(node.left, ctx)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_condition_node(comparator, ctx)
            if isinstance(op, ast.Eq):
                if left != right:
                    return False
            elif isinstance(op, ast.NotEq):
                if left == right:
                    return False
            else:
                raise ValueError("Unsupported comparison operator")
            left = right
        return True

    if isinstance(node, ast.Name):
        if node.id in ctx:
            return ctx[node.id]
        return node.id

    if isinstance(node, ast.Attribute):
        dotted = dotted_name(node)
        if dotted and dotted in ctx:
            return ctx[dotted]
        base = _eval_condition_node(node.value, ctx)
        if isinstance(base, dict):
            return base.get(node.attr)
        return getattr(base, node.attr, None)

    if isinstance(node, ast.Constant):
        return node.value

    raise ValueError("Unsupported expression node")

def is_condition_supported(cond: str) -> bool:
    try:
        _ = _eval_condition_node(ast.parse(_normalize_condition_text(cond), mode="eval"), {})
    except Exception:
        return False
    return True

def eval_condition(cond: Optional[str], ctx: Dict[str, Any]) -> bool:
    """
    Supported:
      - blank / NaN -> True
      - boolean expressions with and/or/not, ==, !=
      - attribute access into ctx dicts (e.g. challenge_target.type == 'item')
      - direct boolean flags in ctx (e.g. attack.success)
    """
    if cond is None:
        return True
    cond = str(cond).strip()
    if cond == "" or cond.lower() == "nan":
        return True

    try:
        parsed = ast.parse(_normalize_condition_text(cond), mode="eval")
        return bool(_eval_condition_node(parsed, ctx))
    except Exception:
        ctx.setdefault("_warnings", []).append(f"UNPARSEABLE_CONDITION: {cond}")
        return False


# -----------------------------
# Zone helpers
# -----------------------------

def get_zone(state: GameState, pid: int, zone: str) -> List[int]:
    p = state.players[pid]
    z = zone.strip()
    if z == "player.hand": return p.hand
    if z == "player.party": return p.party
    if z == "player.captured_monsters": return p.captured_monsters
    if z == "discard_pile": return state.discard_pile
    if z == "draw_pile": return state.draw_pile
    if z == "monster_row": return state.monster_row
    raise KeyError(f"Unknown zone: {zone}")

def destroy_hero_card(state: GameState, engine: Engine, victim_pid: int, hero_id: int, log: List[str]):
    p = state.players[victim_pid]
    if hero_id not in p.party:
        return False

    # discard attached items first
    items = list(p.hero_items.get(hero_id, []))
    if items:
        for item_id in items:
            state.discard_pile.append(item_id)
        p.hero_items[hero_id] = []
        log.append(f"[P{victim_pid}] hero {hero_id} dies -> discarded items: {format_card_list(items, engine.card_meta)}")

    # remove hero
    p.party.remove(hero_id)
    state.discard_pile.append(hero_id)
    log.append(f"[P{victim_pid}] hero destroyed/sacrificed -> {hero_id}:{engine.card_meta.get(hero_id,{}).get('name','?')}")
    return True

# -----------------------------
# Roll resolver with modifier window
# -----------------------------

def resolve_roll_event(state: GameState,
                       engine: Engine,
                       roller_pid: int,
                       roll_reason: str,
                       rng: random.Random,
                       log: List[str],
                       goal: Optional[Tuple[str, int]] = None,
                       mode: str = "threshold") -> int:
    """
    mode:
      - 'threshold': use goal ('>=',X) or ('<=',X) to decide whether to play mods
      - 'maximize': roller wants high, others want low (challenge-style)
    """
    base = roll_2d6(rng)
    total = base
    log.append(f"[ROLL:{roll_reason}] P{roller_pid} base 2d6 = {base}")

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
                # small preference; for <= we prefer lowering, for >= we prefer raising
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

        # play modifier
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


# -----------------------------
# Effect engine (MVP handlers)
# -----------------------------

def _handle_draw(step: EffectStep,
                 state: GameState,
                 engine: Engine,
                 pid: int,
                 ctx: Dict[str, Any],
                 rng: random.Random,
                 log: List[str]):
    n = step.amount if step.amount is not None else 1
    for _ in range(n):
        if not state.draw_pile:
            break
        cid = state.draw_pile.pop()
        state.players[pid].hand.append(cid)
        ctx["drawn_card"] = engine.card_meta.get(cid, {"id": cid, "type": "unknown"})
        log.append(f"[P{pid}] drew card_id={cid} ({ctx['drawn_card'].get('name','?')})")

def _handle_discard(step: EffectStep,
                    state: GameState,
                    engine: Engine,
                    pid: int,
                    ctx: Dict[str, Any],
                    rng: random.Random,
                    log: List[str]):
    n = step.amount if step.amount is not None else 1
    hand = state.players[pid].hand
    for _ in range(min(n, len(hand))):
        cid = hand.pop(0)  # MVP: discard first
        state.discard_pile.append(cid)
        log.append(f"[P{pid}] discarded card_id={cid}")

def _handle_move(step: EffectStep,
                 state: GameState,
                 engine: Engine,
                 pid: int,
                 ctx: Dict[str, Any],
                 rng: random.Random,
                 log: List[str]):
    if not step.source_zone or not step.dest_zone:
        ctx.setdefault("_warnings", []).append(f"MOVE_CARD_MISSING_ZONES: {step.name}")
        return
    src = get_zone(state, pid, step.source_zone)
    dst = get_zone(state, pid, step.dest_zone)
    if not src:
        return
    cid = src.pop(0)
    dst.append(cid)
    log.append(f"[P{pid}] move_card {cid} {step.source_zone} -> {step.dest_zone}")

def _handle_steal(step: EffectStep,
                  state: GameState,
                  engine: Engine,
                  pid: int,
                  ctx: Dict[str, Any],
                  rng: random.Random,
                  log: List[str]):
    opp = (pid + 1) % len(state.players)
    opp_hand = state.players[opp].hand
    if not opp_hand:
        return
    cid = opp_hand.pop(0)
    state.players[pid].hand.append(cid)
    ctx["stolen_card"] = engine.card_meta.get(cid, {"id": cid, "type": "unknown"})
    log.append(f"[P{pid}] stole card_id={cid} from P{opp}")

def _handle_play_immediately(step: EffectStep,
                             state: GameState,
                             engine: Engine,
                             pid: int,
                             ctx: Dict[str, Any],
                             rng: random.Random,
                             log: List[str]):
    stolen = ctx.get("stolen_card")
    if not isinstance(stolen, dict):
        return
    cid = int(stolen["id"])
    if cid in state.players[pid].hand:
        state.players[pid].hand.remove(cid)
        state.discard_pile.append(cid)
        log.append(f"[P{pid}] played immediately card_id={cid} (MVP: moved to discard)")

def _handle_play_drawn_immediately(step: EffectStep,
                                   state: GameState,
                                   engine: Engine,
                                   pid: int,
                                   ctx: Dict[str, Any],
                                   rng: random.Random,
                                   log: List[str]):
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
        log=log,
        cost_override=0,
        allow_challenge=True,
    )

def _handle_deny_challenge(step: EffectStep,
                           state: GameState,
                           engine: Engine,
                           pid: int,
                           ctx: Dict[str, Any],
                           rng: random.Random,
                           log: List[str]):
    ctx["challenge.denied"] = True
    log.append(f"[P{pid}] deny_challenge triggered ({step.name})")

def _handle_search_and_draw(step: EffectStep,
                            state: GameState,
                            engine: Engine,
                            pid: int,
                            ctx: Dict[str, Any],
                            rng: random.Random,
                            log: List[str]):
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

def _handle_destroy_hero(step: EffectStep,
                         state: GameState,
                         engine: Engine,
                         pid: int,
                         ctx: Dict[str, Any],
                         rng: random.Random,
                         log: List[str]):
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

def _handle_sacrifice_hero(step: EffectStep,
                           state: GameState,
                           engine: Engine,
                           pid: int,
                           ctx: Dict[str, Any],
                           rng: random.Random,
                           log: List[str]):
    victim_pid = ctx.get("target_pid", pid)
    hid = victim_choose_hero_to_sacrifice(state, engine, victim_pid)
    if hid is None:
        ctx.setdefault("_warnings", []).append("sacrifice_hero: no heroes to sacrifice")
        return

    log.append(f"[P{pid}] sacrifice_hero by P{victim_pid} chooses hero {hid}")
    destroy_hero_card(state, engine, victim_pid, hid, log)

def _handle_do_nothing(step: EffectStep,
                       state: GameState,
                       engine: Engine,
                       pid: int,
                       ctx: Dict[str, Any],
                       rng: random.Random,
                       log: List[str]):
    return

def _handle_deny(step: EffectStep,
                 state: GameState,
                 engine: Engine,
                 pid: int,
                 ctx: Dict[str, Any],
                 rng: random.Random,
                 log: List[str]):
    if "challenge.denied" in ctx:
        ctx["challenge.denied"] = True
        log.append(f"[P{pid}] deny (challenge) via {step.name}")
    else:
        ctx["denied"] = True
        log.append(f"[P{pid}] deny via {step.name}")

def _handle_protection_from_steal(step: EffectStep,
                                  state: GameState,
                                  engine: Engine,
                                  pid: int,
                                  ctx: Dict[str, Any],
                                  rng: random.Random,
                                  log: List[str]):
    ctx["protect.steal"] = True

def _handle_protection_from_destroy(step: EffectStep,
                                    state: GameState,
                                    engine: Engine,
                                    pid: int,
                                    ctx: Dict[str, Any],
                                    rng: random.Random,
                                    log: List[str]):
    ctx["protect.destroy"] = True

def _handle_protection_from_challenge(step: EffectStep,
                                      state: GameState,
                                      engine: Engine,
                                      pid: int,
                                      ctx: Dict[str, Any],
                                      rng: random.Random,
                                      log: List[str]):
    ctx["protect.challenge"] = True

EFFECT_HANDLERS: Dict[str, Any] = {
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

def resolve_effect(step: EffectStep,
                   state: GameState,
                   engine: Engine,
                   pid: int,
                   ctx: Dict[str, Any],
                   rng: random.Random,
                   log: List[str]):
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
        log.append(f"[P{pid}] roll 2d6={final} vs {step.roll_condition} -> {'PASS' if ok else 'FAIL'} ({step.name})")
        if not ok:
            return

    ek = (step.effect_kind or "").strip()
    handler = EFFECT_HANDLERS.get(ek)
    if handler is None:
        ctx.setdefault("_warnings", []).append(f"UNIMPLEMENTED_EFFECT_KIND: {ek} ({step.name})")
        return

    handler(step, state, engine, pid, ctx, rng, log)


# -----------------------------
# Challenge window
# -----------------------------

def maybe_challenge_play(state: GameState,
                         engine: Engine,
                         pid_playing: int,
                         played_card_id: int,
                         rng: random.Random,
                         log: List[str]) -> bool:
    """
    Returns True if play is cancelled by a successful challenge.
    """
    played_type = str(engine.card_meta.get(played_card_id, {}).get("type", "unknown")).lower()
    if not is_challengeable_card_type(played_type):
        return False

    # Context for on_challenge effects (captured monsters)
    ctx = {
        "challenge_target": engine.card_meta.get(played_card_id, {"id": played_card_id, "type": "unknown"}),
        "challenge.denied": False,
    }

    # Fire on_challenge triggers from ALL players' captured monsters (global passive)
    for pstate in state.players:
        for mid in pstate.captured_monsters:
            for step in engine.monster_effects.get(mid, []):
                if "on_challenge" in step.triggers():
                    resolve_effect(step, state, engine, pstate.pid, ctx, rng, log)

    if ctx.get("challenge.denied"):
        log.append("[Challenge] DENIED by effect")
        return False  # play continues, challenge cancelled

    # Find first opponent with a challenge card
    n = len(state.players)
    challenger_pid = None
    challenge_card_id = None
    for offset in range(1, n):
        opid = (pid_playing + offset) % n
        ccid = find_challenge_card_in_hand(state.players[opid], engine.card_meta)
        if ccid is not None:
            challenger_pid = opid
            challenge_card_id = ccid
            break

    if challenger_pid is None:
        return False

    # MVP: probabilistic challenge to avoid constant cancels
    if rng.random() > 0.35:
        return False

    # Consume challenge card
    challenger = state.players[challenger_pid]
    challenger.hand.remove(challenge_card_id)
    state.discard_pile.append(challenge_card_id)

    log.append(
        f"[P{challenger_pid}] CHALLENGE played {challenge_card_id} ({engine.card_meta.get(challenge_card_id,{}).get('name','?')}) "
        f"to challenge {played_card_id} ({engine.card_meta.get(played_card_id,{}).get('name','?')}) by P{pid_playing}"
    )

    # Roll-off with modifier windows in maximize mode
    r_challenger = resolve_roll_event(
        state=state, engine=engine,
        roller_pid=challenger_pid,
        roll_reason="challenge:challenger",
        rng=rng, log=log,
        mode="maximize",
    )
    r_playing = resolve_roll_event(
        state=state, engine=engine,
        roller_pid=pid_playing,
        roll_reason="challenge:played",
        rng=rng, log=log,
        mode="maximize",
    )

    log.append(f"[Challenge] P{challenger_pid} rolls {r_challenger} vs P{pid_playing} rolls {r_playing}")

    if r_challenger > r_playing:
        log.append("[Challenge] SUCCESS: play cancelled")
        return True
    else:
        log.append("[Challenge] FAIL: play continues")
        return False


# -----------------------------
# Core actions
# -----------------------------

def play_card_from_hand(state: GameState,
                        engine: Engine,
                        pid: int,
                        card_id: int,
                        rng: random.Random,
                        log: List[str],
                        cost_override: Optional[int] = None,
                        allow_challenge: bool = True):
    p = state.players[pid]
    if card_id not in p.hand:
        return
    cost = 1 if cost_override is None else cost_override
    if p.action_points < cost:
        return

    meta = engine.card_meta.get(card_id, {})
    ctype = str(meta.get("type", "unknown")).strip().lower()

    # remove from hand (commit attempt)
    p.hand.remove(card_id)
    log.append(f"[P{pid}] PLAY {card_id} ({meta.get('name','?')} / {ctype}) cost={cost}")

    # Challenge window
    if allow_challenge:
        cancelled = maybe_challenge_play(state, engine, pid, card_id, rng, log)
        if cancelled:
            state.discard_pile.append(card_id)
            log.append(f"[P{pid}] play of {card_id} cancelled -> discard")
            p.action_points -= cost
            return

    # Route card by type
    if ctype == "hero":
        p.party.append(card_id)
        log.append(f"[P{pid}] -> entered party: {card_id}:{meta.get('name','?')}")

    elif ctype == "item":
        if not p.party:
            state.discard_pile.append(card_id)
            log.append(f"[P{pid}] WARN played item with no heroes; discarded {card_id}")
        else:
            target_hero = p.party[0]  # MVP: first hero
            p.hero_items[target_hero].append(card_id)
            log.append(
                f"[P{pid}] -> attached item {card_id}:{meta.get('name','?')} "
                f"to hero {target_hero}:{engine.card_meta.get(target_hero, {}).get('name','?')}"
            )

    else:
        # magic/challenge/modifier/etc: discard after resolution for MVP
        state.discard_pile.append(card_id)

    # Resolve on_play/auto/on_activation for that card (your effects.csv uses these triggers)
    ctx: Dict[str, Any] = {
        "played_card_id": card_id,
        "played_card_type": ctype,
        "attached_to_hero": p.party[0] if (ctype == "item" and p.party) else None,
    }
    for step in engine.effects_by_card.get(card_id, []):
        trig = step.triggers()
        if "on_play" in trig or "auto" in trig or "on_activation" in trig:
            resolve_effect(step, state, engine, pid, ctx, rng, log)

    p.action_points -= cost
    for w in ctx.get("_warnings", []):
        log.append(f"[P{pid}] WARN {w}")

def action_draw(state: GameState, engine: Engine, pid: int, rng: random.Random, log: List[str]) -> bool:
    p = state.players[pid]
    if p.action_points < 1:
        return False
    if not state.draw_pile:
        return False

    cid = state.draw_pile.pop()
    p.hand.append(cid)  # IMPORTANT: add to hand first

    drawn = engine.card_meta.get(cid, {"id": cid, "type": "unknown"})
    ctx = {"drawn_card": drawn}

    log.append(f"[P{pid}] ACTION draw (cost 1) -> {cid} ({drawn.get('name','?')} / {drawn.get('type','?')})")
    p.action_points -= 1

    # Fire on_draw triggers from captured monsters
    for mid in p.captured_monsters:
        for step in engine.monster_effects.get(mid, []):
            if "on_draw" in step.triggers():
                resolve_effect(step, state, engine, pid, ctx, rng, log)

    for w in ctx.get("_warnings", []):
        log.append(f"[P{pid}] WARN {w}")

    return True

def action_activate_hero(state: GameState, engine: Engine, pid: int, rng: random.Random, log: List[str]) -> bool:
    p = state.players[pid]
    if p.action_points < 1:
        return False
    if not p.party:
        return False

    for hero_id in p.party:
        steps = engine.effects_by_card.get(hero_id, [])
        if any("on_activation" in s.triggers() for s in steps):
            log.append(f"[P{pid}] ACTION activate hero (cost 1) -> {hero_id} ({engine.card_meta.get(hero_id,{}).get('name','?')})")
            p.action_points -= 1

            ctx = {"activated_hero_id": hero_id}
            for step in steps:
                if "on_activation" in step.triggers() or "auto" in step.triggers():
                    resolve_effect(step, state, engine, pid, ctx, rng, log)

            for w in ctx.get("_warnings", []):
                log.append(f"[P{pid}] WARN {w}")
            return True

    return False

def action_attack_monster(state: GameState, engine: Engine, pid: int, monster_id: int, rng: random.Random, log: List[str]) -> bool:
    p = state.players[pid]
    if p.action_points < 2:
        return False
    if monster_id not in state.monster_row:
        return False

    rule = engine.monster_attack_rules.get(monster_id)
    if not rule or not rule.success_condition:
        log.append(f"[P{pid}] WARN monster {monster_id} has no on_attacked rule/success_condition")
        return False

    p.action_points -= 2
    log.append(f"[P{pid}] ACTION attack monster (cost 2) -> {monster_id} ({engine.card_meta.get(monster_id,{}).get('name','?')})")

    op, target = parse_simple_condition(rule.success_condition)
    final = resolve_roll_event(
        state=state,
        engine=engine,
        roller_pid=pid,
        roll_reason=f"monster:{engine.card_meta.get(monster_id,{}).get('name','?')}",
        rng=rng,
        log=log,
        goal=(op, target),
        mode="threshold",
    )
    success = goal_satisfied(final, op, target)
    outcome = "SUCCESS" if success else "FAIL"
    log.append(f"[P{pid}] monster attack roll 2d6={final} -> {outcome} (success:{rule.success_condition} fail:{rule.fail_condition})")

    ctx = {
        "attack_roll": final,
        "attack.success": success,
        "attack.fail": not success,
        "target_monster_id": monster_id,
    }

    # Resolve monster effects (success_action/fail_action rows are just EffectSteps with triggers set)
    for step in engine.monster_effects.get(monster_id, []):
        resolve_effect(step, state, engine, pid, ctx, rng, log)

    # Refill monster row if success removed the monster (often done via move_card)
    if success:
        if monster_id in state.monster_row:
            state.monster_row.remove(monster_id)
            p.captured_monsters.append(monster_id)
            log.append(f"[P{pid}] captured monster -> {monster_id}")

        if state.monster_deck:
            new_mid = state.monster_deck.pop()
            state.monster_row.append(new_mid)
            log.append(f"[SETUP] refill monster_row -> {new_mid} ({engine.card_meta.get(new_mid,{}).get('name','?')})")

    for w in ctx.get("_warnings", []):
        log.append(f"[P{pid}] WARN {w}")

    return True


# -----------------------------
# Simple policy for action selection
# -----------------------------

def choose_and_take_action(state: GameState, engine: Engine, pid: int, rng: random.Random, log: List[str]) -> bool:
    p = state.players[pid]
    if p.action_points <= 0:
        return False

    # 1) Attack monster if possible
    if p.action_points >= 2 and state.monster_row:
        return action_attack_monster(state, engine, pid, state.monster_row[0], rng, log)

    # 2) Activate hero if possible
    if action_activate_hero(state, engine, pid, rng, log):
        return True

    # 3) Play first non-modifier card if possible (avoid trying to "play" modifiers from hand)
    for cid in list(p.hand):
        ctype = str(engine.card_meta.get(cid, {}).get("type", "")).lower()
        if ctype == "modifier":
            continue
        play_card_from_hand(state, engine, pid, cid, rng, log)
        return True

    # 4) Draw
    return action_draw(state, engine, pid, rng, log)

def pick_opponent_pid(state: GameState, pid: int) -> Optional[int]:
    # MVP: next player clockwise who has at least one hero
    n = len(state.players)
    for off in range(1, n):
        op = (pid + off) % n
        if state.players[op].party:
            return op
    return None

def attacker_choose_hero_to_destroy(state: GameState, engine: Engine, victim_pid: int) -> Optional[int]:
    # MVP: attacker picks "best" hero by heuristic (most items attached)
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
    # MVP: victim sacrifices "worst" hero (fewest items)
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

# -----------------------------
# Loading CSVs
# -----------------------------

def load_effects() -> Dict[int, List[EffectStep]]:
    df = pd.read_csv(EFFECTS_CSV)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    steps_by_card: Dict[int, List[EffectStep]] = {}
    for _, r in df.iterrows():
        raw_amt = None if pd.isna(r.get("amount")) else str(r.get("amount")).strip()
        amount = None
        amount_expr = None
        if raw_amt and raw_amt.lower() != "nan":
            try:
                amount = int(float(raw_amt))
            except ValueError:
                amount_expr = raw_amt

        step = EffectStep(
            name=str(r.get("name", "")),
            card_id=int(r["card_id"]),
            step=int(r.get("step", 1)),
            trigger=str(r.get("trigger", "") or ""),
            effect_kind=str(r.get("effect_kind", "") or ""),
            source_zone=None if pd.isna(r.get("source_zone")) else str(r.get("source_zone")),
            dest_zone=None if pd.isna(r.get("dest_zone")) else str(r.get("dest_zone")),
            filter_expr=None if pd.isna(r.get("filter")) else str(r.get("filter")),
            amount=amount,
            amount_expr=amount_expr,
            requires_roll=bool(r.get("requires_roll")) if not pd.isna(r.get("requires_roll")) else False,
            roll_condition=None if pd.isna(r.get("roll_condition")) else str(r.get("roll_condition")),
            condition=None if pd.isna(r.get("condition")) else str(r.get("condition")),
            notes=None if pd.isna(r.get("notes")) else str(r.get("notes")),
        )
        steps_by_card.setdefault(step.card_id, []).append(step)

    for cid in steps_by_card:
        steps_by_card[cid].sort(key=lambda s: s.step)

    return steps_by_card

def load_card_meta() -> Dict[int, Dict[str, Any]]:
    df = pd.read_csv(CARDS_CSV)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    meta: Dict[int, Dict[str, Any]] = {}
    for _, r in df.iterrows():
        cid = int(r["id"])
        ctype = str(r.get("card_type", "unknown")).strip().lower()
        meta[cid] = {
            "id": cid,
            "name": str(r.get("name", f"card_{cid}")),
            "type": ctype,
            "subtype": str(r.get("subtype", "") if not pd.isna(r.get("subtype")) else ""),
            "action_cost": int(r.get("action_cost", 1) if not pd.isna(r.get("action_cost")) else 1),
            "copies_in_deck": int(r.get("copies_in_deck", 1) if not pd.isna(r.get("copies_in_deck")) else 1),
        }
    return meta

def load_monsters(monsters_csv: str = MONSTERS_CSV) -> Tuple[Dict[int, MonsterRule], Dict[int, List[EffectStep]]]:
    df = pd.read_csv(monsters_csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]

    attack_rule: Dict[int, MonsterRule] = {}
    effects: Dict[int, List[EffectStep]] = {}

    for _, r in df.iterrows():
        mid = int(r["card_id"])
        trigger = str(r.get("trigger", "") or "").strip()

        if trigger == "on_attacked":
            attack_rule[mid] = MonsterRule(
                monster_id=mid,
                success_condition=None if pd.isna(r.get("success_condition")) else str(r.get("success_condition")).strip(),
                fail_condition=None if pd.isna(r.get("fail_condition")) else str(r.get("fail_condition")).strip(),
                success_action=None if pd.isna(r.get("success_action")) else str(r.get("success_action")).strip(),
                fail_action=None if pd.isna(r.get("fail_action")) else str(r.get("fail_action")).strip(),
            )
            continue

        raw_amt = None if pd.isna(r.get("amount")) else str(r.get("amount")).strip()
        amount = None
        amount_expr = None
        if raw_amt and raw_amt.lower() != "nan":
            try:
                amount = int(float(raw_amt))
            except ValueError:
                amount_expr = raw_amt

        step = EffectStep(
            name=str(r.get("name", f"monster_{mid}")),
            card_id=mid,
            step=int(r.get("step", 1)),
            trigger=str(r.get("trigger", "") or ""),
            effect_kind=str(r.get("effect_kind", "") or ""),
            source_zone=None if pd.isna(r.get("source_zone")) else str(r.get("source_zone")),
            dest_zone=None if pd.isna(r.get("dest_zone")) else str(r.get("dest_zone")),
            filter_expr=None if pd.isna(r.get("filter")) else str(r.get("filter")),
            amount=amount,
            amount_expr=amount_expr,
            requires_roll=bool(r.get("requires_roll")) if not pd.isna(r.get("requires_roll")) else False,
            roll_condition=None if pd.isna(r.get("success_condition")) else str(r.get("success_condition")),
            condition=None if pd.isna(r.get("condition")) else str(r.get("condition")),
            notes=None if pd.isna(r.get("notes")) else str(r.get("notes")),
        )
        effects.setdefault(mid, []).append(step)

    for mid in effects:
        effects[mid].sort(key=lambda s: s.step)

    return attack_rule, effects

def build_modifier_options(card_meta, effects_by_card) -> Dict[int, List[int]]:
    """
    Returns: {modifier_card_id: [delta1, delta2, ...]}
    Extracts integer deltas from effect rows (amount_expr/amount/notes/filter_expr).
    """
    out: Dict[int, List[int]] = {}
    for cid, m in card_meta.items():
        if str(m.get("type", "")).lower() != "modifier":
            continue

        opts = set()
        for step in effects_by_card.get(cid, []):
            candidates: List[str] = []
            if getattr(step, "amount_expr", None):
                candidates.append(step.amount_expr)
            if getattr(step, "amount", None) is not None:
                candidates.append(str(step.amount))
            if step.notes:
                candidates.append(step.notes)
            if step.filter_expr:
                candidates.append(step.filter_expr)

            for text in candidates:
                if not text:
                    continue
                for s in re.findall(r"[-+]?\d+", str(text)):
                    try:
                        opts.add(int(s))
                    except ValueError:
                        pass

        out[cid] = sorted(opts, reverse=True) if opts else []
    return out


# -----------------------------
# Decks + setup + logging
# -----------------------------

def build_decks(card_meta: Dict[int, Dict[str, Any]]) -> Tuple[List[int], List[int], List[int]]:
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

def setup_game(state: GameState, engine: Engine, rng: random.Random, log: List[str]):
    # Party leaders
    for p in state.players:
        if state.party_leader_deck:
            leader = state.party_leader_deck.pop()
            p.party_leader = leader
            log.append(f"[P{p.pid}] party leader = {leader} ({engine.card_meta.get(leader, {}).get('name','?')})")

    # Starting hands: 3
    for p in state.players:
        for _ in range(3):
            if not state.draw_pile:
                break
            cid = state.draw_pile.pop()
            p.hand.append(cid)
            m = engine.card_meta.get(cid, {})
            log.append(f"[P{p.pid}] starting hand drew {cid} ({m.get('name','?')} / {m.get('type','?')})")

    # Reveal 3 monsters
    for i in range(3):
        if not state.monster_deck:
            break
        mid = state.monster_deck.pop()
        state.monster_row.append(mid)
        log.append(f"[SETUP] monster_row[{i}] = {mid} ({engine.card_meta.get(mid, {}).get('name','?')})")

def format_card_list(card_ids, card_meta):
    if not card_ids:
        return "—"
    return ", ".join(f"{cid}:{card_meta.get(cid, {}).get('name','?')}" for cid in card_ids)

def log_turn_state(state: GameState, engine: Engine, pid: int, log: List[str]):
    p = state.players[pid]
    log.append("")
    log.append(f"--- TURN START: Player {pid} ---")
    log.append(f"Actions: {p.action_points}")
    log.append(
        f"Party Leader: {p.party_leader}:{engine.card_meta.get(p.party_leader, {}).get('name','?')}"
        if p.party_leader is not None else "Party Leader: —"
    )
    log.append(f"Hand ({len(p.hand)}): {format_card_list(p.hand, engine.card_meta)}")
    log.append(f"Party ({len(p.party)}): {format_card_list(p.party, engine.card_meta)}")
    if p.party:
        for hid in p.party:
            items = p.hero_items.get(hid, [])
            if items:
                log.append(f"  Items on {hid}:{engine.card_meta.get(hid, {}).get('name','?')} -> {format_card_list(items, engine.card_meta)}")
    log.append(f"Captured Monsters ({len(p.captured_monsters)}): {format_card_list(p.captured_monsters, engine.card_meta)}")
    log.append(f"Monster Row ({len(state.monster_row)}): {format_card_list(state.monster_row, engine.card_meta)}")
    log.append("-" * 40)


# -----------------------------
# Run game
# -----------------------------

def build_engine() -> Engine:
    effects_by_card = load_effects()
    card_meta = load_card_meta()
    monster_attack_rules, monster_effects = load_monsters(MONSTERS_CSV)
    modifier_options_by_card_id = build_modifier_options(card_meta, effects_by_card)
    return Engine(
        effects_by_card=effects_by_card,
        card_meta=card_meta,
        monster_attack_rules=monster_attack_rules,
        monster_effects=monster_effects,
        modifier_options_by_card_id=modifier_options_by_card_id,
    )

def run_game(seed: int = 1, turns: int = 10, n_players: int = 4) -> List[str]:
    rng = random.Random(seed)
    engine = build_engine()

    draw_deck, monster_deck, leader_deck = build_decks(engine.card_meta)
    rng.shuffle(draw_deck)
    rng.shuffle(monster_deck)
    rng.shuffle(leader_deck)

    players = [PlayerState(pid=i) for i in range(n_players)]
    state = GameState(
        players=players,
        draw_pile=draw_deck,
        monster_deck=monster_deck,
        party_leader_deck=leader_deck,
    )

    log: List[str] = []
    setup_game(state, engine, rng, log)

    for t in range(turns):
        state.turn = t + 1
        pid = t % len(state.players)
        state.active_pid = pid
        p = state.players[pid]
        p.action_points = 3

        log_turn_state(state, engine, pid, log)

        safety = 30
        while p.action_points > 0 and safety > 0:
            acted = choose_and_take_action(state, engine, pid, rng, log)
            if not acted:
                break
            safety -= 1

    return log


if __name__ == "__main__":
    for line in run_game(seed=7, turns=8):
        print(line)
