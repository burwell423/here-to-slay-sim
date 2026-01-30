import json
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

import os

from .constants import CARDS_CSV, EFFECTS_JSON, MONSTERS_CSV, MONSTERS_JSON, TUNING_JSON
from .game_helpers import parse_attack_requirements
from .models import EffectStep, Engine, MonsterRule
from .tuning import compute_card_tuning_value


def load_effects() -> Dict[int, List[EffectStep]]:
    steps_by_card: Dict[int, List[EffectStep]] = {}
    with open(EFFECTS_JSON, encoding="utf-8") as f:
        rows = json.load(f)

    for r in rows:
        raw_amt = str(r.get("amount") or "").strip() or None
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
            source_zone=None if str(r.get("source_zone") or "").strip() in ("", "nan") else str(r.get("source_zone")),
            dest_zone=None if str(r.get("dest_zone") or "").strip() in ("", "nan") else str(r.get("dest_zone")),
            filter_expr=None if str(r.get("filter") or "").strip() in ("", "nan") else str(r.get("filter")),
            amount=amount,
            amount_expr=amount_expr,
            requires_roll=str(r.get("requires_roll") or "").strip().lower() in ("true", "1", "yes"),
            roll_condition=None if str(r.get("roll_condition") or "").strip() in ("", "nan") else str(r.get("roll_condition")),
            condition=None if str(r.get("condition") or "").strip() in ("", "nan") else str(r.get("condition")),
            notes=None if str(r.get("notes") or "").strip() in ("", "nan") else str(r.get("notes")),
            duration=None if str(r.get("duration") or "").strip() in ("", "nan") else str(r.get("duration")),
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


def load_monsters(monsters_json: str = MONSTERS_JSON) -> Tuple[Dict[int, MonsterRule], Dict[int, List[EffectStep]]]:
    attack_rule: Dict[int, MonsterRule] = {}
    effects: Dict[int, List[EffectStep]] = {}

    with open(monsters_json, encoding="utf-8") as f:
        payload = json.load(f)

    df = pd.read_csv(MONSTERS_CSV)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    attack_requirements: Dict[int, str] = {}
    for _, r in df.iterrows():
        raw = str(r.get("attack_requirements") or "").strip()
        if not raw or raw.lower() == "nan":
            continue
        mid = int(r["card_id"])
        attack_requirements.setdefault(mid, raw)

    for r in payload.get("attack_rules", []):
        mid = int(r["monster_id"])
        raw_attack_requirements = r.get("attack_requirements")
        if isinstance(raw_attack_requirements, str):
            raw_attack_requirements = raw_attack_requirements.strip()
            if raw_attack_requirements.lower() == "nan":
                raw_attack_requirements = ""
        parsed_requirements = parse_attack_requirements(raw_attack_requirements or attack_requirements.get(mid))
        attack_rule[mid] = MonsterRule(
            monster_id=mid,
            success_condition=None if str(r.get("success_condition") or "").strip() in ("", "nan") else str(r.get("success_condition")).strip(),
            fail_condition=None if str(r.get("fail_condition") or "").strip() in ("", "nan") else str(r.get("fail_condition")).strip(),
            success_action=None if str(r.get("success_action") or "").strip() in ("", "nan") else str(r.get("success_action")).strip(),
            fail_action=None if str(r.get("fail_action") or "").strip() in ("", "nan") else str(r.get("fail_action")).strip(),
            attack_requirements=parsed_requirements or None,
        )

    for r in payload.get("effects", []):
        mid = int(r["card_id"])
        raw_amt = str(r.get("amount") or "").strip() or None
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
            source_zone=None if str(r.get("source_zone") or "").strip() in ("", "nan") else str(r.get("source_zone")),
            dest_zone=None if str(r.get("dest_zone") or "").strip() in ("", "nan") else str(r.get("dest_zone")),
            filter_expr=None if str(r.get("filter") or "").strip() in ("", "nan") else str(r.get("filter")),
            amount=amount,
            amount_expr=amount_expr,
            requires_roll=str(r.get("requires_roll") or "").strip().lower() in ("true", "1", "yes"),
            roll_condition=None if str(r.get("roll_condition") or "").strip() in ("", "nan") else str(r.get("roll_condition")),
            condition=None if str(r.get("condition") or "").strip() in ("", "nan") else str(r.get("condition")),
            notes=None if str(r.get("notes") or "").strip() in ("", "nan") else str(r.get("notes")),
            duration=None if str(r.get("duration") or "").strip() in ("", "nan") else str(r.get("duration")),
        )
        effects.setdefault(mid, []).append(step)

    for mid in effects:
        effects[mid].sort(key=lambda s: s.step)

    return attack_rule, effects


def build_modifier_options(card_meta: Dict[int, Dict[str, Any]], effects_by_card: Dict[int, List[EffectStep]]) -> Dict[int, List[int]]:
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


def load_tuning_overrides(path: str = TUNING_JSON) -> Dict[int, float]:
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    overrides: Dict[int, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                overrides[int(key)] = float(value)
            except (TypeError, ValueError):
                continue
    return overrides


def apply_tuning_values(
    card_meta: Dict[int, Dict[str, Any]],
    effects_by_card: Dict[int, List[EffectStep]],
    monster_attack_rules: Dict[int, MonsterRule],
    monster_effects: Dict[int, List[EffectStep]],
    overrides: Dict[int, float],
) -> None:
    for cid, meta in card_meta.items():
        meta["tuning_value"] = compute_card_tuning_value(
            cid,
            meta,
            effects_by_card,
            monster_attack_rules,
            monster_effects,
            overrides=overrides,
        )


def build_engine() -> Engine:
    effects_by_card = load_effects()
    card_meta = load_card_meta()
    monster_attack_rules, monster_effects = load_monsters(MONSTERS_JSON)
    tuning_overrides = load_tuning_overrides()
    apply_tuning_values(card_meta, effects_by_card, monster_attack_rules, monster_effects, tuning_overrides)
    modifier_options_by_card_id = build_modifier_options(card_meta, effects_by_card)
    return Engine(
        effects_by_card=effects_by_card,
        card_meta=card_meta,
        monster_attack_rules=monster_attack_rules,
        monster_effects=monster_effects,
        modifier_options_by_card_id=modifier_options_by_card_id,
    )
