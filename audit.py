#!/usr/bin/env python3
import re
import sys
from typing import Set, List, Tuple
import pandas as pd

EFFECTS_CSV = "cards - effects.csv"
MONSTERS_CSV = "cards - monsters.csv"

# Update this list to match what you currently implement in resolve_effect()
IMPLEMENTED_EFFECT_KINDS = {
    "draw_card", "draw_cards",
    "discard_card", "discard_cards",
    "move_card",
    "steal_card",
    "play_immediately",          # (your MVP stolen-card version)
    "play_drawn_immediately",    # (Malamammoth)
    "search_and_draw",
    "deny_challenge",

    # If you've added these in simulate.py, include them here:
    "destroy_hero",
    "sacrifice_hero",
}

# Current eval_condition supports:
# - blank / NaN
# - literal true/false
# - ctx boolean flags (can't know ahead of time)
# - "<var>.type==<type>"
TYPE_CMP_RE = re.compile(r"^([a-zA-Z_]\w*)\.type\s*==\s*([a-zA-Z_]\w*)$")

# parse_simple_condition supports: >=7, <=4, ==9, >8, <3 optionally with "2d6" prefix
ROLL_RE = re.compile(r"^\s*(?:2d6\s*)?(>=|<=|==|>|<)\s*(\d+)\s*$")


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    return df


def _norm(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _is_blank(x) -> bool:
    s = _norm(x)
    return s == "" or s.lower() == "nan"


def condition_likely_unparseable(cond: str) -> bool:
    """
    Heuristic: flag conditions that your current eval_condition won't parse.
    This does NOT know runtime ctx flags, so it only flags what is definitely unsupported.
    """
    cond = cond.strip()
    if cond == "" or cond.lower() == "nan":
        return False
    if cond.lower() in ("true", "false"):
        return False
    if TYPE_CMP_RE.match(cond):
        return False
    # If it contains spaces/operators or multiple dots, likely unsupported
    # Examples: "challenge.target.card.type==item" or "a && b" or "x==y"
    # (Your evaluator only supports ".type==")
    return True


def roll_likely_unparseable(cond: str) -> bool:
    cond = cond.strip()
    if cond == "" or cond.lower() == "nan":
        return False
    return ROLL_RE.match(cond) is None


def main() -> int:
    eff = _read_csv(EFFECTS_CSV)
    mon = _read_csv(MONSTERS_CSV)

    # ---------- EFFECT KIND COVERAGE ----------
    effect_kinds_effects: Set[str] = set(
        k for k in (_norm(x).lower() for x in eff.get("effect_kind", [])) if k
    )

    # monsters: only rows that are not "on_attacked" (those are rules)
    mon_non_attack = mon[mon["trigger"].fillna("").astype(str).str.strip().ne("on_attacked")]
    effect_kinds_monsters: Set[str] = set(
        k for k in (_norm(x).lower() for x in mon_non_attack.get("effect_kind", [])) if k
    )

    all_effect_kinds = sorted(effect_kinds_effects | effect_kinds_monsters)
    unimplemented = sorted(set(all_effect_kinds) - set(IMPLEMENTED_EFFECT_KINDS))

    # ---------- CONDITION COVERAGE ----------
    bad_conditions: List[Tuple[str, str, str]] = []
    # tuple: (source, name, condition)

    # effects.csv conditions
    if "condition" in eff.columns:
        for _, r in eff.iterrows():
            cond = _norm(r.get("condition"))
            if not _is_blank(cond) and condition_likely_unparseable(cond):
                bad_conditions.append(("effects", _norm(r.get("name")), cond))

    # monsters.csv conditions
    if "condition" in mon.columns:
        for _, r in mon.iterrows():
            cond = _norm(r.get("condition"))
            if not _is_blank(cond) and condition_likely_unparseable(cond):
                bad_conditions.append(("monsters", _norm(r.get("name")), cond))

    # ---------- ROLL CONDITIONS ----------
    bad_rolls: List[Tuple[str, str, str, str]] = []
    # tuple: (source, name, field, roll_condition)

    # effects.csv roll_condition where requires_roll true
    if "requires_roll" in eff.columns and "roll_condition" in eff.columns:
        for _, r in eff.iterrows():
            req = r.get("requires_roll")
            req_bool = bool(req) if not pd.isna(req) else False
            if req_bool:
                rc = _norm(r.get("roll_condition"))
                if roll_likely_unparseable(rc):
                    bad_rolls.append(("effects", _norm(r.get("name")), "roll_condition", rc))

    # monsters.csv: on_attacked success/fail conditions
    attack_rows = mon[mon["trigger"].fillna("").astype(str).str.strip().eq("on_attacked")]
    for _, r in attack_rows.iterrows():
        sc = _norm(r.get("success_condition"))
        fc = _norm(r.get("fail_condition"))
        if not _is_blank(sc) and roll_likely_unparseable(sc):
            bad_rolls.append(("monsters", _norm(r.get("name")), "success_condition", sc))
        if not _is_blank(fc) and roll_likely_unparseable(fc):
            bad_rolls.append(("monsters", _norm(r.get("name")), "fail_condition", fc))

    # ---------- PRINT REPORT ----------
    print("=== Coverage Audit ===\n")

    print(f"[1] Effect kinds found (total {len(all_effect_kinds)}):")
    for k in all_effect_kinds:
        tag = "OK" if k in IMPLEMENTED_EFFECT_KINDS else "MISSING"
        print(f"  - {k:<28} {tag}")
    print()

    print(f"[2] Unimplemented effect kinds (total {len(unimplemented)}):")
    if not unimplemented:
        print("  (none)")
    else:
        for k in unimplemented:
            print(f"  - {k}")
    print()

    print(f"[3] Conditions likely unparseable by current eval_condition() (total {len(bad_conditions)}):")
    if not bad_conditions:
        print("  (none)")
    else:
        for src, name, cond in bad_conditions[:200]:
            print(f"  - [{src}] {name}: {cond}")
        if len(bad_conditions) > 200:
            print(f"  ... ({len(bad_conditions) - 200} more)")
    print()

    print(f"[4] Roll conditions likely unparseable by current parse_simple_condition() (total {len(bad_rolls)}):")
    if not bad_rolls:
        print("  (none)")
    else:
        for src, name, field, rc in bad_rolls:
            print(f"  - [{src}] {name} {field}: {rc}")
    print()

    # Exit non-zero if something is missing (handy for CI)
    if unimplemented or bad_conditions or bad_rolls:
        print("STATUS: issues found")
        return 2
    print("STATUS: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
