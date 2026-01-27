#!/usr/bin/env python3
import json
import sys
from typing import List, Set, Tuple

from hts_sim.conditions import is_condition_supported, parse_roll_condition
from hts_sim.constants import EFFECTS_JSON, MONSTERS_JSON
from hts_sim.effects import SUPPORTED_EFFECT_KINDS


def _read_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _norm(x) -> str:
    if x is None:
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
    return not is_condition_supported(cond)


def roll_likely_unparseable(cond: str) -> bool:
    cond = cond.strip()
    if cond == "" or cond.lower() == "nan":
        return False
    return parse_roll_condition(cond) is None


def main() -> int:
    eff = _read_json(EFFECTS_JSON)
    mon = _read_json(MONSTERS_JSON)

    # ---------- EFFECT KIND COVERAGE ----------
    effect_kinds_effects: Set[str] = set(
        k for k in (_norm(x.get("effect_kind")).lower() for x in eff) if k
    )

    effect_kinds_monsters: Set[str] = set(
        k for k in (_norm(x.get("effect_kind")).lower() for x in mon.get("effects", [])) if k
    )

    all_effect_kinds = sorted(effect_kinds_effects | effect_kinds_monsters)
    unimplemented = sorted(set(all_effect_kinds) - set(SUPPORTED_EFFECT_KINDS))

    # ---------- CONDITION COVERAGE ----------
    bad_conditions: List[Tuple[str, str, str]] = []
    # tuple: (source, name, condition)

    # effects.csv conditions
    for r in eff:
        cond = _norm(r.get("condition"))
        if not _is_blank(cond) and condition_likely_unparseable(cond):
            bad_conditions.append(("effects", _norm(r.get("name")), cond))

    for r in mon.get("effects", []):
        cond = _norm(r.get("condition"))
        if not _is_blank(cond) and condition_likely_unparseable(cond):
            bad_conditions.append(("monsters", _norm(r.get("name")), cond))

    # ---------- ROLL CONDITIONS ----------
    bad_rolls: List[Tuple[str, str, str, str]] = []
    # tuple: (source, name, field, roll_condition)

    # effects.csv roll_condition where requires_roll true
    for r in eff:
        req_bool = str(r.get("requires_roll") or "").strip().lower() in ("true", "1", "yes")
        if req_bool:
            rc = _norm(r.get("roll_condition"))
            if roll_likely_unparseable(rc):
                bad_rolls.append(("effects", _norm(r.get("name")), "roll_condition", rc))

    for r in mon.get("attack_rules", []):
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
        tag = "OK" if k in SUPPORTED_EFFECT_KINDS else "MISSING"
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

    print(f"[4] Roll conditions likely unparseable by current parse_roll_condition() (total {len(bad_rolls)}):")
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
