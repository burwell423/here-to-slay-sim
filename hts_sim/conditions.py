import ast
import re
from typing import Any, Dict, Optional, Tuple

ROLL_RE = re.compile(r"^\s*(?:2d6\s*)?(>=|<=|==|>|<)\s*(\d+)\s*$")
SUPPORTED_BOOL_NAMES = {"true": "True", "false": "False"}


def roll_2d6(rng: "random.Random") -> int:
    return rng.randint(1, 6) + rng.randint(1, 6)


def roll_2d6_detail(rng: "random.Random") -> Tuple[int, int, int]:
    first = rng.randint(1, 6)
    second = rng.randint(1, 6)
    return first, second, first + second


def check_roll(roll_value: int, cond: str) -> bool:
    """
    cond examples: '>=5', '<=7', '==9'
    """
    m = re.match(r"^\s*(>=|<=|==|>|<)\s*(\d+)\s*$", str(cond))
    if not m:
        raise ValueError(f"Unparseable roll_condition: {cond}")
    op, num_s = m.group(1), m.group(2)
    target = int(num_s)
    if op == ">=":
        return roll_value >= target
    if op == "<=":
        return roll_value <= target
    if op == "==":
        return roll_value == target
    if op == ">":
        return roll_value > target
    if op == "<":
        return roll_value < target
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

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute) and node.func.attr == "contains":
            base = _eval_condition_node(node.func.value, ctx)
            if base is None or len(node.args) != 1:
                return False
            needle = _eval_condition_node(node.args[0], ctx)
            if isinstance(needle, str):
                needle = needle.strip().lower()
            if isinstance(base, dict):
                return needle in base
            if isinstance(base, (list, set, tuple)):
                normalized = {
                    item.strip().lower() if isinstance(item, str) else item for item in base
                }
                return needle in normalized
            if isinstance(base, str):
                return str(needle) in base
            return False
        raise ValueError("Unsupported function call")

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
