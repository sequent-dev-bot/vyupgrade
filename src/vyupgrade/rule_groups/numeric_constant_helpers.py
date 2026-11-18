from __future__ import annotations

import ast
import re
from collections.abc import Callable

from ..analysis import is_integer_type, normalize_type
# Keep the existing helper imports available to rule-only callers.
from ..constant_evaluation import (
    eval_integer_constant_expr,
    integer_constant_values as integer_constant_values,
)
from ..source import split_top_level_args
from .numeric_types import integer_value_fits_type


def constant_range_iteration_bound(args: str, values: dict[str, int]) -> int | None:
    parts = split_top_level_args(args)
    if parts is None:
        return None
    if len(parts) == 1:
        stop = eval_integer_constant_expr(parts[0], values)
        if stop is None or stop < 0:
            return None
        return stop
    if len(parts) != 2:
        return None
    start = eval_integer_constant_expr(parts[0], values)
    stop = eval_integer_constant_expr(parts[1], values)
    if start is None or stop is None or stop < start:
        return None
    return stop - start


def retype_integer_operands(
    expr: str,
    target: str,
    types: dict[str, str],
    values: dict[str, int],
    cast_variable: Callable[[str], bool],
) -> str | None:
    """``expr`` with its integer operands in ``target``'s type, or ``None`` if unsafe.

    Before 0.3.2 the compiler folded constants into untyped literals before typing
    them. A constant of another type is converted where every intermediate value of
    its arithmetic fits ``target``; otherwise the arithmetic of literals and
    constants becomes its folded value, which must fit. A variable of another type is converted only when
    ``cast_variable`` accepts its type. Calls, attributes and subscripts keep their
    own types and are left alone.
    """
    text = f"({expr})"
    try:
        tree = ast.parse(text, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    rows = [0, *(match.end() for match in re.finditer("\n", text))]
    edits: list[tuple[int, int, str]] = []

    def offset(row: int, column: int) -> int:
        start = rows[row - 1]
        return start + len(text[start:].encode("utf-8")[:column].decode("utf-8"))

    def span(node: ast.expr) -> tuple[int, int]:
        assert node.end_lineno is not None and node.end_col_offset is not None
        return offset(node.lineno, node.col_offset), offset(node.end_lineno, node.end_col_offset)

    def mismatched(name: str) -> bool:
        declared = normalize_type(types.get(name) or "")
        return is_integer_type(declared) and declared != target

    def constant_only(node: ast.expr) -> bool:
        if isinstance(node, ast.Constant):
            return type(node.value) is int
        if isinstance(node, ast.Name):
            return node.id in values
        if isinstance(node, ast.UnaryOp):
            return constant_only(node.operand)
        if isinstance(node, ast.BinOp):
            return constant_only(node.left) and constant_only(node.right)
        return False

    def fits(node: ast.expr) -> bool:
        start, end = span(node)
        value = eval_integer_constant_expr(text[start:end], values)
        return value is not None and integer_value_fits_type(value, target)

    def visit(node: ast.expr) -> bool:
        if (
            isinstance(node, (ast.BinOp, ast.UnaryOp))
            and constant_only(node)
            and any(isinstance(n, ast.Name) and mismatched(n.id) for n in ast.walk(node))
            and not all(fits(n) for n in ast.walk(node) if isinstance(n, ast.expr))
        ):
            # Computing it in the target type would overflow where folding did not.
            start, end = span(node)
            value = eval_integer_constant_expr(text[start:end], values)
            if value is None or not integer_value_fits_type(value, target):
                return False
            edits.append((start, end, f"({value})" if value < 0 else str(value)))
            return True
        if isinstance(node, ast.Name):
            if not mismatched(node.id):
                return True
            if node.id in values:
                if not integer_value_fits_type(values[node.id], target):
                    return False
            elif not cast_variable(normalize_type(types[node.id])):
                return True
            start, end = span(node)
            edits.append((start, end, f"convert({node.id}, {target})"))
            return True
        if isinstance(node, ast.BinOp):
            return visit(node.left) and visit(node.right)
        if isinstance(node, ast.UnaryOp):
            return visit(node.operand)
        return True

    if not visit(tree.body):
        return None
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text[1:-1]
