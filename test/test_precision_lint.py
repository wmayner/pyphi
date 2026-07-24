"""Forbid raw comparisons of φ/Φ/α magnitudes outside the tolerant layer.

A comparison operator or min/max/sorted call whose operand is an
attribute named like a φ magnitude must route through pyphi.numerics or
pyphi.resolve_ties, or carry an explicit waiver comment
(``# numerics: exact — <reason>``) on its line or the line above.

Known blind spot: only attribute-named operands (e.g. ``x.phi``) are
matched. A φ magnitude first bound to a local name — ``phi = float(x.phi)``
and then compared as a bare ``phi`` — is invisible to this lint, since the
comparison node no longer carries a matching ``ast.Attribute``. The guard
is a backstop against the common attribute-comparison mistake, not a proof
that every raw φ comparison has been routed through the tolerant layer.
"""

import ast
from pathlib import Path

PHI_ATTRS = {
    "phi",
    "alpha",
    "big_phi",
    "normalized_phi",
    "signed_phi",
    "signed_normalized_phi",
    "sum_phi",
    "intrinsic_information",
}
ALLOWED_MODULES = {"numerics.py", "resolve_ties.py"}
PYPHI = Path(__file__).parent.parent / "pyphi"


def _mentions_phi_attr(node: ast.AST) -> bool:
    return any(
        isinstance(sub, ast.Attribute) and sub.attr in PHI_ATTRS
        for sub in ast.walk(node)
    )


def _waived(lines: list[str], lineno: int) -> bool:
    for candidate in (lineno - 1, lineno - 2):  # the line and the line above
        if 0 <= candidate < len(lines) and "# numerics: exact" in lines[candidate]:
            return True
    return False


def _violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    found = []

    def flag(node, what):
        if not _waived(lines, node.lineno):
            found.append(f"{path.relative_to(PYPHI.parent)}:{node.lineno} {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # ``is`` / ``is not`` (e.g. ``phi is None``) are identity checks,
            # never magnitude comparisons, so they cannot be the tolerant-
            # comparison bug class this lint guards.
            if all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                continue
            operands = [node.left, *node.comparators]
            if any(
                isinstance(op, ast.Attribute) and op.attr in PHI_ATTRS for op in operands
            ):
                flag(node, "raw comparison of a φ attribute")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"min", "max", "sorted"} and (
                any(_mentions_phi_attr(arg) for arg in node.args)
                or any(
                    kw.arg == "key" and _mentions_phi_attr(kw.value)
                    for kw in node.keywords
                )
            ):
                flag(node, f"raw {node.func.id}() over a φ attribute")
    return found


def test_no_raw_phi_comparisons():
    violations = []
    for path in sorted(PYPHI.rglob("*.py")):
        if path.name in ALLOWED_MODULES:
            continue
        violations.extend(_violations(path))
    assert not violations, (
        "Raw φ/α comparisons found. Route through pyphi.numerics or "
        "pyphi.resolve_ties, or add '# numerics: exact — <reason>':\n"
        + "\n".join(violations)
    )
