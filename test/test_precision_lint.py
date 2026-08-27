"""Forbid raw comparisons of φ/Φ/α magnitudes outside the tolerant layer.

A comparison, min/max/sorted call, or isclose/allclose call whose operand
mentions an attribute named like a φ magnitude — directly (``x.phi``) or
wrapped (``float(x.phi)``, ``round(x.phi, 4)``, ``-x.phi``, ``x.phi + y``)
— must route through pyphi.numerics or pyphi.resolve_ties, or carry an
explicit waiver comment (``# numerics: exact — <reason>``) on its line or
the line above.

Known blind spot: only attribute-named operands (e.g. ``x.phi``) are
matched. A φ magnitude first bound to a local name — ``phi = float(x.phi)``
and then compared as a bare ``phi`` — is invisible to this lint, since the
comparison node no longer carries a matching ``ast.Attribute``. The guard
is a backstop against the common attribute-comparison mistake, not a proof
that every raw φ comparison has been routed through the tolerant layer.
"""

import ast
import textwrap
from pathlib import Path

import pytest

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
# isclose/allclose carry their own default tolerances, which are not
# PyPhi's configured precision — comparing φ through them bypasses the
# tolerant layer just as surely as a raw ``==`` does.
ISCLOSE_NAMES = {"isclose", "allclose"}
PYPHI = Path(__file__).parent.parent / "pyphi"

# Existing pyphi/ sites the widened matcher flags that are correct as
# written. Keyed by (path relative to the repo root, exact stripped source
# line) so entries survive unrelated line drift; if the line itself is
# edited, the waiver lapses and the site is re-examined.
ALLOWLIST: dict[tuple[str, str], str] = {
    (
        "pyphi/models/distinction.py",
        "or float(self.cause.phi) < float(self.effect.phi)",
    ): (
        "tolerant ≤ composed from numerics.eq (previous line of the same "
        "BoolOp) plus a strict <; the eq guard makes the raw < tie-safe"
    ),
}


def _mentions_phi_attr(node: ast.AST) -> bool:
    """Whether a φ-magnitude attribute appears anywhere in ``node``.

    Identity-check subtrees (``x.phi is None``) are pruned: they test
    presence, never magnitude, and the model layer legitimately compares
    their boolean results with ``==``/``!=``.
    """
    if isinstance(node, ast.Compare) and all(
        isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops
    ):
        return False
    if isinstance(node, ast.Attribute) and node.attr in PHI_ATTRS:
        return True
    return any(_mentions_phi_attr(child) for child in ast.iter_child_nodes(node))


def _waived(lines: list[str], lineno: int) -> bool:
    for candidate in (lineno - 1, lineno - 2):  # the line and the line above
        if 0 <= candidate < len(lines) and "# numerics: exact" in lines[candidate]:
            return True
    return False


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _violations(path: Path, relpath: str) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    found = []

    def flag(node, what):
        if _waived(lines, node.lineno):
            return
        stripped = lines[node.lineno - 1].strip()
        if (relpath, stripped) in ALLOWLIST:
            return
        found.append(f"{relpath}:{node.lineno} {what}")

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            # ``is`` / ``is not`` (e.g. ``phi is None``) are identity checks,
            # never magnitude comparisons, so they cannot be the tolerant-
            # comparison bug class this lint guards.
            if all(isinstance(op, (ast.Is, ast.IsNot)) for op in node.ops):
                continue
            operands = [node.left, *node.comparators]
            if any(_mentions_phi_attr(op) for op in operands):
                flag(node, "raw comparison of a φ attribute")
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            if name in {"min", "max", "sorted"} and (
                any(_mentions_phi_attr(arg) for arg in node.args)
                or any(
                    kw.arg == "key" and _mentions_phi_attr(kw.value)
                    for kw in node.keywords
                )
            ):
                flag(node, f"raw {name}() over a φ attribute")
            elif name in ISCLOSE_NAMES and (
                any(_mentions_phi_attr(arg) for arg in node.args)
                or any(_mentions_phi_attr(kw.value) for kw in node.keywords)
            ):
                flag(node, f"{name}() over a φ attribute bypasses pyphi.numerics")
    return found


def test_no_raw_phi_comparisons():
    violations = []
    for path in sorted(PYPHI.rglob("*.py")):
        if path.name in ALLOWED_MODULES:
            continue
        violations.extend(_violations(path, str(path.relative_to(PYPHI.parent))))
    assert not violations, (
        "Raw φ/α comparisons found. Route through pyphi.numerics or "
        "pyphi.resolve_ties, or add '# numerics: exact — <reason>':\n"
        + "\n".join(violations)
    )


# ============== Matcher self-tests ==============
#
# The matcher is itself a gate; these pin the spellings it must catch and
# the ones it must ignore, so a future simplification cannot silently
# narrow its coverage.

FLAGGED_SNIPPETS = [
    "a.phi == b.phi",
    "float(a.phi) == float(b.phi)",
    "round(a.phi, 4) == b",
    "a.phi > 0",
    "a.phi < b.phi <= c.phi",
    "-a.phi < b",
    "a.phi + x > y",
    "abs(a.phi - b.phi) < eps",
    "a.alpha != b.alpha",
    "min(a.phi, b.phi)",
    "max(x.big_phi for x in xs)",
    "sorted(xs, key=lambda x: x.phi)",
    "math.isclose(a.phi, b.phi)",
    "np.isclose(a.phi, b.phi)",
    "np.allclose(a.sum_phi, b.sum_phi)",
    "isclose(a.phi, 0.5)",
    "np.isclose(x, y, atol=a.phi)",
]

CLEAN_SNIPPETS = [
    "a.phi is None",
    "a.phi is not None",
    "(a.phi is None) != (b.phi is None)",
    "numerics.eq(a.phi, b.phi)",
    "a.nodes == b.nodes",
    "min(a.count, b.count)",
    "phi == other_phi",  # documented blind spot: bare locals are invisible
]


def _lint_snippet(snippet: str, tmp_path: Path) -> list[str]:
    path = tmp_path / "sample.py"
    path.write_text(textwrap.dedent(snippet) + "\n", encoding="utf-8")
    return _violations(path, "sample.py")


@pytest.mark.parametrize("snippet", FLAGGED_SNIPPETS)
def test_matcher_catches_spelling(snippet, tmp_path):
    assert _lint_snippet(snippet, tmp_path), f"matcher missed: {snippet}"


@pytest.mark.parametrize("snippet", CLEAN_SNIPPETS)
def test_matcher_ignores_non_magnitude_uses(snippet, tmp_path):
    assert not _lint_snippet(snippet, tmp_path), f"matcher over-flagged: {snippet}"


def test_waiver_comment_suppresses_flag(tmp_path):
    waived = "# numerics: exact — test waiver\na.phi == b.phi"
    assert not _lint_snippet(waived, tmp_path)


def test_allowlisted_lines_are_still_present_in_source():
    """Every allowlist entry must match a real line, so stale entries are
    removed rather than silently shielding future code."""
    for relpath, stripped in ALLOWLIST:
        source_lines = [
            line.strip() for line in (PYPHI.parent / relpath).read_text().splitlines()
        ]
        assert stripped in source_lines, (
            f"allowlist entry no longer matches any line in {relpath}: {stripped!r}"
        )
