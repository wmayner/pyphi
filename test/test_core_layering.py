"""Architectural assertions for the core/ package layering."""

from __future__ import annotations

import ast
from pathlib import Path

PYPHI = Path(__file__).resolve().parent.parent / "pyphi"
CORE = PYPHI / "core"


def _imports_in(path: Path) -> set[str]:
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def test_repertoire_algebra_does_not_import_formalism() -> None:
    imports = _imports_in(CORE / "repertoire_algebra.py")
    assert not any(i.startswith("pyphi.formalism") for i in imports)
    assert not any(i == ".formalism" for i in imports)


def test_system_does_not_import_formalism_at_module_level() -> None:
    """:class:`pyphi.System` is a value type — formalism dispatch lives in
    :mod:`pyphi.formalism`. Method bodies may import formalism lazily
    (e.g., the ``sia()`` convenience method), but no top-level import.
    """
    src = (PYPHI / "system.py").read_text(encoding="utf-8")
    tree = ast.parse(src, filename="pyphi/system.py")
    for node in tree.body:
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("pyphi.formalism")
        ):
            raise AssertionError(
                f"top-level import of formalism in pyphi/system.py: {node.module}"
            )


def test_core_does_not_import_formalism() -> None:
    """No file in ``pyphi/core/`` may import ``pyphi.formalism`` — at any level."""
    for py in CORE.rglob("*.py"):
        imports = _imports_in(py)
        offenders = [i for i in imports if i.startswith("pyphi.formalism")]
        assert not offenders, f"{py}: imports {offenders}"


def test_system_satisfies_system_public_interface() -> None:
    """A :class:`System` instance exposes all names in PUBLIC_SYSTEM_ATTRS."""
    from pyphi import examples
    from pyphi.protocols import PUBLIC_SYSTEM_ATTRS
    from pyphi.system import System

    cs = System(
        substrate=examples.basic_substrate(),
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )
    declared = {a for a in PUBLIC_SYSTEM_ATTRS if not a.startswith("_")}
    discovered = {a for a in dir(cs) if not a.startswith("_")}
    missing = declared - discovered
    assert not missing, f"System missing public attrs: {sorted(missing)}"
