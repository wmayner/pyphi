"""Architectural assertions for the new core/ package."""

from __future__ import annotations

import ast
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "pyphi" / "core"


def _imports_in(path: Path) -> set[str]:
    src = path.read_text()
    tree = ast.parse(src, filename=str(path))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
    return out


def test_causal_model_does_not_import_repertoire_algebra() -> None:
    imports = _imports_in(CORE / "causal_model.py")
    assert not any("repertoire_algebra" in i for i in imports)


def test_repertoire_algebra_does_not_import_formalism() -> None:
    imports = _imports_in(CORE / "repertoire_algebra.py")
    assert not any(i.startswith("pyphi.formalism") for i in imports)
    assert not any(i == ".formalism" for i in imports)


def test_core_does_not_import_subsystem_module_top_level() -> None:
    """The core package may use the legacy subsystem inside function bodies
    during the worktree (the ``_legacy_subsystem`` helper), but no module in
    core/ should import it at the top level after the worktree is healthy.
    """
    for py in CORE.rglob("*.py"):
        src = py.read_text()
        tree = ast.parse(src, filename=str(py))
        for node in tree.body:
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and "subsystem" in node.module
            ):
                raise AssertionError(
                    f"{py}: top-level import of subsystem ({node.module})"
                )


def test_candidate_system_satisfies_subsystem_public_interface() -> None:
    """CandidateSystem instance exposes all names in PUBLIC_SUBSYSTEM_ATTRS.

    We instantiate to discover dataclass fields (auto-generated __init__),
    then dir() the instance to pick up methods, properties, and fields.
    """
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel
    from pyphi.protocols import PUBLIC_SUBSYSTEM_ATTRS

    cm = CausalModel.from_network(examples.basic_network())
    cs = CandidateSystem(causal_model=cm, state=(1, 0, 0), node_indices=(0, 1, 2))
    declared = {a for a in PUBLIC_SUBSYSTEM_ATTRS if not a.startswith("_")}
    discovered = {a for a in dir(cs) if not a.startswith("_")}
    missing = declared - discovered
    assert not missing, f"CandidateSystem missing public attrs: {sorted(missing)}"
