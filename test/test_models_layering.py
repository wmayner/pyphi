"""Architectural assertions for the pyphi.models tier.

Models are pure data: no formalism dispatch, no kernel-operation calls.
Walks the AST so lazy imports inside method bodies are also caught
(same pattern as ``test_core_layering``).
"""

from __future__ import annotations

import ast
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "pyphi" / "models"


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


def test_models_does_not_import_formalism() -> None:
    """`pyphi.models.*` is a pure-data tier: no formalism dispatch."""
    for py in MODELS.rglob("*.py"):
        imports = _imports_in(py)
        offenders = [i for i in imports if i.startswith("pyphi.formalism")]
        assert not offenders, f"{py}: imports {offenders}"


def test_models_does_not_import_kernel_ops() -> None:
    """`pyphi.models.*` does not import the repertoire-algebra kernel.

    Models are containers for results; they do not call computation.
    """
    for py in MODELS.rglob("*.py"):
        imports = _imports_in(py)
        offenders = [
            i
            for i in imports
            if i == "pyphi.core.repertoire_algebra" or i.endswith(".repertoire_algebra")
        ]
        assert not offenders, f"{py}: imports {offenders}"
