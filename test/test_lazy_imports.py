"""Lazy-import discipline (P6a).

These tests pin the deferred-import contract that keeps free-threaded
CPython safe to use with PyPhi: heavy C extensions whose modules are not
free-thread-safe (currently graphillion's ``_graphillion`` lacks
``PyMod_GIL_NOT_USED``) must not load at ``import pyphi`` time. They load
only when the user explicitly invokes the relations / set-family code
that needs them. Workers that compute mechanism-level phi or unfolding
without relations stay no-GIL safe.

When P6b lands the OxiDD-based ZDD backend, graphillion will become
optional; until then this test enforces the deferral.
"""

from __future__ import annotations

import subprocess
import sys


def _check_module_after_import(check_module: str) -> bool:
    """Return ``True`` iff ``check_module`` is loaded after ``import pyphi``.

    Spawned in a subprocess so the parent process's already-imported
    modules don't pollute the result.
    """
    code = (
        "import sys\n"
        "import pyphi  # noqa: F401\n"
        f"print(int({check_module!r} in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return result.stdout.strip().endswith("1")


def test_graphillion_not_loaded_at_pyphi_import() -> None:
    """``import pyphi`` must not eagerly load graphillion.

    graphillion's ``_graphillion`` C extension does not declare
    ``PyMod_GIL_NOT_USED``, so loading it under free-threaded Python
    re-enables the GIL process-wide. Workers that don't compute relations
    must not pay this cost. The deferred-import pattern in
    ``pyphi.relations`` and ``pyphi.combinatorics`` is what enforces this.
    """
    assert not _check_module_after_import("graphillion"), (
        "graphillion was loaded eagerly by `import pyphi`; this re-enables "
        "the GIL on free-threaded Python. Make the import deferred — see "
        "pyphi/combinatorics.py and pyphi/relations.py for the pattern."
    )
