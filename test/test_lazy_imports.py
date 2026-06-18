"""Lazy-import discipline.

These tests pin the deferred-import contract that keeps free-threaded
CPython safe to use with PyPhi: optional heavy modules must not load at
``import pyphi`` time, only when the user explicitly invokes the code that
needs them.
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


def test_xarray_backend_not_loaded_at_pyphi_import():
    """The xarray FactoredTPM backend must load only when requested.

    xarray is an optional extra; an eager import would make ``import
    pyphi`` fail entirely in environments without it.
    """
    assert not _check_module_after_import("pyphi.core.tpm._factored_backends_xarray")
