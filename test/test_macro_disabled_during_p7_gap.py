"""Sentinel: confirms macro is correctly disabled during P7→P7b.

Deleted by P7b's first commit.
"""

import pytest


def test_macro_module_imports_successfully() -> None:
    from pyphi import macro  # noqa: F401


def test_macro_system_construction_raises() -> None:
    from pyphi.macro import MacroSystem

    with pytest.raises(NotImplementedError, match="P7b"):
        MacroSystem(None, None)
