"""Tests for pyphi.core.unit — Unit dataclass."""

from __future__ import annotations

import dataclasses

import pytest


def test_unit_is_frozen() -> None:
    """Unit instances cannot be mutated."""
    from pyphi.core.unit import Unit

    u = Unit(index=0, label="A")
    with pytest.raises(dataclasses.FrozenInstanceError):
        u.index = 1  # type: ignore[misc]


def test_unit_equality_by_value() -> None:
    """Two Units with equal fields compare equal."""
    from pyphi.core.unit import Unit

    assert Unit(0, "A") == Unit(0, "A")
    assert Unit(0, "A") != Unit(1, "A")


def test_unit_is_hashable() -> None:
    """Unit instances hash to themselves consistently."""
    from pyphi.core.unit import Unit

    u = Unit(0, "A")
    assert hash(u) == hash(Unit(0, "A"))
    assert {u, Unit(0, "A")} == {u}


def test_unit_has_alphabet_size_default_2() -> None:
    """Unit defaults to alphabet_size=2."""
    from pyphi.core.unit import Unit

    u = Unit(index=0, label="A")
    assert u.alphabet_size == 2


def test_unit_alphabet_size_overridable() -> None:
    """Unit.alphabet_size accepts a non-default value."""
    from pyphi.core.unit import Unit

    u = Unit(index=0, label="A", alphabet_size=3)
    assert u.alphabet_size == 3
