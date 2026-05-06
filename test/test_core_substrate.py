"""Tests for pyphi.core.substrate — Substrate dataclass."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


def test_substrate_is_frozen() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit

    units = (Unit(0, "A"), Unit(1, "B"))
    cm = np.zeros((2, 2), dtype=int)
    s = Substrate(units=units, connectivity_matrix=cm)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.units = ()  # type: ignore[misc]


def test_substrate_n_units() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit

    units = (Unit(0, "A"), Unit(1, "B"), Unit(2, "C"))
    cm = np.zeros((3, 3), dtype=int)
    s = Substrate(units=units, connectivity_matrix=cm)
    assert s.n_units == 3


def test_substrate_node_labels() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit

    units = (Unit(0, "A"), Unit(1, "B"))
    cm = np.zeros((2, 2), dtype=int)
    s = Substrate(units=units, connectivity_matrix=cm)
    assert tuple(s.node_labels) == ("A", "B")


def test_substrate_equality_includes_cm() -> None:
    from pyphi.core.substrate import Substrate
    from pyphi.core.unit import Unit

    units = (Unit(0, "A"),)
    cm1 = np.array([[0]], dtype=int)
    cm2 = np.array([[1]], dtype=int)
    assert Substrate(units, cm1) == Substrate(units, cm1)
    assert Substrate(units, cm1) != Substrate(units, cm2)
