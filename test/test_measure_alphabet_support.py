"""Tests for measure alphabet-support metadata."""

from __future__ import annotations

import pytest

from pyphi.measures.distribution import actual_causation_measures
from pyphi.measures.distribution import composite_measures
from pyphi.measures.distribution import distribution_measures
from pyphi.measures.distribution import state_aware_measures
from pyphi.measures.distribution import stateful_distribution_measures

_ALL_REGISTRIES = [
    distribution_measures,
    state_aware_measures,
    composite_measures,
    stateful_distribution_measures,
    actual_causation_measures,
]


@pytest.mark.parametrize("registry", _ALL_REGISTRIES)
def test_all_measures_declare_supports_alphabet(registry) -> None:
    """Every registered measure has a callable supports_alphabet attribute."""
    for name in registry.all():
        measure = registry[name]
        assert hasattr(measure, "supports_alphabet"), (
            f"Measure {name!r} in {registry!r} is missing supports_alphabet"
        )
        assert callable(measure.supports_alphabet), (
            f"Measure {name!r}.supports_alphabet is not callable"
        )


def test_emd_supports_alphabet_binary_only() -> None:
    """EMD supports_alphabet returns True for binary-only, False when k>2."""
    emd = distribution_measures["EMD"]
    assert emd.supports_alphabet((2, 2, 2)) is True
    assert emd.supports_alphabet((2, 3, 2)) is False
    assert emd.supports_alphabet((3, 3)) is False


def test_l1_supports_alphabet_any() -> None:
    """L1 supports_alphabet returns True for any alphabet."""
    l1 = distribution_measures["L1"]
    assert l1.supports_alphabet((2, 2)) is True
    assert l1.supports_alphabet((3, 4, 5)) is True


def test_aid_supports_alphabet_any() -> None:
    """AID supports_alphabet returns True for any alphabet."""
    aid = distribution_measures["AID"]
    assert aid.supports_alphabet((2, 2)) is True
    assert aid.supports_alphabet((3, 4, 5)) is True


def test_generalized_intrinsic_difference_supports_alphabet_any() -> None:
    """GENERALIZED_INTRINSIC_DIFFERENCE supports_alphabet returns True for any alphabet."""
    gid = composite_measures["GENERALIZED_INTRINSIC_DIFFERENCE"]
    assert gid.supports_alphabet((2, 2)) is True
    assert gid.supports_alphabet((3, 4, 5)) is True


def test_intrinsic_information_supports_alphabet_any() -> None:
    """INTRINSIC_INFORMATION supports_alphabet returns True for any alphabet."""
    ii = composite_measures["INTRINSIC_INFORMATION"]
    assert ii.supports_alphabet((2, 2)) is True
    assert ii.supports_alphabet((3, 4, 5)) is True
