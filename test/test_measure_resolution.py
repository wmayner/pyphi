"""Resolver helpers: name → typed measure callable."""

from __future__ import annotations

import pytest

from pyphi.measures.distribution import composite_measures
from pyphi.measures.distribution import distribution_measures
from pyphi.measures.distribution import resolve_distribution_measure
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from pyphi.measures.distribution import state_aware_measures
from pyphi.measures.distribution import stateful_distribution_measures


def test_resolve_mechanism_measure_accepts_state_aware() -> None:
    metric = resolve_mechanism_measure("INTRINSIC_DIFFERENTIATION")
    assert metric is state_aware_measures["INTRINSIC_DIFFERENTIATION"]


def test_resolve_mechanism_measure_accepts_stateful_distribution() -> None:
    metric = resolve_mechanism_measure("IIT_4.0_SMALL_PHI")
    assert metric is stateful_distribution_measures["IIT_4.0_SMALL_PHI"]


def test_resolve_mechanism_measure_accepts_composite() -> None:
    metric = resolve_mechanism_measure("GENERALIZED_INTRINSIC_DIFFERENCE")
    assert metric is composite_measures["GENERALIZED_INTRINSIC_DIFFERENCE"]


def test_resolve_system_measure_returns_composite() -> None:
    metric = resolve_system_measure("INTRINSIC_INFORMATION")
    assert metric is composite_measures["INTRINSIC_INFORMATION"]


def test_resolve_system_measure_rejects_state_aware() -> None:
    with pytest.raises(ValueError, match="Unknown system measure"):
        resolve_system_measure("INTRINSIC_DIFFERENTIATION")


def test_resolve_distribution_measure_returns_distribution_metric() -> None:
    metric = resolve_distribution_measure("EMD")
    assert metric is distribution_measures["EMD"]


def test_unknown_metric_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError, match="Available:"):
        resolve_mechanism_measure("NONSENSE_METRIC")
