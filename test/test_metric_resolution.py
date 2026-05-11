"""Resolver helpers: name → typed metric callable."""

from __future__ import annotations

import pytest

from pyphi.metrics.distribution import composite_metrics
from pyphi.metrics.distribution import distribution_metrics
from pyphi.metrics.distribution import resolve_alpha_measure
from pyphi.metrics.distribution import resolve_mechanism_metric
from pyphi.metrics.distribution import resolve_system_metric
from pyphi.metrics.distribution import state_aware_metrics
from pyphi.metrics.distribution import stateful_distribution_metrics


def test_resolve_mechanism_metric_accepts_state_aware() -> None:
    metric = resolve_mechanism_metric("INTRINSIC_DIFFERENTIATION")
    assert metric is state_aware_metrics["INTRINSIC_DIFFERENTIATION"]


def test_resolve_mechanism_metric_accepts_stateful_distribution() -> None:
    metric = resolve_mechanism_metric("IIT_4.0_SMALL_PHI")
    assert metric is stateful_distribution_metrics["IIT_4.0_SMALL_PHI"]


def test_resolve_mechanism_metric_accepts_composite() -> None:
    metric = resolve_mechanism_metric("GENERALIZED_INTRINSIC_DIFFERENCE")
    assert metric is composite_metrics["GENERALIZED_INTRINSIC_DIFFERENCE"]


def test_resolve_system_metric_returns_composite() -> None:
    metric = resolve_system_metric("INTRINSIC_INFORMATION")
    assert metric is composite_metrics["INTRINSIC_INFORMATION"]


def test_resolve_system_metric_rejects_state_aware() -> None:
    with pytest.raises(ValueError, match="Unknown system metric"):
        resolve_system_metric("INTRINSIC_DIFFERENTIATION")


def test_resolve_alpha_measure_returns_distribution_metric() -> None:
    metric = resolve_alpha_measure("EMD")
    assert metric is distribution_metrics["EMD"]


def test_unknown_metric_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError, match="Available:"):
        resolve_mechanism_metric("NONSENSE_METRIC")
