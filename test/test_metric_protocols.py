"""Pin every registered metric's Protocol membership.

Catches signature drift: if a metric's function signature changes such
that it no longer matches its declared Protocol, the test fails.

Protocol membership is checked via ``inspect.signature`` rather than
``isinstance``, because the registered metrics are plain functions and
plain functions do not carry the ``name``/``asymmetric`` class-level
attributes that the runtime-checkable Protocols declare.  The
structural intent — that each metric satisfies exactly one Protocol
shape — is captured by inspecting parameter names.

Protocol shapes:
  DistributionMetric          : exactly two required params (p, q)
  StateAwareMetric            : exactly two required params (p, state) — no q
  CompositeMetric             : first three params name forward, partitioned,
                                selectivity repertoires
  StatefulDistributionMetric  : exactly three required params (p, q, state)
"""

from __future__ import annotations

import inspect

import pytest

from pyphi.metrics import distribution

# ---------------------------------------------------------------------------
# Classification lists
# ---------------------------------------------------------------------------

DISTRIBUTION_METRICS = [
    "EMD",
    "L1",
    "ENTROPY_DIFFERENCE",
    "PSQ2",
    "MP2Q",
    "KLD",
    "ID",
    "AID",
    "KLM",
    "BLD",
]

STATE_AWARE_METRICS = [
    "INTRINSIC_DIFFERENTIATION",
]

COMPOSITE_METRICS = [
    "GENERALIZED_INTRINSIC_DIFFERENCE",
    "INTRINSIC_SPECIFICATION",
    "INTRINSIC_INFORMATION",
]

STATEFUL_DISTRIBUTION_METRICS = [
    "IIT_4.0_SMALL_PHI",
    "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
    "APMI",
]

# ---------------------------------------------------------------------------
# Helpers for structural signature checking
# ---------------------------------------------------------------------------


def _required_params(metric) -> list[str]:
    """Return the names of required positional parameters (no default)."""
    sig = inspect.signature(metric)
    return [
        p.name
        for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
        and p.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]


def _all_params(metric) -> list[str]:
    """Return all parameter names (required and optional)."""
    sig = inspect.signature(metric)
    return list(sig.parameters.keys())


def _satisfies_distribution_metric(metric) -> bool:
    """Check (p, q) shape: required params are exactly p and q."""
    req = _required_params(metric)
    return req == ["p", "q"]


def _satisfies_state_aware_metric(metric) -> bool:
    """Check (p, state) shape: required params are exactly p and state."""
    req = _required_params(metric)
    return req == ["p", "state"]


def _satisfies_composite_metric(metric) -> bool:
    """Check (forward, partitioned, selectivity, ...) shape."""
    params = _all_params(metric)
    if len(params) < 3:
        return False
    return (
        "forward" in params[0]
        and "partitioned" in params[1]
        and "selectivity" in params[2]
    )


def _satisfies_stateful_distribution_metric(metric) -> bool:
    """Check (p, q, state) shape: required params are exactly p, q, state."""
    req = _required_params(metric)
    return req == ["p", "q", "state"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DISTRIBUTION_METRICS)
def test_distribution_metric_satisfies_protocol(name: str) -> None:
    """DistributionMetric: required params are exactly (p, q)."""
    metric = distribution.measures[name]
    assert _satisfies_distribution_metric(metric), (
        f"{name!r} does not satisfy DistributionMetric Protocol — "
        f"required params: {_required_params(metric)!r}"
    )
    # Confirm it doesn't satisfy any other Protocol shape
    assert not _satisfies_state_aware_metric(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMetric"
    )
    assert not _satisfies_composite_metric(metric), (
        f"{name!r} unexpectedly satisfies CompositeMetric"
    )
    assert not _satisfies_stateful_distribution_metric(metric), (
        f"{name!r} unexpectedly satisfies StatefulDistributionMetric"
    )


@pytest.mark.parametrize("name", STATE_AWARE_METRICS)
def test_state_aware_metric_satisfies_protocol(name: str) -> None:
    """StateAwareMetric: required params are exactly (p, state)."""
    metric = distribution.measures[name]
    assert _satisfies_state_aware_metric(metric), (
        f"{name!r} does not satisfy StateAwareMetric Protocol — "
        f"required params: {_required_params(metric)!r}"
    )
    assert not _satisfies_distribution_metric(metric), (
        f"{name!r} unexpectedly satisfies DistributionMetric"
    )
    assert not _satisfies_composite_metric(metric), (
        f"{name!r} unexpectedly satisfies CompositeMetric"
    )


@pytest.mark.parametrize("name", COMPOSITE_METRICS)
def test_composite_metric_satisfies_protocol(name: str) -> None:
    """CompositeMetric: params include forward, partitioned, selectivity."""
    metric = distribution.measures[name]
    assert _satisfies_composite_metric(metric), (
        f"{name!r} does not satisfy CompositeMetric Protocol — "
        f"all params: {_all_params(metric)!r}"
    )
    assert not _satisfies_distribution_metric(metric), (
        f"{name!r} unexpectedly satisfies DistributionMetric"
    )
    assert not _satisfies_state_aware_metric(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMetric"
    )


@pytest.mark.parametrize("name", STATEFUL_DISTRIBUTION_METRICS)
def test_stateful_distribution_metric_satisfies_protocol(name: str) -> None:
    """StatefulDistributionMetric: required params are exactly (p, q, state)."""
    metric = distribution.measures[name]
    assert _satisfies_stateful_distribution_metric(metric), (
        f"{name!r} does not satisfy StatefulDistributionMetric Protocol — "
        f"required params: {_required_params(metric)!r}"
    )
    assert not _satisfies_state_aware_metric(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMetric"
    )
    assert not _satisfies_composite_metric(metric), (
        f"{name!r} unexpectedly satisfies CompositeMetric"
    )
