"""Pin every registered metric's Protocol membership.

Catches signature drift: if a metric's function signature changes such
that it no longer matches its declared Protocol, the test fails.

Protocol membership is checked via ``inspect.signature`` rather than
``isinstance``, because the registered metrics are plain functions and
plain functions do not carry the ``name``/``asymmetric`` class-level
attributes that the runtime-checkable Protocols declare.  The
structural intent — that each metric satisfies exactly one Protocol
shape — is captured by the shared ``satisfies_*`` predicates in
:mod:`pyphi.metrics.protocols`, which the typed registries also use at
registration time.

Protocol shapes:
  DistributionMetric          : exactly two required params (p, q)
  StateAwareMetric            : exactly two required params (p, state) — no q
  CompositeMetric             : first three params name forward, partitioned,
                                selectivity repertoires
  StatefulDistributionMetric  : exactly three required params (p, q, state)
"""

from __future__ import annotations

import pytest

from pyphi.metrics import distribution
from pyphi.metrics.protocols import satisfies_composite_metric
from pyphi.metrics.protocols import satisfies_distribution_metric
from pyphi.metrics.protocols import satisfies_state_aware_metric
from pyphi.metrics.protocols import satisfies_stateful_distribution_metric

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
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", DISTRIBUTION_METRICS)
def test_distribution_metric_satisfies_protocol(name: str) -> None:
    """DistributionMetric: required params are exactly (p, q)."""
    metric = distribution.distribution_metrics[name]
    assert satisfies_distribution_metric(metric), (
        f"{name!r} does not satisfy DistributionMetric Protocol"
    )
    # Confirm it doesn't satisfy any other Protocol shape
    assert not satisfies_state_aware_metric(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMetric"
    )
    assert not satisfies_composite_metric(metric), (
        f"{name!r} unexpectedly satisfies CompositeMetric"
    )
    assert not satisfies_stateful_distribution_metric(metric), (
        f"{name!r} unexpectedly satisfies StatefulDistributionMetric"
    )


@pytest.mark.parametrize("name", STATE_AWARE_METRICS)
def test_state_aware_metric_satisfies_protocol(name: str) -> None:
    """StateAwareMetric: required params are exactly (p, state)."""
    metric = distribution.state_aware_metrics[name]
    assert satisfies_state_aware_metric(metric), (
        f"{name!r} does not satisfy StateAwareMetric Protocol"
    )
    assert not satisfies_distribution_metric(metric), (
        f"{name!r} unexpectedly satisfies DistributionMetric"
    )
    assert not satisfies_composite_metric(metric), (
        f"{name!r} unexpectedly satisfies CompositeMetric"
    )


@pytest.mark.parametrize("name", COMPOSITE_METRICS)
def test_composite_metric_satisfies_protocol(name: str) -> None:
    """CompositeMetric: params include forward, partitioned, selectivity."""
    metric = distribution.composite_metrics[name]
    assert satisfies_composite_metric(metric), (
        f"{name!r} does not satisfy CompositeMetric Protocol"
    )
    assert not satisfies_distribution_metric(metric), (
        f"{name!r} unexpectedly satisfies DistributionMetric"
    )
    assert not satisfies_state_aware_metric(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMetric"
    )


@pytest.mark.parametrize("name", STATEFUL_DISTRIBUTION_METRICS)
def test_stateful_distribution_metric_satisfies_protocol(name: str) -> None:
    """StatefulDistributionMetric: required params are exactly (p, q, state)."""
    metric = distribution.stateful_distribution_metrics[name]
    assert satisfies_stateful_distribution_metric(metric), (
        f"{name!r} does not satisfy StatefulDistributionMetric Protocol"
    )
    assert not satisfies_state_aware_metric(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMetric"
    )
    assert not satisfies_composite_metric(metric), (
        f"{name!r} unexpectedly satisfies CompositeMetric"
    )
