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
  DistributionMeasure          : exactly two required params (p, q)
  StateAwareMeasure            : exactly two required params (p, state) — no q
  CompositeMeasure             : first three params name forward, partitioned,
                                selectivity repertoires
  StatefulDistributionMeasure  : exactly three required params (p, q, state)
"""

from __future__ import annotations

import pytest

from pyphi.metrics import distribution
from pyphi.metrics.protocols import satisfies_composite_measure
from pyphi.metrics.protocols import satisfies_distribution_measure
from pyphi.metrics.protocols import satisfies_state_aware_measure
from pyphi.metrics.protocols import satisfies_stateful_distribution_measure

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
    """DistributionMeasure: required params are exactly (p, q)."""
    metric = distribution.distribution_measures[name]
    assert satisfies_distribution_measure(metric), (
        f"{name!r} does not satisfy DistributionMeasure Protocol"
    )
    # Confirm it doesn't satisfy any other Protocol shape
    assert not satisfies_state_aware_measure(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMeasure"
    )
    assert not satisfies_composite_measure(metric), (
        f"{name!r} unexpectedly satisfies CompositeMeasure"
    )
    assert not satisfies_stateful_distribution_measure(metric), (
        f"{name!r} unexpectedly satisfies StatefulDistributionMeasure"
    )


@pytest.mark.parametrize("name", STATE_AWARE_METRICS)
def test_state_aware_metric_satisfies_protocol(name: str) -> None:
    """StateAwareMeasure: required params are exactly (p, state)."""
    metric = distribution.state_aware_measures[name]
    assert satisfies_state_aware_measure(metric), (
        f"{name!r} does not satisfy StateAwareMeasure Protocol"
    )
    assert not satisfies_distribution_measure(metric), (
        f"{name!r} unexpectedly satisfies DistributionMeasure"
    )
    assert not satisfies_composite_measure(metric), (
        f"{name!r} unexpectedly satisfies CompositeMeasure"
    )


@pytest.mark.parametrize("name", COMPOSITE_METRICS)
def test_composite_metric_satisfies_protocol(name: str) -> None:
    """CompositeMeasure: params include forward, partitioned, selectivity."""
    metric = distribution.composite_measures[name]
    assert satisfies_composite_measure(metric), (
        f"{name!r} does not satisfy CompositeMeasure Protocol"
    )
    assert not satisfies_distribution_measure(metric), (
        f"{name!r} unexpectedly satisfies DistributionMeasure"
    )
    assert not satisfies_state_aware_measure(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMeasure"
    )


@pytest.mark.parametrize("name", STATEFUL_DISTRIBUTION_METRICS)
def test_stateful_distribution_metric_satisfies_protocol(name: str) -> None:
    """StatefulDistributionMeasure: required params are exactly (p, q, state)."""
    metric = distribution.stateful_distribution_measures[name]
    assert satisfies_stateful_distribution_measure(metric), (
        f"{name!r} does not satisfy StatefulDistributionMeasure Protocol"
    )
    assert not satisfies_state_aware_measure(metric), (
        f"{name!r} unexpectedly satisfies StateAwareMeasure"
    )
    assert not satisfies_composite_measure(metric), (
        f"{name!r} unexpectedly satisfies CompositeMeasure"
    )
