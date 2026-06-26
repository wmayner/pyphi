"""Pin every registered metric's Protocol membership.

Catches signature drift: if a metric's function signature changes such
that it no longer matches its declared Protocol, the test fails.

Protocol membership is checked via ``inspect.signature`` rather than
``isinstance``, because the registered metrics are plain functions and
plain functions do not carry the ``name``/``asymmetric`` class-level
attributes that the runtime-checkable Protocols declare.  The
structural intent — that each metric satisfies exactly one Protocol
shape — is captured by the shared ``satisfies_*`` predicates in
:mod:`pyphi.measures.protocols`, which the typed registries also use at
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

from pyphi.measures import distribution
from pyphi.measures.protocols import satisfies_composite_measure
from pyphi.measures.protocols import satisfies_distribution_measure
from pyphi.measures.protocols import satisfies_state_aware_measure
from pyphi.measures.protocols import satisfies_stateful_distribution_measure

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


# ---------------------------------------------------------------------------
# Structural-attribute tests
#
# Composite measures carry attributes that drive dispatch in
# ``pyphi.formalism.iit4`` and ``pyphi.core.repertoire_algebra``, replacing
# string comparisons on ``measure.name``:
#
#   - ``applies_ii_cap``: True only for INTRINSIC_INFORMATION (Eq. 23 cap).
#   - ``partition_measure``: the measure used at partition level (II swaps
#     to GID; GID uses itself, encoded as ``None``).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", COMPOSITE_METRICS)
def test_composite_measure_has_applies_ii_cap_attribute(name: str) -> None:
    metric = distribution.composite_measures[name]
    assert hasattr(metric, "applies_ii_cap"), (
        f"{name!r} composite measure missing ``applies_ii_cap`` attribute"
    )
    assert isinstance(metric.applies_ii_cap, bool)


@pytest.mark.parametrize("name", COMPOSITE_METRICS)
def test_composite_measure_has_partition_measure_attribute(name: str) -> None:
    metric = distribution.composite_measures[name]
    assert hasattr(metric, "partition_measure"), (
        f"{name!r} composite measure missing ``partition_measure`` attribute"
    )


def test_only_intrinsic_information_applies_cap() -> None:
    ii = distribution.composite_measures["INTRINSIC_INFORMATION"]
    gid = distribution.composite_measures["GENERALIZED_INTRINSIC_DIFFERENCE"]
    assert ii.applies_ii_cap is True
    assert gid.applies_ii_cap is False


def test_intrinsic_information_partition_measure_is_gid() -> None:
    ii = distribution.composite_measures["INTRINSIC_INFORMATION"]
    gid = distribution.composite_measures["GENERALIZED_INTRINSIC_DIFFERENCE"]
    assert ii.partition_measure is gid


def test_generalized_intrinsic_difference_partition_measure_is_none() -> None:
    gid = distribution.composite_measures["GENERALIZED_INTRINSIC_DIFFERENCE"]
    assert gid.partition_measure is None
