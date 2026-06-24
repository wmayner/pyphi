"""N2 — parallel ≡ sequential invariant (loky process scheduler).

A standing CI guard that the loky-backed *parallel* SIA/CES path produces
results identical to *sequential* evaluation, for both IIT 3.0 and IIT 4.0.

Why this exists. The IIT 3.0 EMD goldens (``basic_iit3_emd``,
``xor_iit3_emd``) once hit an intermittent loky ``BrokenProcessPool`` — the
P9 cache-registry-leak trigger (``clear_all`` walking a growing per-``Network``
``PurviewCache`` registry crashed workers on un-serialize), fixed by making
those caches anonymous. The golden harness sidesteps the path entirely by
running ``parallel=False``, and the pre-existing
``test_parallel_and_sequential_ces_are_equal`` runs at *default* thresholds
where small substrates collapse to sequential — so the loky path had no
standing coverage. (Audit 2026-06-13: the flake does not reproduce under
forced loky even with ``clear_all`` between every run; the P9 fix resolved
it.) These tests force the loky scheduler so any recurrence of the flake, or
any sequential/parallel divergence, fails CI loudly instead of going
unchecked.

The thread/dask schedulers are exercised by ``test_scheduler.py`` /
``test_parallel.py``; the SIA/CES compute entry points dispatch only through
the loky-backed process scheduler, which this module covers.
"""

from __future__ import annotations

import pytest

from pyphi import System
from pyphi import examples
from pyphi import utils
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.formalism import iit3


def _outer_loky(threshold: int = 2) -> dict:
    """Config override that forces loky on the two outer evaluation levels
    that the SIA/CES path dispatches — the SIA cut level
    (``parallel_partition_evaluation``) and the CES concept level
    (``parallel_concept_evaluation``) — while leaving inner levels (purviews,
    relations) at their defaults. Cuts and concepts are the historical
    ``BrokenProcessPool`` path; forcing the inner levels too only adds tiny
    cross-process dispatches (notably the IIT 4.0 relation cohort) that
    dominate runtime without widening coverage.

    This engages real cross-process dispatch — ``map_reduce`` parallelizes only
    when a level produces more than one chunk, so ``sequential_threshold`` must
    be low — without the ``threshold=1`` dispatch explosion that makes every
    tiny per-purview/per-partition op a separate cloudpickle round-trip (a
    single such IIT 3.0 SIA exceeds two minutes; outer-only is ~0.6s after the
    one-time executor spin-up).
    """
    c = config.infrastructure
    forced = {"parallel": True, "sequential_threshold": threshold}
    keys = (
        "parallel_partition_evaluation",
        "parallel_concept_evaluation",
    )
    return {"parallel": True, **{k: {**getattr(c, k), **forced} for k in keys}}


# Binary golden substrates. basic/xor are the historically loky-flaky IIT 3.0
# goldens — kept in the always-on CI lane; rule110/grid3 add breadth under the
# ``--slow`` opt-in (their forced-loky SIA pays more cut evaluations).
_SUBSTRATES = {
    "basic": (examples.basic_substrate, (1, 0, 0)),
    "xor": (examples.xor_substrate, (0, 0, 0)),
    "rule110": (examples.rule110_substrate, (1, 0, 1)),
    "grid3": (examples.grid3_substrate, (1, 0, 0)),
}
_PARAMS = [
    pytest.param("basic"),
    pytest.param("xor"),
    pytest.param("rule110", marks=pytest.mark.slow),
    pytest.param("grid3", marks=pytest.mark.slow),
]


def _ces_signature(distinctions) -> list[tuple[tuple[int, ...], float]]:
    """A reorder-stable structural fingerprint of a cause-effect structure:
    each distinction's mechanism and its |small_phi|, rounded to the active
    numerical precision so sub-precision float-reassociation noise from
    cross-process reduction never masquerades as a real divergence."""
    p = config.numerics.precision
    return sorted((tuple(c.mechanism), round(float(c.phi), p)) for c in distinctions)


@pytest.mark.emd
@pytest.mark.parametrize("name", _PARAMS)
def test_iit3_sia_and_ces_parallel_equals_sequential(name: str) -> None:
    """IIT 3.0 (EMD): forced-loky SIA φ and CES match sequential exactly.

    Exercises the loky path on both the SIA cut level and the CES concept
    level — the exact path of the historical ``basic``/``xor`` flake.
    """
    factory, state = _SUBSTRATES[name]
    with config.override(**presets.iit3, parallel=False):
        seq_sia = iit3.sia(System(factory(), state))
        seq_ces = _ces_signature(iit3.ces(System(factory(), state)).distinctions)
    with config.override(**presets.iit3, **_outer_loky()):
        par_sia = iit3.sia(System(factory(), state))
        par_ces = _ces_signature(iit3.ces(System(factory(), state)).distinctions)

    assert utils.eq(seq_sia.phi, par_sia.phi), (
        f"{name}: IIT 3.0 SIA φ diverged under loky — sequential "
        f"{seq_sia.phi} vs parallel {par_sia.phi}"
    )
    assert seq_ces == par_ces, (
        f"{name}: IIT 3.0 CES diverged under loky.\n"
        f"  sequential: {seq_ces}\n  parallel:   {par_ces}"
    )


@pytest.mark.parametrize("name", _PARAMS)
def test_iit4_sia_parallel_equals_sequential(name: str) -> None:
    """IIT 4.0 (2023, GID): forced-loky SIA equals sequential structurally."""
    factory, state = _SUBSTRATES[name]
    with config.override(**presets.iit4_2023, parallel=False):
        seq = System(factory(), state).sia()
    with config.override(**presets.iit4_2023, **_outer_loky()):
        par = System(factory(), state).sia()

    assert seq == par, (
        f"{name}: IIT 4.0 SIA diverged under loky — sequential φ {seq.phi} "
        f"vs parallel φ {par.phi}"
    )


@pytest.mark.emd
def test_iit3_concepts_cost_balanced_chunking_equals_sequential() -> None:
    """Force a tiny chunksize so the concept level actually chunks under loky,
    exercising the size_func cost-balanced packing end-to-end; the resulting
    CES must match sequential exactly.
    """
    factory, state = _SUBSTRATES["basic"]
    # chunksize=2 with ~7 mechanisms forms several cost-balanced chunks
    concept_cfg = {
        **config.infrastructure.parallel_concept_evaluation,
        "parallel": True,
        "sequential_threshold": 1,
        "chunksize": 2,
    }
    with config.override(**presets.iit3, parallel=False):
        seq = _ces_signature(iit3.ces(System(factory(), state)).distinctions)
    with config.override(
        **presets.iit3, parallel=True, parallel_concept_evaluation=concept_cfg
    ):
        par = _ces_signature(iit3.ces(System(factory(), state)).distinctions)

    assert seq == par, f"cost-balanced concept chunking diverged:\n{seq}\n{par}"
