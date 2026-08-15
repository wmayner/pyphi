"""The full-state repertoire sweeps admit nothing to the kernel cache.

A sweep that evaluates a repertoire at every state of a mechanism or purview
reads each intermediate exactly once, so admitting them would grow the cache
by the state count while returning no hits. These tests pin that the sweeps
run under ``transient_repertoires`` and that suppressing admission changes no
value.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import cache
from pyphi import examples
from pyphi.core import repertoire_algebra as ra
from pyphi.direction import Direction
from pyphi.system import System


@pytest.fixture(autouse=True)
def _clear_caches():
    cache.clear_all()
    yield
    cache.clear_all()


def _system() -> System:
    return System(
        substrate=examples.basic_substrate(),
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )


def _kernel_size() -> int:
    return sum(c.size for c in ra._kernel_caches.values())


def test_transient_scope_suppresses_admission_but_not_reads():
    cs = _system()
    idx = cs.node_indices

    ra.effect_repertoire(cs, idx, idx)
    warm = _kernel_size()
    assert warm > 0

    with ra.transient_repertoires():
        # A read of an entry admitted outside the scope still hits.
        ra.effect_repertoire(cs, idx, idx)
        # A fresh computation is returned but not admitted.
        ra.effect_repertoire(cs, idx, idx, mechanism_state=(1, 1, 1))
    assert _kernel_size() == warm

    # Outside the scope the same fresh computation is admitted.
    ra.effect_repertoire(cs, idx, idx, mechanism_state=(1, 1, 1))
    assert _kernel_size() > warm


def test_transient_scope_is_restored_on_exception():
    cs = _system()
    idx = cs.node_indices
    with pytest.raises(RuntimeError), ra.transient_repertoires():
        raise RuntimeError("boom")
    before = _kernel_size()
    ra.effect_repertoire(cs, idx, idx, mechanism_state=(1, 1, 1))
    assert _kernel_size() > before


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
def test_full_state_sweep_admits_only_its_own_result(direction):
    """The sweep's per-state intermediates leave the cache untouched.

    The cause sweep memoizes its own assembled repertoire, so it admits one
    entry; the effect sweep memoizes nothing and admits none. Neither admits
    an entry per state, which is what the state count would otherwise cost.
    """
    cs = _system()
    idx = cs.node_indices
    n_states = 2 ** len(idx)

    before = _kernel_size()
    ra.unconstrained_forward_repertoire(cs, direction, idx, idx)
    ra.forward_repertoire(cs, direction, idx, idx, None)
    admitted = _kernel_size() - before

    assert admitted < n_states, (
        f"{direction} sweep admitted {admitted} entries for {n_states} states; "
        "the per-state intermediates are being cached"
    )


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
def test_sweep_values_match_uncached_computation(direction):
    """Suppressing admission is a memory policy, not a change of value."""
    cs = _system()
    idx = cs.node_indices

    unconstrained = np.array(
        ra.unconstrained_forward_repertoire(cs, direction, idx, idx)
    )
    forward = np.array(ra.forward_repertoire(cs, direction, idx, idx, None))

    cache.clear_all()
    with ra.transient_repertoires():
        bare_unconstrained = np.array(
            ra.unconstrained_forward_repertoire(cs, direction, idx, idx)
        )
        bare_forward = np.array(ra.forward_repertoire(cs, direction, idx, idx, None))

    np.testing.assert_array_equal(unconstrained, bare_unconstrained)
    np.testing.assert_array_equal(forward, bare_forward)


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
def test_ten_unit_sweep_holds_cache_occupancy_flat(direction):
    """The guard that bites at scale.

    Admission during the sweep costs one full repertoire per system state:
    at ten units that is 2²⁰ cells to answer 2¹⁰ single-use questions, and
    the cost grows fourfold per unit added. Call counts cannot see this —
    the same repertoires are computed either way — so occupancy is the
    quantity to pin.
    """
    from test.golden.perf_fixtures import noisy_ring

    substrate = noisy_ring(10, seed=2026)
    cs = System(substrate, (0,) * 10, substrate.node_indices)
    idx = cs.node_indices

    before = _kernel_size()
    ra.unconstrained_forward_repertoire(cs, direction, idx, idx)
    ra.forward_repertoire(cs, direction, idx, idx, None)
    admitted = _kernel_size() - before

    # What the sweeps legitimately admit is the current-state repertoire and
    # its per-purview-node factors, plus the cause sweep's own assembled
    # result: a count that scales with the number of units, not with the
    # number of states. Admitting the per-state intermediates instead would
    # scale with the latter (11,264 entries here rather than 11).
    n_units = len(idx)
    assert admitted <= 2 * n_units + 2, (
        f"{direction} sweep admitted {admitted} entries over {2**n_units} "
        f"states; admissions should scale with the {n_units} units"
    )
