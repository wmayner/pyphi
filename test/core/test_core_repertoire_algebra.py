"""Tests for pyphi.core.repertoire_algebra — stateless repertoire functions + cache."""

from __future__ import annotations

import gc

import pytest


def test_memoize_caches_results() -> None:
    """A memoized function returns the cached value on second call."""
    from pyphi.core.repertoire_algebra import _memoize

    call_count = {"n": 0}

    @_memoize
    def f(cs, x):
        call_count["n"] += 1
        return x * 2

    class FakeCs:
        _fingerprint = b"fake-caches-fp"

        def _resolved_background_conditioning(self):
            return "CAUSAL_MARGINALIZATION"

    cs = FakeCs()
    assert f(cs, 3) == 6
    assert f(cs, 3) == 6
    assert call_count["n"] == 1


def test_memoize_evicts_on_gc() -> None:
    """When the last carrier of a fingerprint is GC'd, its entries are evicted."""
    from pyphi.core.repertoire_algebra import _kernel_caches
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        return x * 2

    class FakeCs:
        _fingerprint = b"fake-evict-fp"

        def _resolved_background_conditioning(self):
            return "CAUSAL_MARGINALIZATION"

    cs = FakeCs()
    f(cs, 1)
    f(cs, 2)
    cache = _kernel_caches[f.__name__]
    assert cache.size == 2
    del cs
    gc.collect()
    assert cache.size == 0


def test_memoize_does_not_poison_on_failure() -> None:
    """A raised exception must not pollute the cache."""
    from pyphi.core.repertoire_algebra import _memoize

    call_count = {"n": 0}

    @_memoize
    def f(cs, x):
        call_count["n"] += 1
        if x == 1:
            raise ValueError("boom")
        return x * 2

    class FakeCs:
        _fingerprint = b"fake-poison-fp"

        def _resolved_background_conditioning(self):
            return "CAUSAL_MARGINALIZATION"

    cs = FakeCs()
    with pytest.raises(ValueError):
        f(cs, 1)
    assert f(cs, 2) == 4
    assert call_count["n"] == 2


@pytest.fixture
def cs():
    from pyphi import examples
    from pyphi.system import System

    return System(
        substrate=examples.basic_substrate(),
        state=(1, 0, 0),
        node_indices=(0, 1, 2),
    )


def test_kernel_caches_appear_in_registry() -> None:
    """Each kernel-memoized function registers a policy under kernel.<name>."""
    from pyphi import cache as cache_module
    from pyphi.cache import registry as reg
    from pyphi.core import repertoire_algebra as ra  # noqa: F401  # trigger decoration

    keys = list(reg._registry.keys())
    kernel_keys = [k for k in keys if k.startswith("kernel.")]
    assert kernel_keys, f"expected kernel.* entries; got: {keys}"

    info = cache_module.info()
    assert all(k in info for k in kernel_keys)


def test_kernel_clear_via_registry_clears_kernel_cache(cs) -> None:
    """pyphi.cache.clear('kernel.<name>') empties that kernel cache."""
    from pyphi import cache as cache_module
    from pyphi.core import repertoire_algebra as ra

    ra._single_node_cause_repertoire(cs, 0, frozenset({0, 1}))
    name = "kernel._single_node_cause_repertoire"
    assert cache_module.info()[name].currsize >= 1

    cache_module.clear(name)
    assert cache_module.info()[name].currsize == 0


def test_kernel_cache_respects_memory_full(monkeypatch, cs) -> None:
    """When memory_full() returns True, kernel cache stops adding entries."""
    from pyphi import cache as cache_module
    from pyphi.cache import cache_utils
    from pyphi.core import repertoire_algebra as ra

    cache_module.clear_all()

    monkeypatch.setattr(cache_utils, "memory_full", lambda: True)

    ra._single_node_cause_repertoire(cs, 0, frozenset({0, 1}))
    ra._single_node_cause_repertoire(cs, 0, frozenset({1, 2}))

    info = cache_module.info()["kernel._single_node_cause_repertoire"]
    assert info.currsize == 0, (
        f"expected 0 cached entries when memory full, got {info.currsize}"
    )
    assert info.misses >= 2


def test_forward_cause_repertoire_single_state_poisons_uncomputed(cs) -> None:
    """With ``purview_state`` given, only that state's entry is computed; the
    rest must be NaN, never uninitialized memory."""
    import numpy as np

    from pyphi.core import repertoire_algebra as ra

    rep = ra.forward_cause_repertoire(cs, (0, 1, 2), (0, 1, 2), (0, 0, 0)).squeeze()
    assert np.isfinite(rep[(0, 0, 0)])
    unwritten = [s for s in np.ndindex(rep.shape) if s != (0, 0, 0)]
    assert all(np.isnan(rep[s]) for s in unwritten)


def test_effect_repertoire_rejects_wrong_length_mechanism_state(cs) -> None:
    """A ``mechanism_state`` whose length differs from the mechanism must raise,
    not silently truncate the pairing."""
    from pyphi.core import repertoire_algebra as ra

    with pytest.raises(ValueError, match="mechanism_state"):
        ra.effect_repertoire(cs, (0, 2), (0, 1, 2), mechanism_state=(1,))
    with pytest.raises(ValueError, match="mechanism_state"):
        ra.effect_repertoire(cs, (0, 2), (0, 1, 2), mechanism_state=(1, 0, 1, 1))


def test_cached_repertoire_is_read_only():
    """A caller must not be able to poison the kernel cache in place."""
    import numpy as np

    from pyphi import examples

    system = examples.basic_system()
    r = system.cause_repertoire((0,), (1,))
    with pytest.raises(ValueError, match="read-only"):
        r[...] = 99.0
    again = examples.basic_system().cause_repertoire((0,), (1,))
    assert np.array_equal(again, r)


def test_effect_repertoire_is_read_only():
    from pyphi import examples

    system = examples.basic_system()
    r = system.effect_repertoire((0,), (1,))
    with pytest.raises(ValueError, match="read-only"):
        r[...] = 99.0


def test_max_entropy_distribution_is_read_only():
    from pyphi.distribution import max_entropy_distribution

    d = max_entropy_distribution((0, 1, 2), (1,))
    with pytest.raises(ValueError, match="read-only"):
        d[...] = 99.0


def test_intrinsic_information_collects_noise_tied_states() -> None:
    """States within ``config.numerics.precision`` of the maximum join the
    tie family, even when their float values differ by ulp-level noise."""
    import pyphi
    from pyphi import Direction
    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.measures.distribution import resolve_mechanism_measure

    with pyphi.config.override(**presets.iit4_2026):
        measure = resolve_mechanism_measure(
            pyphi.config.formalism.iit.specification_measure
        )
        system = examples.iit4_2023_fig1a_system()
        spec = system.intrinsic_information(
            Direction.EFFECT, (2,), (0, 2), specification_measure=measure
        )
    # (1, 1) computes 2 ulp below (0, 1); both belong to the tie family.
    assert spec.state == (0, 1)
    assert sorted(t.state for t in spec.ties) == [(0, 1), (1, 1)]


def test_intrinsic_information_winner_is_first_enumerated_tied_state() -> None:
    """The winner is the first tied state in enumeration order — not the raw
    float argmax — and the runner-up is never the winner itself."""
    import pyphi
    from pyphi import Direction
    from pyphi import examples
    from pyphi.conf import presets
    from pyphi.measures.distribution import resolve_mechanism_measure

    with pyphi.config.override(**presets.iit4_2026):
        measure = resolve_mechanism_measure(
            pyphi.config.formalism.iit.specification_measure
        )
        system = examples.iit4_2023_fig1a_system()
        # (1, 1) ties the raw argmax (0, 1) within precision; enumerating it
        # first makes it the winner under enumeration-order selection.
        spec = system.intrinsic_information(
            Direction.EFFECT,
            (2,),
            (0, 2),
            specification_measure=measure,
            states=[(1, 1), (0, 1), (1, 0), (0, 0)],
        )
    assert spec.state == (1, 1)
    assert spec.runner_up_state == (0, 1)
    assert spec.runner_up_state != spec.state
