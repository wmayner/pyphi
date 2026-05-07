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
        pass

    cs = FakeCs()
    assert f(cs, 3) == 6
    assert f(cs, 3) == 6
    assert call_count["n"] == 1


def test_memoize_evicts_on_gc() -> None:
    """When a CandidateSystem is GC'd, its cache entries are evicted."""
    from pyphi.core.repertoire_algebra import _caches
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    cs_id = id(cs)
    f(cs, 1)
    f(cs, 2)
    assert any(k[0] == cs_id for k in _caches[f.__name__])
    del cs
    gc.collect()
    assert not any(k[0] == cs_id for k in _caches[f.__name__])


def test_memoize_does_not_poison_on_failure() -> None:
    """If the wrapped function raises, the cache must not retain a partial entry."""
    from pyphi.core.repertoire_algebra import _memoize

    @_memoize
    def f(cs, x):
        if x < 0:
            raise ValueError("negative")
        return x * 2

    class FakeCs:
        pass

    cs = FakeCs()
    with pytest.raises(ValueError):
        f(cs, -1)
    assert f(cs, 4) == 8
    assert f(cs, 4) == 8


# =============================================================================
# Parity tests — every delegating port must produce the same result as
# calling the equivalent method on a legacy Subsystem.
# =============================================================================


@pytest.fixture
def cs_and_subsystem():
    from pyphi import Subsystem
    from pyphi import examples
    from pyphi.core.candidate_system import CandidateSystem
    from pyphi.core.causal_model import CausalModel

    network = examples.basic_network()
    state = (1, 0, 0)
    nodes = (0, 1, 2)
    cs = CandidateSystem(
        causal_model=CausalModel.from_network(network),
        state=state,
        node_indices=nodes,
    )
    sub = Subsystem(network, state, nodes)
    return cs, sub


@pytest.mark.parametrize(
    "mechanism, purview",
    [((0,), (1,)), ((0, 1), (2,)), ((0, 1, 2), (0, 1, 2))],
)
def test_cause_repertoire_parity(cs_and_subsystem, mechanism, purview) -> None:
    import numpy as np

    from pyphi.core.repertoire_algebra import cause_repertoire

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(
        cause_repertoire(cs, mechanism, purview),
        sub.cause_repertoire(mechanism, purview),
    )


@pytest.mark.parametrize(
    "mechanism, purview",
    [((0,), (1,)), ((0, 1), (2,)), ((0, 1, 2), (0, 1, 2))],
)
def test_effect_repertoire_parity(cs_and_subsystem, mechanism, purview) -> None:
    import numpy as np

    from pyphi.core.repertoire_algebra import effect_repertoire

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(
        effect_repertoire(cs, mechanism, purview),
        sub.effect_repertoire(mechanism, purview),
    )


def test_repertoire_dispatch_parity(cs_and_subsystem) -> None:
    import numpy as np

    from pyphi.core.repertoire_algebra import repertoire
    from pyphi.direction import Direction

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(
        repertoire(cs, Direction.CAUSE, (0,), (1,)),
        sub.repertoire(Direction.CAUSE, (0,), (1,)),
    )


def test_unconstrained_cause_repertoire_parity(cs_and_subsystem) -> None:
    import numpy as np

    from pyphi.core.repertoire_algebra import unconstrained_cause_repertoire

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(
        unconstrained_cause_repertoire(cs, (0, 1, 2)),
        sub.unconstrained_cause_repertoire((0, 1, 2)),
    )


def test_unconstrained_effect_repertoire_parity(cs_and_subsystem) -> None:
    import numpy as np

    from pyphi.core.repertoire_algebra import unconstrained_effect_repertoire

    cs, sub = cs_and_subsystem
    np.testing.assert_array_equal(
        unconstrained_effect_repertoire(cs, (0, 1, 2)),
        sub.unconstrained_effect_repertoire((0, 1, 2)),
    )


def test_forward_repertoire_parity(cs_and_subsystem) -> None:
    import numpy as np

    from pyphi.core.repertoire_algebra import forward_cause_repertoire
    from pyphi.core.repertoire_algebra import forward_effect_repertoire

    cs, sub = cs_and_subsystem
    # forward_cause_repertoire with purview_state=None iterates all states
    # (passing a specific state leaves other entries as np.empty garbage).
    np.testing.assert_array_equal(
        forward_cause_repertoire(cs, (0,), (1,), None),
        sub.forward_cause_repertoire((0,), (1,), None),
    )
    np.testing.assert_array_equal(
        forward_effect_repertoire(cs, (0,), (1,)),
        sub.forward_effect_repertoire((0,), (1,)),
    )


def test_phi_parity(cs_and_subsystem) -> None:
    from pyphi.formalism import phi

    cs, sub = cs_and_subsystem
    assert phi(cs, (0,), (1,)) == pytest.approx(sub.phi((0,), (1,)))


def test_concept_parity(cs_and_subsystem) -> None:
    from pyphi.formalism import concept

    cs, sub = cs_and_subsystem
    assert concept(cs, (0,)).phi == pytest.approx(sub.concept((0,)).phi)


def test_sia_parity(cs_and_subsystem) -> None:
    from pyphi.formalism import sia

    cs, sub = cs_and_subsystem
    assert sia(cs).phi == pytest.approx(sub.sia().phi)


def test_potential_purviews_parity(cs_and_subsystem) -> None:
    from pyphi.core.repertoire_algebra import potential_purviews
    from pyphi.direction import Direction

    cs, sub = cs_and_subsystem
    assert list(potential_purviews(cs, Direction.CAUSE, (0,))) == list(
        sub.potential_purviews(Direction.CAUSE, (0,))
    )
