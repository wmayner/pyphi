"""Pins for ``intrinsic_information``'s winner, tie-family, and runner-up
semantics.

The winner is the first tolerance-tied state in enumeration order
(``utils.all_states``: little-endian, index 0 varies fastest), the tie family
is tolerance-based via ``numerics.eq`` against the raw maximum, and the runner
up is the highest raw value among non-winner states with exact-value ties
broken by enumeration order. These tests pin all three so a vectorized
implementation cannot drift from the enumeration-based semantics.
"""

import numpy as np
import pytest

import pyphi
from pyphi import Direction
from pyphi import examples
from pyphi import numerics
from pyphi.conf import presets
from pyphi.core import repertoire_algebra as ra
from pyphi.utils import all_states


@pytest.fixture(autouse=True)
def _pin_formalism():
    with pyphi.config.override(**presets.iit4_2026, progress_bars=False):
        yield


def _system():
    return pyphi.System(examples.basic_substrate(), (1, 0, 0))


def _crafted_measure(values):
    arr = np.asarray(values, dtype=float)

    def measure(
        forward_repertoire,
        partitioned_forward_repertoire,
        selectivity_repertoire,
    ):
        return arr

    return measure


def test_tolerance_tied_winner_precedes_raw_argmax():
    """A state that ties the maximum within tolerance but comes earlier in
    enumeration order wins over the raw argmax."""
    # Purview (0, 1); enumeration order: (0,0), (1,0), (0,1), (1,1).
    dist = np.empty((2, 2))
    dist[0, 0] = 0.5
    dist[1, 0] = 1.0 - 1e-15  # tolerance-tied, earlier in enumeration
    dist[0, 1] = 1.0  # raw maximum, later in enumeration
    dist[1, 1] = 0.25
    spec = ra.intrinsic_information(
        _system(),
        Direction.EFFECT,
        (0,),
        (0, 1),
        specification_measure=_crafted_measure(dist),  # pyright: ignore[reportArgumentType]
    )
    assert spec.state == (1, 0)
    assert spec.intrinsic_information == 1.0 - 1e-15
    assert [t.state for t in spec.ties] == [(1, 0), (0, 1)]
    assert spec.runner_up_state == (0, 1)
    assert spec.runner_up_intrinsic_information == 1.0


def test_exact_ties_keep_enumeration_order():
    dist = np.empty((2, 2))
    dist[0, 0] = 0.5
    dist[1, 0] = 1.0
    dist[0, 1] = 1.0
    dist[1, 1] = 0.25
    spec = ra.intrinsic_information(
        _system(),
        Direction.EFFECT,
        (0,),
        (0, 1),
        specification_measure=_crafted_measure(dist),  # pyright: ignore[reportArgumentType]
    )
    assert spec.state == (1, 0)
    assert [t.state for t in spec.ties] == [(1, 0), (0, 1)]
    assert spec.runner_up_state == (0, 1)
    assert spec.runner_up_intrinsic_information == 1.0


def test_no_ties_yields_singleton_family():
    dist = np.empty((2, 2))
    dist[0, 0] = 0.1
    dist[1, 0] = 0.2
    dist[0, 1] = 0.9
    dist[1, 1] = 0.3
    spec = ra.intrinsic_information(
        _system(),
        Direction.EFFECT,
        (0,),
        (0, 1),
        specification_measure=_crafted_measure(dist),  # pyright: ignore[reportArgumentType]
    )
    assert spec.state == (0, 1)
    assert [t.state for t in spec.ties] == [(0, 1)]
    assert spec.runner_up_state == (1, 1)
    assert spec.runner_up_intrinsic_information == 0.3


def _reference(cs, direction, mechanism, purview, measure):
    """The enumeration-based algorithm, spelled out directly."""
    alphabet = cs.substrate.factored_tpm.alphabet_sizes
    purview_k = tuple(alphabet[i] for i in purview)
    states = list(all_states(purview_k))
    selectivity = ra.repertoire(cs, direction, mechanism, purview)
    rep = ra.forward_repertoire(cs, direction, mechanism, purview, None)
    unconstrained = ra.unconstrained_forward_repertoire(
        cs, direction, mechanism, purview
    )
    dist = np.asarray(measure(rep, unconstrained, selectivity)).squeeze()
    info = {s: float(dist[s]) for s in states}
    mx = max(info.values())
    tied = [(s, v) for s, v in info.items() if numerics.eq(v, mx)]
    ranked = sorted(info.items(), key=lambda kv: kv[1], reverse=True)
    winner = tied[0][0]
    runner_up = next(((s, v) for s, v in ranked if s != winner), None)
    return winner, tied, runner_up


@pytest.mark.parametrize("direction", [Direction.CAUSE, Direction.EFFECT])
@pytest.mark.parametrize(
    "mechanism,purview",
    [((0,), (0, 1)), ((0, 1), (1, 2)), ((0, 1, 2), (0, 1, 2))],
)
def test_matches_reference_on_real_measure(direction, mechanism, purview):
    """The implementation agrees with the enumeration-based reference under
    the real specification measure, on every field."""
    from pyphi.measures.distribution import resolve_mechanism_measure

    system = _system()
    measure = resolve_mechanism_measure(pyphi.config.formalism.iit.specification_measure)
    spec = ra.intrinsic_information(
        system, direction, mechanism, purview, specification_measure=measure
    )
    winner, tied, runner_up = _reference(system, direction, mechanism, purview, measure)
    assert spec.state == winner
    assert [t.state for t in spec.ties] == [s for s, _ in tied]
    assert [t.intrinsic_information for t in spec.ties] == [v for _, v in tied]
    if runner_up is None:
        assert spec.runner_up_state is None
    else:
        assert spec.runner_up_state == runner_up[0]
        assert spec.runner_up_intrinsic_information == runner_up[1]


def test_uniform_system_ties_every_state():
    """A uniform substrate ties every purview state exactly; the winner is
    the first state in enumeration order."""
    n = 2
    tpm = np.full([2] * n + [n], 0.5)
    substrate = pyphi.Substrate(tpm)
    system = pyphi.System(substrate, (0,) * n)
    from pyphi.measures.distribution import resolve_mechanism_measure

    measure = resolve_mechanism_measure(pyphi.config.formalism.iit.specification_measure)
    spec = ra.intrinsic_information(
        system, Direction.EFFECT, (0,), (0, 1), specification_measure=measure
    )
    assert spec.state == (0, 0)
    assert [t.state for t in spec.ties] == list(all_states((2, 2)))


def test_unconstrained_forward_effect_repertoire_matches_stacked_mean():
    """The running-mean implementation equals the stacked mean exactly on
    small systems (both accumulate sequentially below numpy's pairwise
    threshold)."""
    system = _system()
    mechanism, purview = (0, 1), (0, 2)
    expected = np.stack(
        [
            ra.forward_effect_repertoire(system, mechanism, purview, mechanism_state=s)
            for s in all_states((2, 2))
        ]
    ).mean(axis=0)
    result = ra.unconstrained_forward_effect_repertoire(system, mechanism, purview)
    assert np.array_equal(np.asarray(result), expected)


def _tiny_system(n: int = 3):
    tpm = np.full([2] * n + [n], 0.5)
    return pyphi.System(pyphi.Substrate(tpm), (0,) * n)


def test_unconstrained_forward_effect_repertoire_size_guard(monkeypatch):
    """Above the guard threshold the computation refuses with an estimate
    instead of grinding."""
    monkeypatch.setattr(ra, "_MAX_FORWARD_SWEEP_STATES", 2)
    with pytest.raises(ValueError, match="infeasible"):
        ra.unconstrained_forward_effect_repertoire(_tiny_system(), (0, 1), (0, 2))


def test_forward_cause_repertoire_size_guard(monkeypatch):
    """The cause sweep is bounded too, and it is the one walked first.

    ``Direction.both()`` orders the cause direction ahead of the effect one, so
    a bound on the effect sweep alone lets an oversized system spend the whole
    cause sweep before anything refuses.
    """
    monkeypatch.setattr(ra, "_MAX_FORWARD_SWEEP_STATES", 2)
    with pytest.raises(ValueError, match="infeasible"):
        ra.forward_cause_repertoire(_tiny_system(), (0, 1), (0, 2))


def test_forward_cause_repertoire_single_state_is_not_a_sweep(monkeypatch):
    """Asking for one state walks one state, whatever the sweep bound is."""
    monkeypatch.setattr(ra, "_MAX_FORWARD_SWEEP_STATES", 2)
    assert (
        ra.forward_cause_repertoire(_tiny_system(), (0, 1), (0, 2), (1, 1)) is not None
    )


def test_both_sweep_guards_fire_before_the_specified_state_search_runs(monkeypatch):
    """The intrinsic-information search refuses rather than half-running."""
    from pyphi.formalism.iit4 import system_intrinsic_information
    from pyphi.measures.distribution import resolve_mechanism_measure

    monkeypatch.setattr(ra, "_MAX_FORWARD_SWEEP_STATES", 2)
    system = _tiny_system()
    with pytest.raises(ValueError, match="infeasible"):
        system_intrinsic_information(
            system,
            specification_measure=resolve_mechanism_measure(
                pyphi.config.formalism.iit.specification_measure
            ),
        )
