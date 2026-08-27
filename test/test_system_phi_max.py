from itertools import chain

import pytest

from pyphi import Direction
from pyphi import System
from pyphi.formalism import find_mice
from pyphi.formalism import find_mip
from pyphi.formalism import mic
from pyphi.formalism import mie
from pyphi.formalism import phi_max
from pyphi.models import DirectedBipartition
from pyphi.models import MaximallyIrreducibleCauseOrEffect
from pyphi.models import _null_ria
from pyphi.numerics import eq

from . import example_substrates

# Expected results {{{
# ====================

s = example_substrates.s()
directions = (Direction.CAUSE, Direction.EFFECT)
cuts = (None, DirectedBipartition(Direction.EFFECT, (1, 2), (0,)))
system = {
    cut: System(s.substrate, s.state, s.node_indices, partition=cut) for cut in cuts
}

expected_purview_indices = {
    cuts[0]: {
        Direction.CAUSE: {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (1, 2),
            (0, 1, 2): (0, 1, 2),
        },
        Direction.EFFECT: {
            (1,): (0,),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (0, 1, 2),
        },
    },
    cuts[1]: {
        Direction.CAUSE: {
            (1,): (2,),
            (2,): (0, 1),
            (0, 1): (),
            (0, 1, 2): (),
        },
        Direction.EFFECT: {
            (1,): (2,),
            (2,): (1,),
            (0, 1): (2,),
            (0, 1, 2): (),
        },
    },
}
expected_purviews = {
    cut: {
        direction: dict(expected_purview_indices[cut][direction].items())
        for direction in directions
    }
    for cut in cuts
}
expected_mips = {
    cut: {
        direction: {
            mechanism: find_mip(system[cut], direction, mechanism, purview)
            for mechanism, purview in expected_purviews[cut][direction].items()
        }
        for direction in directions
    }
    for cut in cuts
}
expected_mice = {
    cut: {
        direction: [
            MaximallyIrreducibleCauseOrEffect(mip)
            for mechanism, mip in expected_mips[cut][direction].items()
        ]
        for direction in directions
    }
    for cut in cuts
}

# }}}
# `find_mice` tests {{{
# =====================

mice_scenarios = [
    [
        [(cut, direction, mice) for mice in expected_mice[cut][direction]]
        for direction in directions
    ]
    for cut in cuts
]
# Flatten doubly-nested list of scenarios.
mice_scenarios = list(chain(*list(chain(*mice_scenarios))))


mice_parameter_string = "cut,direction,expected"


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_find_mice(cut, direction, expected):
    result = find_mice(system[cut], direction, expected.mechanism)
    print("Expected:\n", expected)
    print("Result:\n", result)
    assert result == expected


def test_find_mice_empty(s):
    expected = [
        MaximallyIrreducibleCauseOrEffect(_null_ria(direction, (), ()))
        for direction in directions
    ]
    assert all(find_mice(s, mice.direction, mice.mechanism) == mice for mice in expected)


# }}}
# `phi_max` tests {{{
# ===================


@pytest.mark.parametrize(mice_parameter_string, mice_scenarios)
def test_mic_or_mie(cut, direction, expected):
    if direction == Direction.CAUSE:
        result = mic(system[cut], expected.mechanism)
    elif direction == Direction.EFFECT:
        result = mie(system[cut], expected.mechanism)
    assert result == expected


phi_max_scenarios = [
    [
        (cut, cause.mechanism, min(cause.phi, effect.phi))
        for cause, effect in zip(
            expected_mice[cut][Direction.CAUSE],
            expected_mice[cut][Direction.EFFECT],
            strict=False,
        )
    ]
    for cut in cuts
]
# Flatten singly-nested list of scenarios.
phi_max_scenarios = list(chain(*phi_max_scenarios))


@pytest.mark.parametrize("cut,mechanism,expected_phi_max", phi_max_scenarios)
def test_phi_max(cut, expected_phi_max, mechanism):
    assert eq(phi_max(system[cut], mechanism), expected_phi_max)


# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :


def test_explicit_purviews_bound_the_enumeration(s, monkeypatch):
    """An explicit purview list bounds the substrate-level enumeration to the
    largest given purview; the result still equals the unbounded result
    intersected with the given list."""
    from pyphi.core import repertoire_algebra as ra
    from pyphi.direction import Direction
    from pyphi.substrate import Substrate

    system = s
    mechanism = (0,)
    given = [(1,), (2,)]
    unbounded = ra.potential_purviews(system, Direction.CAUSE, mechanism)
    expected = {p for p in unbounded if p in set(given)}

    calls = []
    orig = Substrate.potential_purviews

    def spy(self, direction, mech, max_order=None):
        calls.append(max_order)
        return orig(self, direction, mech, max_order=max_order)

    monkeypatch.setattr(Substrate, "potential_purviews", spy)
    result = ra.potential_purviews(system, Direction.CAUSE, mechanism, purviews=given)
    assert set(result) == expected
    assert calls == [1]


def test_explicit_purviews_accepts_one_shot_iterable(s):
    """A generator of purviews gives the same result as the equivalent list.

    The bound scan over ``purviews`` must not exhaust a one-shot iterable
    before the intersection, which would silently discard every candidate.
    """
    from pyphi.core import repertoire_algebra as ra

    mechanism = (0,)
    given = ra.potential_purviews(s, Direction.CAUSE, mechanism)
    assert given  # control: candidates exist
    from_list = ra.potential_purviews(
        s, Direction.CAUSE, mechanism, purviews=list(given)
    )
    from_iter = ra.potential_purviews(
        s, Direction.CAUSE, mechanism, purviews=iter(list(given))
    )
    assert set(from_iter) == set(from_list) == set(given)


def test_mic_accepts_one_shot_purviews_iterable(s):
    """``System.mic`` with a generator of purviews matches the list result."""
    from test.conftest import IIT_4_CONFIG

    mechanism = (0,)
    with IIT_4_CONFIG:
        given = list(s.potential_purviews(Direction.CAUSE, mechanism))
        assert given  # control: candidates exist
        from_list = s.mic(mechanism, purviews=given)
        from_iter = s.mic(mechanism, purviews=iter(given))
    assert from_iter.purview == from_list.purview
    assert from_iter.phi == from_list.phi
    assert from_iter.purview != ()
