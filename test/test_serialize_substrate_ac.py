import pytest

from pyphi import actual
from pyphi import examples
from pyphi import serialize
from pyphi.actual import Transition
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.complex import Complex
from pyphi.models.complex import ExcludedCandidate

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_transition():
    sub = examples.actual_causation_substrate()
    return Transition(
        sub,
        before_state=(1, 1),
        after_state=(1, 1),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_substrate_round_trips(fmt):
    obj = examples.basic_substrate()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_system_round_trips(fmt):
    obj = examples.basic_system()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_round_trips(fmt):
    obj = make_transition()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_account_round_trips(fmt):
    obj = actual.account(make_transition())
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is type(obj)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_sia_round_trips(fmt):
    obj = actual.sia(make_transition())
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_complex_round_trips(fmt):
    obj = Complex(
        sia=NullSystemIrreducibilityAnalysis(node_indices=(0, 1)),
        substrate=examples.basic_substrate(),
        is_maximal=True,
        excluded=(ExcludedCandidate((0,), 0.1),),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is Complex
