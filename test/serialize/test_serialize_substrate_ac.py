import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi import serialize
from pyphi.actual import Transition
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.complex import Complex
from pyphi.models.complex import ExcludedCandidate
from pyphi.substrate import Substrate

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


@pytest.mark.parametrize("fmt", FORMATS)
def test_kary_substrate_round_trips(fmt):
    obj = examples.gomez_p53_mdm2_substrate()  # ternary p53 + binary Mdm2 units
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert list(restored.node_labels) == list(obj.node_labels)
    assert restored.factored_tpm.state_space == obj.factored_tpm.state_space


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_preserves_noise_background(fmt):
    # OR gate driven by a noised background unit: the EFFECT ratio is
    # nonzero only when noise_background survives the round-trip.
    substrate = Substrate(np.array([[0, 0], [1, 1], [1, 1], [1, 1]]))
    obj = Transition(substrate, (1, 1), (1, 1), (0,), (1,), noise_background=True)
    restored = round_trip(obj, fmt)
    assert restored.noise_background is True
    assert restored == obj
    original_ratio = obj._ratio(Direction.EFFECT, (0,), (1,))
    assert restored._ratio(Direction.EFFECT, (0,), (1,)) == original_ratio
    assert original_ratio != 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_ac_sia_round_trips(fmt):
    sia = actual._null_ac_sia(make_transition(), Direction.CAUSE)
    restored = round_trip(sia, fmt)
    assert restored == sia
    assert len(restored.account) == 0
    assert len(restored.partitioned_account) == 0
