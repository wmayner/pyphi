import numpy as np
import pytest

from pyphi import Direction
from pyphi import config
from pyphi import exceptions
from pyphi.substrate import Substrate


@pytest.fixture()
def substrate():
    size = 3
    tpm = np.ones([2] * size + [size]).astype(float) / 2
    return Substrate(tpm)


def test_substrate_init_validation(substrate):
    with pytest.raises(ValueError):
        # Totally wrong shape
        tpm = np.arange(3).astype(float)
        Substrate(tpm)
    with pytest.raises(ValueError):
        # Non-binary nodes (4 states)
        tpm = np.ones((4, 4, 4, 3)).astype(float)
        Substrate(tpm)

    # Conditionally dependent
    # fmt: off
    tpm = np.array([
            [1, 0.0, 0.0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.0, 0.0, 1],
    ])
    # fmt: on
    with config.override(validate_conditional_independence=False):
        Substrate(tpm)
    with (
        config.override(validate_conditional_independence=True),
        pytest.raises(exceptions.ConditionallyDependentError),
    ):
        Substrate(tpm)


def test_substrate_creates_fully_connected_cm_by_default():
    tpm = np.zeros((2 * 2 * 2, 3))
    substrate = Substrate(tpm, cm=None)
    target_cm = np.ones((3, 3))
    assert np.array_equal(substrate.cm, target_cm)


def test_potential_purviews(s):
    mechanism = (0,)
    assert s.substrate.potential_purviews(Direction.CAUSE, mechanism) == [
        (1,),
        (2,),
        (1, 2),
    ]
    assert s.substrate.potential_purviews(Direction.EFFECT, mechanism) == [(2,)]


def test_node_labels(standard):
    labels = ("A", "B", "C")
    substrate = Substrate(standard.tpm.tpm, node_labels=labels)
    assert substrate.node_labels.labels == labels

    labels = ("A", "B")  # Too few labels
    with pytest.raises(ValueError):
        Substrate(standard.tpm.tpm, node_labels=labels)

    # Auto-generated labels
    substrate = Substrate(standard.tpm.tpm, node_labels=None)
    assert substrate.node_labels.labels == ("n0", "n1", "n2")


def test_num_states(standard):
    assert standard.num_states == 8


def test_repr(standard):
    print(repr(standard))


def test_str(standard):
    print(str(standard))


def test_len(standard):
    assert len(standard) == 3


def test_size(standard):
    assert standard.size == 3
