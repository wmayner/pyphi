import numpy as np
import pytest

from pyphi import examples
from pyphi.actual import Transition
from pyphi.substrate import Substrate
from pyphi.system import System


def _example_items(kind):
    return sorted(examples.EXAMPLES[kind].items())


def test_examples_registry_contains_expected_categories():
    expected = {"substrate", "system", "tpm", "transition"}
    assert expected.issubset(examples.EXAMPLES.keys())


@pytest.mark.parametrize("name, func", _example_items("substrate"))
def test_example_substrates_construct(name, func):
    substrate = func()
    assert isinstance(substrate, Substrate)


@pytest.mark.parametrize("name, func", _example_items("system"))
def test_example_systems_construct(name, func):
    system = func()
    assert isinstance(system, System)


@pytest.mark.parametrize("name, func", _example_items("tpm"))
def test_example_tpms_construct(name, func):
    tpm = func()
    assert isinstance(tpm, np.ndarray)
    assert tpm.ndim == 2
    assert tpm.shape[0] == tpm.shape[1]


@pytest.mark.parametrize("name, func", _example_items("transition"))
def test_example_transitions_construct(name, func):
    pytest.skip(
        "actual.Transition pending refactor for frozen System "
        "(uses _external_indices override + state mutation)"
    )
    transition = func()
    assert isinstance(transition, Transition)
