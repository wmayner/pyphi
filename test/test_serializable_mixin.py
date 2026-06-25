import pytest

from pyphi import examples
from pyphi.substrate import Substrate


def test_instance_save_and_classmethod_load(tmp_path):
    sub = examples.basic_substrate()
    path = tmp_path / "sub.json"
    sub.save(path)
    assert Substrate.load(path) == sub


def test_load_typechecks(tmp_path):
    # A file holding a different type must not load as a Substrate.
    from pyphi import serialize

    sia = examples.basic_system().sia()
    path = tmp_path / "sia.json"
    serialize.save(sia, path)
    with pytest.raises(TypeError):
        Substrate.load(path)


def _all_result_objects():
    from pyphi import actual
    from pyphi import examples
    from pyphi.models.complex import Complex

    system = examples.basic_system()
    sia = system.sia()
    ces = system.ces()
    transition = examples.prevention_transition()
    return {
        "system": system,
        "transition": transition,
        "sia": sia,
        "ces": ces,
        "account": actual.account(transition),
        "acsia": actual.sia(transition),
        "complex": Complex(
            sia=sia, substrate=system.substrate, is_maximal=True, excluded=()
        ),
        "distinctions": ces.distinctions,
        "relations": ces.relations,
    }


@pytest.mark.parametrize(
    "name",
    [
        "system",
        "transition",
        "sia",
        "ces",
        "account",
        "acsia",
        "complex",
        "distinctions",
        "relations",
    ],
)
def test_every_user_facing_type_round_trips_via_method(tmp_path, name):
    obj = _all_result_objects()[name]
    path = tmp_path / f"{name}.json"
    obj.save(path)
    restored = type(obj).load(path)
    assert restored == obj
