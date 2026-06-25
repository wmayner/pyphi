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
