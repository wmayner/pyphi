import pytest

from pyphi.registry import Registry


def test_registry():
    registry = Registry()

    assert "DIFF" not in registry
    assert len(registry) == 0

    @registry.register("DIFF")
    def difference(a, b):
        return a - b

    assert "DIFF" in registry
    assert len(registry) == 1
    assert registry["DIFF"] == difference

    with pytest.raises(KeyError):
        registry["HEIGHT"]
