import pytest

from pyphi.labels import NodeLabels


@pytest.fixture
def nl():
    return NodeLabels(('A', 'B', 'C'), (0, 1, 2))


def test_defaults():
    nd = NodeLabels(None, (0, 1, 2))
    assert nd.labels == ('n0', 'n1', 'n2')


def test_labels2indices(nl):
    assert nl.labels2indices(('A', 'B')) == (0, 1)
    assert nl.labels2indices(('A', 'C')) == (0, 2)


def test_indices2labels(nl):
    assert nl.indices2labels((0, 1)) == ('A', 'B')
    assert nl.indices2labels((0, 2)) == ('A', 'C')

    # assert network.indices2labels((0, 1)) == ('n0', 'n1')
    # assert network.indices2labels((0, 2)) == ('n0', 'n2')


def test_coerce_to_indices(nl):
    assert nl.coerce_to_indices(('B', 'A')) == (0, 1)
    assert nl.coerce_to_indices((0, 2, 1)) == (0, 1, 2)
    assert nl.coerce_to_indices(()) == ()

    with pytest.raises(ValueError):
        nl.coerce_to_indices((0, 'A'))


def test_iterable(nl):
    assert [l for l in nl] == ['A', 'B', 'C']


def test_len(nl):
    assert len(nl) == 3


def test_contains(nl):
    assert 'B' in nl
    assert 'D' not in nl


def test_instantiation_from_other_node_labels_object(nl):
    copied = NodeLabels(nl, (0, 1, 2))
    assert copied == nl
