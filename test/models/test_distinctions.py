"""Tests for predicate selection on distinction bags."""

import pytest

from pyphi import examples
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.distinctions import UnresolvedDistinctions


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


def test_filter_selects_by_predicate(xor_ces):
    result = xor_ces.distinctions.filter(lambda d: len(d.mechanism) == 2)
    assert all(len(d.mechanism) == 2 for d in result)
    assert len(result) == sum(1 for d in xor_ces.distinctions if len(d.mechanism) == 2)


def test_filter_preserves_subtype(xor_ces):
    assert isinstance(xor_ces.distinctions, ResolvedDistinctions)
    result = xor_ces.distinctions.filter(lambda _d: True)
    assert type(result) is type(xor_ces.distinctions)


def test_filter_on_unresolved_preserves_subtype(xor_ces):
    unresolved = UnresolvedDistinctions(xor_ces.distinctions)
    result = unresolved.filter(lambda _d: True)
    assert type(result) is UnresolvedDistinctions


def test_filter_empty_result(xor_ces):
    result = xor_ces.distinctions.filter(lambda _d: False)
    assert len(result) == 0
    assert type(result) is type(xor_ces.distinctions)
