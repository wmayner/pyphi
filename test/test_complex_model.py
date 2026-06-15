from pyphi import jsonify
from pyphi.models.complex import ExcludedCandidate


def test_excluded_candidate_fields():
    e = ExcludedCandidate(node_indices=[1, 2], phi=0.5)
    assert e.node_indices == (1, 2)  # coerced to tuple
    assert e.phi == 0.5
    assert isinstance(e.phi, float)


def test_excluded_candidate_equality_precision_aware():
    a = ExcludedCandidate((1, 2), 0.5)
    b = ExcludedCandidate((1, 2), 0.5 + 1e-15)
    c = ExcludedCandidate((0, 2), 0.5)
    assert a == b  # phi compared up to PRECISION
    assert a != c  # different units


def test_excluded_candidate_hashable_by_units():
    a = ExcludedCandidate((1, 2), 0.5)
    b = ExcludedCandidate((1, 2), 0.5)
    assert hash(a) == hash(b)
    assert len({a, b}) == 1  # identical records collapse
    # Same units, different phi: hash collides (keyed by units) but the
    # phi-aware equality keeps them distinct.
    c = ExcludedCandidate((1, 2), 0.9)
    assert hash(a) == hash(c)
    assert a != c


def test_excluded_candidate_json_round_trip():
    e = ExcludedCandidate((1, 2), 0.5)
    decoded = jsonify.loads(jsonify.dumps(e))
    assert decoded == e
