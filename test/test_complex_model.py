from pyphi import examples
from pyphi import jsonify
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.complex import Complex
from pyphi.models.complex import ExcludedCandidate
from pyphi.substrate import irreducible_sias


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


def _basic_sia():
    """Return (substrate, a real irreducible SIA) under IIT 4.0 defaults."""
    substrate = examples.basic_substrate()
    sias = irreducible_sias(substrate, (1, 0, 0))
    return substrate, sias[0]


def test_complex_delegates_node_indices_and_phi():
    substrate, s = _basic_sia()
    c = Complex(sia=s, substrate=substrate, is_maximal=True)
    assert c.node_indices == s.node_indices
    assert float(c.phi) == float(s.phi)
    assert c.sia is s
    assert c.substrate is substrate
    assert c.is_maximal is True
    assert c.excluded == ()


def test_complex_is_truthy_when_phi_positive():
    substrate, s = _basic_sia()
    c = Complex(sia=s, substrate=substrate, is_maximal=True)
    assert bool(c) is True


def test_complex_null_object_is_falsy_with_empty_units():
    substrate = examples.basic_substrate()
    null = Complex(
        sia=NullSystemIrreducibilityAnalysis(),
        substrate=substrate,
        is_maximal=True,
    )
    assert bool(null) is False
    assert null.node_indices == ()  # None normalized to ()
    assert float(null.phi) == 0.0


def test_complex_orders_by_phi():
    substrate, s = _basic_sia()
    big = Complex(sia=s, substrate=substrate)
    null = Complex(sia=NullSystemIrreducibilityAnalysis(), substrate=substrate)
    assert null < big
    assert max([null, big]) is big


def test_complex_json_round_trip():
    substrate, s = _basic_sia()
    c = Complex(
        sia=s,
        substrate=substrate,
        is_maximal=True,
        excluded=(ExcludedCandidate((1, 2), 0.5),),
    )
    decoded = jsonify.loads(jsonify.dumps(c))
    assert isinstance(decoded, Complex)
    assert decoded.node_indices == c.node_indices
    assert decoded.is_maximal is True
    assert {e.node_indices for e in decoded.excluded} == {(1, 2)}
