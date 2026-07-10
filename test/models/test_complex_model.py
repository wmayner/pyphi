import pytest

from pyphi import examples
from pyphi import serialize
from pyphi import validate
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
    decoded = serialize.loads(serialize.dumps(e))
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
    decoded = serialize.loads(serialize.dumps(c))
    assert isinstance(decoded, Complex)
    assert decoded.node_indices == c.node_indices
    assert decoded.is_maximal is True
    assert {e.node_indices for e in decoded.excluded} == {(1, 2)}


def test_non_overlapping_accepts_disjoint():
    substrate, s = _basic_sia()
    a = Complex(sia=s, substrate=substrate)

    class _Stub:
        def __init__(self, idx):
            self.node_indices = idx

    disjoint = [a, _Stub((9,))]  # (0,1,2) vs (9,) — disjoint
    assert validate.non_overlapping(disjoint) is True


def test_non_overlapping_rejects_overlap():
    class _Stub:
        def __init__(self, idx):
            self.node_indices = idx

    overlapping = [_Stub((0, 1)), _Stub((1, 2))]  # share unit 1
    with pytest.raises(ValueError, match="Exclusion violated"):
        validate.non_overlapping(overlapping)


class _StubSIA:
    phi = 1.0
    node_indices = (0, 1)

    def order_by(self):
        return self.phi

    def _pandas_record(self):
        return {"phi": self.phi}


def test_complex_node_indices_override_and_units():
    units = ("unit-a", "unit-b")  # opaque to Complex; MacroUnits in practice
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        units=units,
        node_indices=(0, 1, 2, 3),
    )
    assert c.node_indices == (0, 1, 2, 3)
    assert c.units == units


def test_complex_defaults_micro():
    c = Complex(sia=_StubSIA(), substrate=None)
    assert c.node_indices == (0, 1)
    assert c.units is None


def test_excluded_candidate_units_default_none():
    e = ExcludedCandidate((1, 2), 0.5)
    assert e.units is None
    e2 = ExcludedCandidate((1, 2), 0.5, units=("u",))
    assert e2.units == ("u",)
    assert e2 == ExcludedCandidate((1, 2), 0.5)  # units not part of identity


def test_exclusion_margin_none_when_unopposed():
    c = Complex(sia=_StubSIA(), substrate=None)
    assert c.exclusion_margin is None
    assert c.effectively_tied is False


def test_exclusion_margin_is_gap_to_best_beaten_rival():
    c = Complex(
        sia=_StubSIA(),  # phi = 1.0
        substrate=None,
        excluded=(
            ExcludedCandidate((0,), 0.25),
            ExcludedCandidate((1,), 0.75),
        ),
    )
    assert c.exclusion_margin == pytest.approx(0.25)
    assert c.effectively_tied is False


def test_exclusion_margin_zero_for_precision_equal_rival():
    # A rival within PRECISION of the complex's own phi counts as beaten
    # (the selection was decided beyond phi) and clamps the margin to 0.
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((1, 2), 1.0 + 1e-15),),
    )
    assert c.exclusion_margin == 0.0
    assert c.effectively_tied is True


def test_exclusion_margin_ignores_shadows():
    # A higher-phi overlapping candidate (carved away by a different
    # complex under the recursive cascade) is not a beaten rival.
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((1, 2), 2.0),),
    )
    assert c.exclusion_margin is None
    assert c.effectively_tied is False


def test_exclusion_margin_mixed_shadows_and_rivals():
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(
            ExcludedCandidate((1, 2), 2.0),  # shadow
            ExcludedCandidate((0,), 0.4),  # beaten
        ),
    )
    assert c.exclusion_margin == pytest.approx(0.6)
    assert c.effectively_tied is False


def test_pandas_record_includes_margin_fields():
    c = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((0,), 0.4),),
    )
    record = c._pandas_record()
    assert record["exclusion_margin"] == pytest.approx(0.6)
    assert record["effectively_tied"] is False


def test_describe_margin_rows_present_only_when_margin_exists():
    with_margin = Complex(
        sia=_StubSIA(),
        substrate=None,
        excluded=(ExcludedCandidate((0,), 0.4),),
    )
    labels = [
        row.label
        for section in with_margin._describe(verbosity=2).sections
        for row in section.rows
    ]
    assert "Selection margin" in labels
    assert "Effectively tied" in labels

    without = Complex(sia=_StubSIA(), substrate=None)
    labels = [
        row.label
        for section in without._describe(verbosity=2).sections
        for row in section.rows
    ]
    assert "Selection margin" not in labels
    assert "Effectively tied" not in labels
