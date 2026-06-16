"""Cross-formalism Protocol conformance tests."""

from __future__ import annotations

from pyphi import config
from pyphi.conf import presets


def test_iit3_sia_satisfies_sia_interface(s):
    from pyphi.formalism import iit3
    from pyphi.models.protocols import SIAInterface

    with config.override(**presets.iit3):
        sia = iit3.sia(s)
    assert isinstance(sia, SIAInterface)


def test_iit4_sia_satisfies_sia_interface(s):
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.models.protocols import SIAInterface

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    assert isinstance(sia, SIAInterface)


def test_iit3_ces_satisfies_ces_interface(s):
    from pyphi.formalism import iit3
    from pyphi.models.protocols import CauseEffectStructureInterface

    with config.override(**presets.iit3):
        ces = iit3.ces(s)
    assert isinstance(ces, CauseEffectStructureInterface)


def test_acsia_satisfies_acsia_interface(transition):
    from pyphi import actual
    from pyphi.models.protocols import AcSIAInterface

    with config.override(**presets.iit3):
        acsia = actual.sia(transition)
    assert isinstance(acsia, AcSIAInterface)


def test_iit3_and_iit4_sia_repr_share_columns(s):
    """The shared SIA columns appear in both formalism reprs identically."""
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        repr_3 = repr(iit3.sia(s))

    repr_4 = repr(FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s))

    # Both reprs must contain the shared field labels
    for label in ("System", "Current state"):
        assert label in repr_3, f"{label!r} missing from IIT 3.0 SIA repr"
        assert label in repr_4, f"{label!r} missing from IIT 4.0 SIA repr"
    # Both surface the partition (label casing differs across formalisms until
    # both render through the unified display model).
    assert "partition" in repr_3.lower()
    assert "partition" in repr_4.lower()


def test_iit3_sia_repr_html(s):
    """IIT 3.0 SIA has a Jupyter HTML repr."""
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia = iit3.sia(s)
    html = sia._repr_html_()
    assert "<table" in html or "<div" in html


def test_iit4_sia_repr_html(s):
    from pyphi.formalism import FORMALISM_REGISTRY

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    html = sia._repr_html_()
    assert "<table" in html or "<div" in html


def test_iit3_ces_repr_html(s):
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        ces = iit3.ces(s)
    html = ces._repr_html_()
    assert "<table" in html or "<div" in html


def test_acsia_repr_html(transition):
    from pyphi import actual

    with config.override(**presets.iit3):
        acsia = actual.sia(transition)
    html = acsia._repr_html_()
    assert "<table" in html or "<div" in html


def test_iit3_sia_neq_iit4_sia(s):
    """Cross-formalism __eq__ returns False without exception."""
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia_3 = iit3.sia(s)
    sia_4 = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)

    # Must not raise; must return False (Python falls back from NotImplemented)
    assert (sia_3 == sia_4) is False
    assert (sia_4 == sia_3) is False
    assert sia_3 != sia_4


def test_iit3_sia_eq_via_notimplemented_on_unrelated(s):
    """Compare an IIT 3.0 SIA to an unrelated type — returns False, no raise."""
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        sia = iit3.sia(s)
    assert (sia == "not a sia") is False
    assert (sia == 42) is False


def test_iit3_ces_json_round_trip(s):
    """IIT 3.0 CES round-trips through jsonify with structural equality."""
    from pyphi import jsonify
    from pyphi.formalism import iit3
    from pyphi.models.ces import CauseEffectStructure

    with config.override(**presets.iit3):
        ces = iit3.ces(s)

    encoded = jsonify.dumps(ces)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, CauseEffectStructure)
    assert decoded.sia.phi == ces.sia.phi
    assert len(decoded.distinctions) == len(ces.distinctions)
    assert decoded.relations.num_relations() == ces.relations.num_relations()


def test_iit3_sia_json_round_trip(s):
    """IIT 3.0 SIA round-trips through jsonify."""
    from pyphi import jsonify
    from pyphi.formalism import iit3
    from pyphi.models.sia import IIT3SystemIrreducibilityAnalysis

    with config.override(**presets.iit3):
        sia = iit3.sia(s)

    encoded = jsonify.dumps(sia)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, IIT3SystemIrreducibilityAnalysis)
    assert decoded.phi == sia.phi
    assert decoded.partition == sia.partition
    assert decoded.partitioned_distinctions == sia.partitioned_distinctions


def test_iit4_sia_json_round_trip(s):
    """IIT 4.0 SIA round-trips through jsonify."""
    from pyphi import jsonify
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis

    sia = FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)
    encoded = jsonify.dumps(sia)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, SystemIrreducibilityAnalysis)
    assert decoded == sia


def test_acsia_json_round_trip(transition):
    """AcSIA round-trips through jsonify."""
    from pyphi import actual
    from pyphi import jsonify
    from pyphi.models.actual_causation import AcSystemIrreducibilityAnalysis

    with config.override(**presets.iit3):
        acsia = actual.sia(transition)

    encoded = jsonify.dumps(acsia)
    decoded = jsonify.loads(encoded)

    assert isinstance(decoded, AcSystemIrreducibilityAnalysis)
    assert decoded.alpha == acsia.alpha
