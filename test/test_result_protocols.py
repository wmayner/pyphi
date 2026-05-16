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


def test_fmt_sia_columns_shared_keys(s):
    """fmt_sia_columns returns the same shared columns for both formalism SIAs."""
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3
    from pyphi.models.fmt import fmt_sia_columns

    with config.override(**presets.iit3):
        cols_3 = dict(fmt_sia_columns(iit3.sia(s)))

    cols_4 = dict(fmt_sia_columns(FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s)))

    shared = {"System", "Current state", "Partition"}
    assert shared.issubset(cols_3.keys())
    assert shared.issubset(cols_4.keys())


def test_iit3_and_iit4_sia_repr_share_columns(s):
    """The shared SIA columns appear in both formalism reprs identically."""
    from pyphi.formalism import FORMALISM_REGISTRY
    from pyphi.formalism import iit3

    with config.override(**presets.iit3):
        repr_3 = repr(iit3.sia(s))

    repr_4 = repr(FORMALISM_REGISTRY["IIT_4_0_2023"].evaluate_system(s))

    # Both reprs must contain the shared column labels
    for label in ("System", "Current state", "Partition"):
        assert label in repr_3, f"{label!r} missing from IIT 3.0 SIA repr"
        assert label in repr_4, f"{label!r} missing from IIT 4.0 SIA repr"
