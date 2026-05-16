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
