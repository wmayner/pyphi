"""Pickle-roundtrip tests for frozen formalism dataclasses."""

from __future__ import annotations

import pickle

import pytest

from pyphi.conf import config
from pyphi.formalism.iit3.formalism import IIT3Formalism
from pyphi.formalism.iit4.formalism import IIT4_2023Formalism
from pyphi.formalism.iit4.formalism import IIT4_2026Formalism


def test_iit3_formalism_is_frozen_dataclass():
    f = IIT3Formalism()
    assert hasattr(f, "config")
    with pytest.raises(Exception, match=r"cannot assign|frozen"):
        f.config = config.formalism  # type: ignore[misc]


def test_iit3_formalism_pickle_roundtrip():
    f = IIT3Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert f2.config == f.config
    assert f2.name == "IIT_3_0"


def test_iit3_formalism_carries_independent_config():
    """Snapshot taken at construction; later global changes don't leak in."""
    f = IIT3Formalism()
    captured = f.config.iit.repertoire_measure
    with config.override(repertoire_measure="L1"):
        assert f.config.iit.repertoire_measure == captured


def test_iit4_2023_formalism_is_frozen_dataclass():
    f = IIT4_2023Formalism()
    assert hasattr(f, "config")
    with pytest.raises(Exception, match=r"cannot assign|frozen"):
        f.config = config.formalism  # type: ignore[misc]


def test_iit4_2023_formalism_pickle_roundtrip():
    f = IIT4_2023Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert f2.config == f.config
    assert f2.name == "IIT_4_0_2023"


def test_iit4_2026_formalism_is_frozen_dataclass():
    f = IIT4_2026Formalism()
    assert hasattr(f, "config")
    with pytest.raises(Exception, match=r"cannot assign|frozen"):
        f.config = config.formalism  # type: ignore[misc]


def test_iit4_2026_formalism_pickle_roundtrip():
    f = IIT4_2026Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert f2.config == f.config
    assert f2.name == "IIT_4_0_2026"


def test_iit4_formalisms_carry_independent_config():
    f = IIT4_2023Formalism()
    captured = f.config.iit.repertoire_measure
    with config.override(repertoire_measure="L1"):
        assert f.config.iit.repertoire_measure == captured
