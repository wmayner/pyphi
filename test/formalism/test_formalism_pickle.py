"""Pickle-roundtrip tests for frozen formalism dataclasses."""

from __future__ import annotations

import pickle

from pyphi.formalism.actual_causation.formalism import AC2019Formalism
from pyphi.formalism.iit3.formalism import IIT3Formalism
from pyphi.formalism.iit4.formalism import IIT4_2023Formalism
from pyphi.formalism.iit4.formalism import IIT4_2026Formalism


def test_iit3_formalism_pickle_roundtrip():
    f = IIT3Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert type(f2) is IIT3Formalism
    assert f2.name == "IIT_3_0"


def test_iit4_2023_formalism_pickle_roundtrip():
    f = IIT4_2023Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert type(f2) is IIT4_2023Formalism
    assert f2.name == "IIT_4_0_2023"


def test_iit4_2026_formalism_pickle_roundtrip():
    f = IIT4_2026Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert type(f2) is IIT4_2026Formalism
    assert f2.name == "IIT_4_0_2026"


def test_ac_formalism_pickle_roundtrip():
    f = AC2019Formalism()
    f2 = pickle.loads(pickle.dumps(f))
    assert type(f2) is AC2019Formalism
