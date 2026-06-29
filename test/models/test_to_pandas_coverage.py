"""N9: every Displayable result type exports to a labeled pandas structure."""

from __future__ import annotations

import pandas as pd

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets


def test_sia_to_pandas_series_has_phi():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    s = sia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)


def test_iit3_sia_to_pandas_series_has_phi():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit3):
        sia = substrate.sia(state)
    s = sia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)


def test_complex_and_excluded_to_pandas_series():
    from pyphi.models.complex import Complex
    from pyphi.models.complex import ExcludedCandidate

    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    cx = Complex(sia=sia, substrate=substrate, is_maximal=True, excluded=())
    s = cx.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)
    assert bool(s["is_maximal"]) is True

    ec = ExcludedCandidate(node_indices=(0, 1), phi=0.0)
    es = ec.to_pandas()
    assert isinstance(es, pd.Series)
    assert tuple(es["node_indices"]) == (0, 1)
