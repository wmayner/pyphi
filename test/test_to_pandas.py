"""Tests for the unified to_pandas labeled-export convention."""

import numpy as np
import pandas as pd
import pytest

from pyphi import examples
from pyphi.direction import Direction
from pyphi.labels import NodeLabels
from pyphi.models.pandas import distribution_rows
from pyphi.models.pandas import record_to_series
from pyphi.models.pandas import records_to_frame
from pyphi.models.pandas import state_multiindex
from pyphi.utils import all_states


def test_state_multiindex_binary():
    labels = NodeLabels(("A", "B"), (0, 1))
    mi = state_multiindex(labels, (0, 1))
    assert list(mi.names) == ["A", "B"]
    assert list(mi) == list(all_states(2))


def test_state_multiindex_kary():
    labels = NodeLabels(("A", "B"), (0, 1))
    mi = state_multiindex(labels, (0, 1), alphabet=(3, 2))
    assert list(mi) == list(all_states((3, 2)))


def test_distribution_rows_binary():
    # shape (2, 2) over purview (0, 1); flatten is little-endian (order="F")
    rep = np.array([[0.1, 0.4], [0.1, 0.4]])
    rows = distribution_rows(Direction.CAUSE, "repertoire", (0, 1), rep, None)
    assert [r["state"] for r in rows] == list(all_states((2, 2)))
    assert rows[0] == {
        "direction": "CAUSE",
        "kind": "repertoire",
        "purview": (0, 1),
        "state": (0, 0),
        "probability": 0.1,
    }
    assert sum(r["probability"] for r in rows) == pytest.approx(1.0)


def test_distribution_rows_kary_states():
    rep = np.arange(6, dtype=float).reshape(3, 2)
    rep = rep / rep.sum()
    rows = distribution_rows(Direction.EFFECT, "repertoire", (0, 1), rep, None)
    assert [r["state"] for r in rows] == list(all_states((3, 2)))
    assert all(r["direction"] == "EFFECT" for r in rows)


def test_distribution_rows_none_is_empty():
    assert distribution_rows(Direction.CAUSE, "repertoire", (0,), None, None) == []


def test_record_to_series_preserves_order():
    series = record_to_series({"phi": 0.5, "mechanism": ("A",)}, name="X")
    assert list(series.index) == ["phi", "mechanism"]
    assert series.name == "X"


def test_records_to_frame_empty_has_columns():
    frame = records_to_frame([], index="mechanism", columns=["mechanism", "phi"])
    assert frame.index.name == "mechanism"
    assert list(frame.columns) == ["phi"]


@pytest.fixture(scope="module")
def basic_ces():
    return examples.basic_system().ces()


def test_distinction_to_pandas_is_labeled_series(basic_ces):
    distinction = next(iter(basic_ces.distinctions))
    series = distinction.to_pandas()
    assert isinstance(series, pd.Series)
    assert {
        "phi",
        "mechanism",
        "mechanism_state",
        "cause_purview",
        "effect_purview",
    } <= set(series.index)
    # units render as label strings, never raw ints
    assert all(isinstance(label, str) for label in series["mechanism"])


def test_mice_to_pandas_is_labeled_series(basic_ces):
    mice = next(iter(basic_ces.distinctions)).cause
    series = mice.to_pandas()
    assert isinstance(series, pd.Series)
    assert all(isinstance(label, str) for label in series["mechanism"])
    assert str(series["direction"]) == "CAUSE"


def test_ria_to_pandas_is_labeled_series(basic_ces):
    ria = next(iter(basic_ces.distinctions)).cause.ria
    series = ria.to_pandas()
    assert isinstance(series, pd.Series)
    assert all(isinstance(label, str) for label in series["purview"])
    assert series["direction"] in ("CAUSE", "EFFECT")
