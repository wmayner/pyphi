"""Tests for the unified object-display model (pyphi.display)."""

import dataclasses

import numpy as np
import pytest

from pyphi.display.description import Description
from pyphi.display.description import Inline
from pyphi.display.description import Nested
from pyphi.display.description import Row
from pyphi.display.description import Section
from pyphi.display.description import Table
from pyphi.display.numbers import format_value
from pyphi.display.render import ascii as ascii_backend


def test_format_value_rounds_floats_to_6_sig_figs():
    assert format_value(0.41503749927884376) == "0.415037"


def test_format_value_handles_numpy_scalars():
    assert format_value(np.float64(3.0)) == "3"


def test_format_value_passes_through_non_numbers():
    assert format_value((1, 0, 0)) == "(1, 0, 0)"
    assert format_value(None) == "None"
    assert format_value("A,B,C") == "A,B,C"


def test_description_is_frozen_and_composes():
    d = Description(
        title="Demo",
        subtitle="x 1",
        sections=(
            Section(label=None, rows=(Row("System", "A,B,C"),)),
            Section(label="Cause", rows=(Row("purview", "(1,1,0)", (("II_c", 3.0),)),)),
            Section(label="List", body=(Table(headers=("a", "b"), rows=(("1", "2"),)),)),
        ),
        compact="Demo(x=1)",
    )
    assert d.title == "Demo"
    assert d.sections[1].rows[0].extra == (("II_c", 3.0),)
    assert d.sections[2].body[0].headers == ("a", "b")
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.title = "no"  # type: ignore[misc]


def test_nested_and_inline_are_components():
    inner = Description(title="Inner", compact="Inner()")
    s = Section(label=None, body=(Nested(inner), Inline(text="x ─── y")))
    assert isinstance(s.body[0], Nested)
    assert s.body[1].text == "x ─── y"


def test_ascii_visual_len_counts_codepoints():
    assert ascii_backend._vis_len("φ_s") == 3
    assert ascii_backend._vis_len("A,B,C") == 5


def test_ascii_pad_right():
    assert ascii_backend._pad("ab", 5) == "ab   "
    assert ascii_backend._pad("abcde", 3) == "abcde"


def test_ascii_framed_line_has_borders_and_width():
    line = ascii_backend._framed("hi", 6)
    assert line.startswith("│ ") and line.endswith(" │")
    assert ascii_backend._vis_len(line) == 6 + 4  # content width + "│ " + " │"


def test_ascii_format_rows_aligns_labels():
    rows = (Row("System", "A,B,C"), Row("φ_s", 0.41503749927884376))
    lines = ascii_backend._format_rows(rows)
    assert lines[0] == "System   A,B,C"
    assert lines[1] == "φ_s      0.415037"


def test_ascii_row_extra_fields_appended():
    rows = (Row("purview", "(1,1,0)", (("II_c", 3.0), ("int.diff", 0.0))),)
    lines = ascii_backend._format_rows(rows)
    assert lines[0] == "purview   (1,1,0)   II_c 3   int.diff 0"


def test_ascii_format_table_aligns_columns():
    t = Table(
        headers=("dir", "purview", "α"),  # noqa: RUF001
        rows=(("CAUSE", "OR", 0.415037), ("EFFECT", "AND", 0.415037)),
    )
    lines = ascii_backend._format_table(t)
    assert lines[0] == "dir      purview   α"  # noqa: RUF001
    assert lines[1] == "CAUSE    OR        0.415037"
    assert lines[2] == "EFFECT   AND       0.415037"
