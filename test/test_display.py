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
