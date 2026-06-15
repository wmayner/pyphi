"""Tests for the unified object-display model (pyphi.display)."""

import numpy as np

from pyphi.display.numbers import format_value


def test_format_value_rounds_floats_to_6_sig_figs():
    assert format_value(0.41503749927884376) == "0.415037"


def test_format_value_handles_numpy_scalars():
    assert format_value(np.float64(3.0)) == "3"


def test_format_value_passes_through_non_numbers():
    assert format_value((1, 0, 0)) == "(1, 0, 0)"
    assert format_value(None) == "None"
    assert format_value("A,B,C") == "A,B,C"
