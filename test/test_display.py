"""Tests for the unified object-display model (pyphi.display)."""

import dataclasses

import numpy as np
import pytest

import pyphi
from pyphi.display import Displayable
from pyphi.display import render as render_pkg
from pyphi.display.description import Description
from pyphi.display.description import Inline
from pyphi.display.description import Nested
from pyphi.display.description import Row
from pyphi.display.description import Section
from pyphi.display.description import Table
from pyphi.display.numbers import format_value
from pyphi.display.render import ascii as ascii_backend
from pyphi.display.render import html as html_backend


def test_format_value_rounds_floats_to_6_sig_figs():
    assert format_value(0.41503749927884376) == "0.415037"


def test_format_value_handles_numpy_scalars():
    assert format_value(np.float64(3.0)) == "3.0"


def test_format_value_whole_floats_keep_decimal():
    assert format_value(0.0) == "0.0"
    assert format_value(3.0) == "3.0"
    assert format_value(0.41503749927884376) == "0.415037"


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
    assert lines[0] == "purview   (1,1,0)   II_c 3.0   int.diff 0.0"


def test_ascii_format_table_aligns_columns():
    t = Table(
        headers=("dir", "purview", "α"),  # noqa: RUF001
        rows=(("CAUSE", "OR", 0.415037), ("EFFECT", "AND", 0.415037)),
    )
    lines = ascii_backend._format_table(t)
    assert lines[0] == "dir      purview   α"  # noqa: RUF001
    assert lines[1] == "CAUSE    OR        0.415037"
    assert lines[2] == "EFFECT   AND       0.415037"


def test_ascii_render_full_card():
    d = Description(
        title="Demo",
        subtitle="φ_s 0.415037",
        sections=(
            Section(label=None, rows=(Row("System", "A,B,C"),)),
            Section(label="Cause", rows=(Row("purview", "(1,1,0)", (("II_c", 3.0),)),)),
            Section(label="Links", body=(Table(("a", "b"), (("1", "2"),)),)),
            Section(label="MIP", body=(Inline("{A,BC}"),)),
        ),
    )
    out = ascii_backend.render(d, verbosity=2)
    lines = out.splitlines()
    assert lines[0].startswith("╭─ Demo ") and lines[0].endswith("╮")
    assert lines[-1].startswith("╰") and lines[-1].endswith("╯")
    widths = {ascii_backend._vis_len(line) for line in lines}
    assert len(widths) == 1  # all lines equal width
    assert "├─ Cause " in out
    assert "II_c 3" in out


def test_html_render_has_scoped_panel_and_escapes():
    d = Description(
        title="Demo<>",
        subtitle="φ_s 0.415037",
        sections=(Section(label="Cause", rows=(Row("purview", "(1,1,0) & <x>"),)),),
    )
    out = html_backend.render(d, verbosity=2)
    assert 'class="pyphi-card"' in out
    assert "pyphi-section" in out
    assert "Demo&lt;&gt;" in out  # title escaped
    assert "&lt;x&gt;" in out  # value escaped
    assert "<style" in out  # style block present


def test_html_style_injected_once_per_render():
    out = html_backend.render(Description(title="A"), verbosity=2)
    assert out.count("<style") == 1


def test_render_dispatches_by_backend_name():
    d = Description(title="Demo")
    assert render_pkg.render(d, backend="ascii", verbosity=2).startswith("╭")
    assert 'class="pyphi-card"' in render_pkg.render(d, backend="html", verbosity=2)


def test_render_unknown_backend_raises():
    with pytest.raises(KeyError):
        render_pkg.render(Description(title="x"), backend="rich", verbosity=2)


class _Demo(Displayable):
    def _describe(self, verbosity):  # noqa: ARG002
        return Description(
            title="Demo",
            subtitle="x 1",
            sections=(Section(label=None, rows=(Row("x", 1),)),),
            compact="Demo(x=1)",
        )


def test_displayable_repr_and_str_match_ascii_card():
    obj = _Demo()
    assert repr(obj) == str(obj)
    assert repr(obj).startswith("╭─ Demo")


def test_displayable_low_verbosity_uses_compact():
    obj = _Demo()
    with pyphi.config.override(repr_verbosity=0):
        assert repr(obj) == "Demo(x=1)"


def test_displayable_html_and_mimebundle():
    obj = _Demo()
    assert 'class="pyphi-card"' in obj._repr_html_()
    bundle = obj._repr_mimebundle_()
    assert set(bundle) == {"text/plain", "text/html"}


def test_iit4_sia_describe_structure():
    s = pyphi.examples.basic_system()
    sia = s.sia()
    d = sia._describe(2)
    assert d.title == "SystemIrreducibilityAnalysis"
    labels = [r.label for sec in d.sections for r in sec.rows]
    assert "System" in labels
    assert "φ_s" in (d.subtitle or "")
    section_labels = [sec.label for sec in d.sections]
    assert "Cause" in section_labels and "Effect" in section_labels
    out = repr(sia)
    assert out.startswith("╭─ SystemIrreducibilityAnalysis")
    assert "0.41503749927884376" not in out  # numbers are rounded
