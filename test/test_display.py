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
    d = Description(
        title="Demo",
        sections=(Section(label=None, rows=(Row("x", 1),)),),
    )
    assert render_pkg.render(d, backend="ascii", verbosity=2).startswith("╭")
    assert 'class="pyphi-card"' in render_pkg.render(d, backend="html", verbosity=2)


def test_leaf_ascii_render_is_compact_no_box():
    d = Description(title="X", compact="2 parts {A,BC}")
    out = ascii_backend.render(d, verbosity=2)
    assert out == "2 parts {A,BC}"
    assert "╭" not in out


def test_leaf_html_render_has_leaf_not_card():
    d = Description(title="X", compact="2 parts {A,BC}")
    out = html_backend.render(d, verbosity=2)
    assert 'class="pyphi-leaf"' in out
    assert 'class="pyphi-card"' not in out


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
    assert "φ_s" in labels
    section_labels = [sec.label for sec in d.sections]
    assert "Cause" in section_labels and "Effect" in section_labels
    out = repr(sia)
    assert out.startswith("╭─ SystemIrreducibilityAnalysis")
    assert "0.41503749927884376" not in out  # numbers are rounded


def test_iit3_sia_describe_structure():
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    pyphi.config.progress_bars = False
    with pyphi.config.override(**presets.iit3):
        s = pyphi.examples.basic_system()
        sia = iit3.sia(s)

    d = sia._describe(2)
    assert d.title == "IIT3SystemIrreducibilityAnalysis"
    labels = [r.label for sec in d.sections for r in sec.rows]
    assert "Φ" in labels
    assert "System" in labels
    assert "Current state" in labels
    assert "Partition" in labels
    out = repr(sia)
    assert out.startswith("╭")
    assert "pyphi-card" in sia._repr_html_()


def test_iit3_sia_low_verbosity_compact():
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    pyphi.config.progress_bars = False
    with pyphi.config.override(**presets.iit3):
        s = pyphi.examples.basic_system()
        sia = iit3.sia(s)

    with pyphi.config.override(repr_verbosity=0):
        out = repr(sia)
    assert out.startswith("IIT3SystemIrreducibilityAnalysis(Φ=")


def test_ces_describe_structure():
    import pyphi
    from pyphi.conf import presets
    from pyphi.formalism import iit3

    pyphi.config.progress_bars = False
    with pyphi.config.override(**presets.iit3):
        s = pyphi.examples.basic_system()
        ces = iit3.ces(s)

    d = ces._describe(2)
    assert d.title == "CauseEffectStructure"
    section_labels = [sec.label for sec in d.sections]
    assert "Distinctions" in section_labels
    top_row_labels = [r.label for r in d.sections[0].rows]
    assert "Φ" in top_row_labels
    assert "Distinctions" in top_row_labels
    assert "Σφ_d" in top_row_labels
    assert "Relations" in top_row_labels
    assert "Σφ_r" in top_row_labels
    out = repr(ces)
    assert out.startswith("╭")
    assert "pyphi-card" in ces._repr_html_()


def test_ces_distinctions_table_has_headers():
    import pyphi
    from pyphi.conf import presets
    from pyphi.display.description import Table
    from pyphi.formalism import iit3

    pyphi.config.progress_bars = False
    with pyphi.config.override(**presets.iit3):
        s = pyphi.examples.basic_system()
        ces = iit3.ces(s)

    d = ces._describe(2)
    distinctions_section = next(sec for sec in d.sections if sec.label == "Distinctions")
    assert len(distinctions_section.body) == 1
    table = distinctions_section.body[0]
    assert isinstance(table, Table)
    assert table.headers == ("Mechanism", "φ_d", "Cause purview", "Effect purview")
    assert len(table.rows) == len(ces.distinctions)


# ---------------------------------------------------------------------------
# Partition display (Steps 2 leaf path)
# ---------------------------------------------------------------------------


def test_partition_nullcut_card_and_concise():
    from pyphi import models
    from pyphi.models.partitions import concise_partition

    cut = models.NullCut((2, 3))
    out = repr(cut)
    assert out.startswith("╭─ NullCut")  # rich card now
    assert "Connections cut" in out
    assert concise_partition(cut) == "NullCut((2, 3))"  # concise embedding form
    assert 'class="pyphi-card"' in cut._repr_html_()


def test_partition_directed_bipartition_card_and_grid():
    from pyphi.direction import Direction
    from pyphi.models.partitions import DirectedBipartition
    from pyphi.models.partitions import concise_partition

    p = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))
    out = repr(p)
    assert out.startswith("╭─ DirectedBipartition · effect")
    assert "Severed connections" in out
    assert "✕" in out  # cut grid mark
    c = concise_partition(p)
    assert "[0]" in c and "[1,2]" in c
    assert "pyphi-effect" in p._repr_html_()  # directed partition is toned


def test_partition_joint_bipartition_card_and_concise():
    from pyphi.labels import NodeLabels
    from pyphi.models.partitions import JointBipartition
    from pyphi.models.partitions import Part
    from pyphi.models.partitions import concise_partition

    nl = NodeLabels("ABCDE", tuple(range(5)))
    jb = JointBipartition(Part((0,), (0, 4)), Part((), (1,)), node_labels=nl)
    out = repr(jb)
    assert out.startswith("╭─ JointBipartition")
    c = concise_partition(jb)
    assert "A/A,E" in c and "∅/B" in c
    assert 'class="pyphi-card"' in jb._repr_html_()


def test_cut_grid_marks_match_removed_edges():
    from pyphi.direction import Direction
    from pyphi.models.partitions import DirectedBipartition
    from pyphi.models.partitions import _cut_grid

    p = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))  # severs 0->1, 0->2
    grid = _cut_grid(p)
    assert grid.headers == ("", "0", "1", "2")
    row0 = next(r for r in grid.rows if r[0] == "0")
    assert row0 == ("0", "·", "✕", "✕")
    row1 = next(r for r in grid.rows if r[0] == "1")
    assert row1 == ("1", "·", "·", "·")


def test_sia_embeds_concise_partition_one_line():
    pyphi.config.progress_bars = False
    sia = pyphi.examples.basic_system().sia()
    part_lines = [ln for ln in repr(sia).splitlines() if "Partition" in ln]
    assert len(part_lines) == 1
    assert "╭" not in part_lines[0] and "╰" not in part_lines[0]


def test_sia_html_has_cause_effect_colors():
    pyphi.config.progress_bars = False
    h = pyphi.examples.basic_system().sia()._repr_html_()
    assert "#D55C00" in h and "#009E73" in h
    assert "pyphi-label pyphi-cause" in h
    assert "pyphi-label pyphi-effect" in h


# ---------------------------------------------------------------------------
# RIA display (Step 3)
# ---------------------------------------------------------------------------


def _basic_mic():
    pyphi.config.progress_bars = False
    s = pyphi.examples.basic_system()
    return s.distinctions()[0].cause


def _basic_ria():
    return _basic_mic().ria


def test_ria_describe_structure():
    ria = _basic_ria()
    d = ria._describe(2)
    assert d.title == "RepertoireIrreducibilityAnalysis"
    labels = [r.label for sec in d.sections for r in sec.rows]
    assert "φ" in labels
    assert "Direction" in labels
    assert "Mechanism" in labels
    assert "Purview" in labels
    assert "Partition" in labels


def test_ria_repr_is_card():
    ria = _basic_ria()
    out = repr(ria)
    assert out.startswith("╭─ RepertoireIrreducibilityAnalysis")
    assert "CAUSE" in out
    assert "0.41503" not in out  # numbers are rounded


def test_ria_html_is_card():
    assert 'class="pyphi-card"' in _basic_ria()._repr_html_()


def test_ria_low_verbosity_compact():
    ria = _basic_ria()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(ria)
    assert out.startswith("RepertoireIrreducibilityAnalysis(φ=")


# ---------------------------------------------------------------------------
# MICE display (Step 3)
# ---------------------------------------------------------------------------


def test_mice_describe_structure():
    mic = _basic_mic()
    d = mic._describe(2)
    assert d.title.startswith("MaximallyIrreducible")
    labels = [r.label for sec in d.sections for r in sec.rows]
    assert "φ" in labels
    assert "Purview ties" in labels


def test_mice_repr_is_card():
    mic = _basic_mic()
    out = repr(mic)
    assert out.startswith("╭─ MaximallyIrreducibleCause")


def test_mice_html_is_card():
    assert 'class="pyphi-card"' in _basic_mic()._repr_html_()


def test_mice_low_verbosity_compact():
    mic = _basic_mic()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(mic)
    assert out.startswith("MaximallyIrreducibleCause(φ=")


# ---------------------------------------------------------------------------
# Task 14: System, Substrate, Node, and state-spec display
# ---------------------------------------------------------------------------


def _basic_system():
    pyphi.config.progress_bars = False
    return pyphi.examples.basic_system()


def test_system_is_leaf():
    s = _basic_system()
    out = repr(s)
    assert out == "System(A, B, C)"
    assert "╭" not in out
    assert 'class="pyphi-leaf"' in s._repr_html_()
    assert 'class="pyphi-card"' not in s._repr_html_()


def test_system_str_equals_repr():
    s = _basic_system()
    assert str(s) == repr(s) == "System(A, B, C)"


def test_system_str_matches_expected_label_subset():
    """str(system) must produce e.g. 'System(B, C)' for a node subset."""
    from pyphi.substrate import Substrate
    from pyphi.system import System

    sub = pyphi.examples.basic_substrate()
    sub2 = Substrate(sub._legacy_binary_joint(), node_labels=("A", "B", "C"))
    sys = System(sub2, (0, 0, 0), ("B", "C"))
    assert str(sys) == "System(B, C)"


def test_substrate_is_leaf():
    s = _basic_system()
    sub = s.substrate
    out = repr(sub)
    assert out.startswith("Substrate(")
    assert "╭" not in out
    assert 'class="pyphi-leaf"' in sub._repr_html_()
    assert 'class="pyphi-card"' not in sub._repr_html_()


def test_node_is_leaf():
    s = _basic_system()
    node = s.nodes[0]
    out = repr(node)
    assert out == node.label
    assert "╭" not in out
    assert 'class="pyphi-leaf"' in node._repr_html_()
    assert 'class="pyphi-card"' not in node._repr_html_()


def test_unit_state_is_leaf():
    from pyphi.models.state_specification import UnitState

    us_on = UnitState(0, 1, "A")
    us_off = UnitState(0, 0, "A")
    assert repr(us_on) == "A"
    assert repr(us_off) == "a"
    assert 'class="pyphi-leaf"' in us_on._repr_html_()
    assert 'class="pyphi-card"' not in us_on._repr_html_()


def test_state_specification_is_card():
    s = _basic_system()
    sia = s.sia()
    spec = sia.system_state.cause
    out = repr(spec)
    assert "╭" in out
    assert "Direction" in out
    assert "Purview" in out
    assert "Specified state" in out
    assert 'class="pyphi-card"' in spec._repr_html_()


def test_system_state_specification_is_card():
    s = _basic_system()
    sia = s.sia()
    sys_state = sia.system_state
    out = repr(sys_state)
    assert "╭" in out
    assert "Cause" in out
    assert "Effect" in out
    assert "Purview" in out
    assert "Specified state" in out
    assert 'class="pyphi-card"' in sys_state._repr_html_()


# ---------------------------------------------------------------------------
# Task 15: Actual-causation and relations display
# ---------------------------------------------------------------------------


def _ac_transition():
    """Small OR-gate transition used throughout the AC display tests."""
    import numpy as np

    from pyphi import actual
    from pyphi.substrate import Substrate

    pyphi.config.progress_bars = False
    tpm = np.array(
        [
            [0, 0.5, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
        ]
    )
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    return actual.Transition(substrate, (0, 1, 1), (1, 0, 0), (1, 2), (0,))


def _ac_account_and_sia():
    from pyphi import actual
    from pyphi.conf import presets

    t = _ac_transition()
    with pyphi.config.override(**presets.iit3):
        acc = actual.account(t)
        acsia = actual.sia(t)
    return acc, acsia


def test_account_is_card():
    acc, _ = _ac_account_and_sia()
    out = repr(acc)
    assert out.startswith("╭")
    assert "Causal links" in out
    assert "Σα" in out


def test_account_html_is_card():
    acc, _ = _ac_account_and_sia()
    assert 'class="pyphi-card"' in acc._repr_html_()


def test_account_html_has_table():
    acc, _ = _ac_account_and_sia()
    html = acc._repr_html_()
    assert "<table" in html
    assert "Direction" in html
    assert "Mechanism" in html


def test_account_low_verbosity_compact():
    acc, _ = _ac_account_and_sia()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(acc)
    assert out.startswith("Account(")
    assert "links" in out


def test_acsia_is_card():
    _, acsia = _ac_account_and_sia()
    out = repr(acsia)
    assert out.startswith("╭")
    assert "α" in out  # noqa: RUF001
    assert "Direction" in out
    assert "System" in out


def test_acsia_html_is_card():
    _, acsia = _ac_account_and_sia()
    assert 'class="pyphi-card"' in acsia._repr_html_()


def test_acsia_low_verbosity_compact():
    _, acsia = _ac_account_and_sia()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(acsia)
    assert out.startswith("AcSystemIrreducibilityAnalysis(")


# ---------------------------------------------------------------------------
# Relations display
# ---------------------------------------------------------------------------


def _xor_relations():
    """XOR system with concrete relations (15 relations)."""
    pyphi.config.progress_bars = False
    s = pyphi.examples.xor_system()
    ces = s.ces()
    return ces.relations


def test_concrete_relations_is_card():
    rels = _xor_relations()
    out = repr(rels)
    assert out.startswith("╭")
    assert "Relations" in out
    assert "Σφ_r" in out


def test_concrete_relations_html_is_card():
    rels = _xor_relations()
    assert 'class="pyphi-card"' in rels._repr_html_()


def test_concrete_relations_html_has_table():
    rels = _xor_relations()
    html = rels._repr_html_()
    assert "<table" in html
    assert "φ_r" in html


def test_concrete_relations_table_has_header_row():
    rels = _xor_relations()
    out = repr(rels)
    assert "Relata (mechanisms)" in out
    assert "Degree" in out


def test_relation_is_card():
    rels = _xor_relations()
    r = next(iter(rels))
    out = repr(r)
    assert out.startswith("╭")
    assert "φ_r" in out
    assert "Degree" in out


def test_relation_html_is_card():
    rels = _xor_relations()
    r = next(iter(rels))
    assert 'class="pyphi-card"' in r._repr_html_()


def test_relation_low_verbosity_compact():
    rels = _xor_relations()
    r = next(iter(rels))
    with pyphi.config.override(repr_verbosity=0):
        out = repr(r)
    assert out.startswith("Relation(")


# ---------------------------------------------------------------------------
# B21: Distinction, Distinctions, Complex display
# ---------------------------------------------------------------------------


def _basic_distinctions():
    pyphi.config.progress_bars = False
    s = pyphi.examples.basic_system()
    return s.distinctions()


def _basic_distinction():
    return next(iter(_basic_distinctions()))


def _basic_complex():
    pyphi.config.progress_bars = False
    sub = pyphi.examples.basic_substrate()
    return sub.maximal_complex((0, 0, 0))


def test_distinction_describe_structure():
    d = _basic_distinction()
    desc = d._describe(2)
    assert desc.title == "Distinction"
    all_row_labels = [r.label for sec in desc.sections for r in sec.rows]
    assert "Mechanism" in all_row_labels
    assert "φ_d" in all_row_labels
    assert "Purview" in all_row_labels
    assert "Specified state" in all_row_labels
    section_labels = [sec.label for sec in desc.sections]
    assert "Cause" in section_labels
    assert "Effect" in section_labels


def test_distinction_repr_is_card():
    d = _basic_distinction()
    out = repr(d)
    assert out.startswith("╭─ Distinction")
    assert "Mechanism" in out
    assert "φ_d" in out


def test_distinction_html_is_card():
    d = _basic_distinction()
    assert 'class="pyphi-card"' in d._repr_html_()


def test_distinction_low_verbosity_compact():
    d = _basic_distinction()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(d)
    assert out.startswith("Distinction(")
    assert "φ_d=" in out


def test_distinctions_describe_structure():
    dists = _basic_distinctions()
    desc = dists._describe(2)
    # Title should be the concrete subclass name
    assert "Distinctions" in desc.title
    section_labels = [sec.label for sec in desc.sections]
    assert "Distinctions" in section_labels
    top_rows = [r.label for r in desc.sections[0].rows]
    assert "Distinctions" in top_rows
    assert "Σφ_d" in top_rows


def test_distinctions_table_has_headers():
    dists = _basic_distinctions()
    desc = dists._describe(2)
    distinctions_sec = next(sec for sec in desc.sections if sec.label == "Distinctions")
    assert len(distinctions_sec.body) == 1
    table = distinctions_sec.body[0]
    assert isinstance(table, Table)
    assert table.headers == ("Mechanism", "φ_d", "Cause purview", "Effect purview")
    assert len(table.rows) == len(dists)


def test_distinctions_repr_is_card():
    dists = _basic_distinctions()
    out = repr(dists)
    assert out.startswith("╭─")
    assert "Distinctions" in out


def test_distinctions_html_is_card():
    dists = _basic_distinctions()
    assert 'class="pyphi-card"' in dists._repr_html_()


def test_distinctions_low_verbosity_compact():
    dists = _basic_distinctions()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(dists)
    assert "distinctions" in out
    assert "Σφ_d=" in out


def test_complex_describe_structure():
    cx = _basic_complex()
    desc = cx._describe(2)
    assert desc.title == "Complex"
    all_row_labels = [r.label for sec in desc.sections for r in sec.rows]
    assert "Φ" in all_row_labels
    assert "Nodes" in all_row_labels
    assert "Is maximal" in all_row_labels
    assert "Excluded candidates" in all_row_labels


def test_complex_repr_is_card():
    cx = _basic_complex()
    out = repr(cx)
    assert out.startswith("╭─ Complex")
    assert "Φ" in out
    assert "Is maximal" in out


def test_complex_html_is_card():
    cx = _basic_complex()
    assert 'class="pyphi-card"' in cx._repr_html_()


def test_complex_low_verbosity_compact():
    cx = _basic_complex()
    with pyphi.config.override(repr_verbosity=0):
        out = repr(cx)
    assert out.startswith("Complex(")
    assert "Φ=" in out
    assert "is_maximal=" in out


def test_excluded_candidate_is_leaf():
    from pyphi.models.complex import ExcludedCandidate

    ec = ExcludedCandidate((0, 1), 0.25)
    out = repr(ec)
    assert out.startswith("ExcludedCandidate(")
    assert "(0, 1)" in out
    assert "φ=0.25" in out
    assert "╭" not in out
    assert 'class="pyphi-leaf"' in ec._repr_html_()
    assert 'class="pyphi-card"' not in ec._repr_html_()


def _all_displayable_subclasses():
    """Every concrete Displayable subclass reachable after importing pyphi."""
    import pyphi  # noqa: F401  (ensure result modules are imported)
    from pyphi.display import Displayable

    seen, stack, out = set(), list(Displayable.__subclasses__()), []
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        out.append(cls)
        stack.extend(cls.__subclasses__())
    return out


def test_every_displayable_type_overrides_describe():
    """Ship criterion: every result type renders via its own _describe()."""
    from pyphi.display.mixin import Displayable

    offenders = [
        cls.__name__
        for cls in _all_displayable_subclasses()
        if cls._describe is Displayable._describe
    ]
    assert not offenders, f"Displayable types without a _describe(): {offenders}"


_GOLDEN_IIT4_SIA = """\
╭─ SystemIrreducibilityAnalysis ────────╮
│ φ_s              0.415037             │
│ Normalized φ_s   0.207519             │
│ System           A,B,C                │
│ Current state    (1, 0, 0)            │
├─ Cause ───────────────────────────────┤
│ Specified state             (1, 1, 0) │
│ Intrinsic information       3.0       │
│ Intrinsic differentiation   0.0       │
├─ Effect ──────────────────────────────┤
│ Specified state             (0, 0, 1) │
│ Intrinsic information       3.0       │
│ Intrinsic differentiation   0.0       │
├─ MIP ─────────────────────────────────┤
│ Partition   2 parts: {A,BC}           │
│ Tied MIPs   0                         │
╰───────────────────────────────────────╯"""


def test_iit4_sia_ascii_golden():
    """Exact-render anchor for the locked IIT 4.0 SIA card."""
    pyphi.config.progress_bars = False
    sia = pyphi.examples.basic_system().sia()
    assert repr(sia) == _GOLDEN_IIT4_SIA


def test_capped_table_respects_config_and_records_overflow():
    from pyphi.display.tables import capped_table

    items = list(range(100))
    with pyphi.config.override(repr_max_table_rows=10):
        t = capped_table(("n",), items, lambda i: (i,), total=len(items))
    assert len(t.rows) == 10
    assert t.overflow == 90


def test_ascii_table_shows_overflow_line():
    t = Table(headers=("n",), rows=(("1",), ("2",)), overflow=98)
    lines = ascii_backend._format_table(t)
    assert lines[-1] == "… 98 more"


def test_html_table_is_scrollable_and_shows_overflow():
    t = Table(headers=("n",), rows=(("1",),), overflow=5)
    out = html_backend._table_html(t)
    assert "pyphi-scroll" in out
    assert "… 5 more" in out


def test_ces_distinctions_table_truncates_under_config_cap():
    pyphi.config.progress_bars = False
    ces = pyphi.examples.basic_system().ces()  # 2 distinctions
    with pyphi.config.override(repr_max_table_rows=1):
        out = repr(ces)
    assert "… 1 more" in out  # 2 distinctions, cap 1 -> 1 hidden


def test_distinctions_table_colors_cause_effect_purview_headers():
    pyphi.config.progress_bars = False
    h = pyphi.examples.basic_system().ces()._repr_html_()
    # Inline color (not just a class) so the tone beats the more-specific
    # ``table.pyphi-table th`` rule and notebook front-ends' own table CSS.
    assert 'style="color:#D55C00">Cause purview' in h
    assert 'style="color:#009E73">Effect purview' in h


def test_unresolved_distinctions_colors_purview_headers():
    from pyphi.models.distinctions import UnresolvedDistinctions

    pyphi.config.progress_bars = False
    distinctions = pyphi.examples.basic_system().ces().distinctions
    unresolved = UnresolvedDistinctions(tuple(distinctions))
    h = unresolved._repr_html_()
    assert 'style="color:#D55C00">Cause purview' in h
    assert 'style="color:#009E73">Effect purview' in h


def test_account_links_table_colors_direction_cells():
    from pyphi import actual

    pyphi.config.progress_bars = False
    account = actual.account(pyphi.examples.prevention_transition())
    h = account._repr_html_()
    # Per-cell direction coloring: CAUSE cells orange, EFFECT cells green.
    assert 'style="color:#D55C00">' in h  # a CAUSE link cell
    assert 'style="color:#009E73">' in h  # an EFFECT link cell


def test_cut_grid_html_uses_inline_alignment():
    from pyphi.direction import Direction
    from pyphi.display.render.html import _table_html
    from pyphi.models.partitions import DirectedBipartition
    from pyphi.models.partitions import _cut_grid

    html = _table_html(_cut_grid(DirectedBipartition(Direction.EFFECT, (0,), (1, 2))))
    assert "pyphi-grid" in html
    assert "text-align:center" in html  # data/header columns centered inline
    assert "text-align:right" in html  # row-label column
    assert "pyphi-scroll" not in html  # grids are not wrapped/scrolled
