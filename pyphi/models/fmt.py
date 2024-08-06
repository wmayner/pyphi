# models/fmt.py
"""Helper functions for formatting pretty representations of PyPhi models."""

from fractions import Fraction
from itertools import chain, cycle
from typing import Iterable

import numpy as np
from toolz import concat

from .. import utils
from ..conf import config
from ..direction import Direction
from .cuts import CompleteSystemPartition, NullCut

# REPR_VERBOSITY levels
LOW = 0
MEDIUM = 1
HIGH = 2

# Unicode symbols
SMALL_PHI = "\u03C6"
BIG_PHI = "\u03A6"
ALPHA = "\u03B1"
TOP_LEFT_CORNER = "\u250C"
TOP_RIGHT_CORNER = "\u2510"
BOTTOM_LEFT_CORNER = "\u2514"
BOTTOM_RIGHT_CORNER = "\u2518"
HORIZONTAL_BAR = "\u2500"
VERTICAL_SIDE = "\u2502"
HEADER_BAR_1 = "\u2550"
HEADER_BAR_2 = "\u2501"
HEADER_BAR_3 = "\u254D"
DOTTED_HEADER = "\u2574"
LINE = "\u2501"
ARROW_LEFT = "\u25C0" + LINE * 2
ARROW_RIGHT = LINE * 2 + "\u25B6"
BACKWARD_CUT_SYMBOL = ARROW_LEFT + "/ /" + LINE * 2
FORWARD_CUT_SYMBOL = LINE * 2 + "/ /" + ARROW_RIGHT
EMPTY_SET = "\u2205"
MULTIPLY = "\u2715"

CUT_SYMBOLS_BY_DIRECTION = {
    Direction.CAUSE: BACKWARD_CUT_SYMBOL,
    Direction.EFFECT: FORWARD_CUT_SYMBOL,
}

NICE_DENOMINATORS = list(range(16)) + [16, 32, 64, 128]


def make_repr(self, attrs):
    """Construct a repr string.

    If `config.REPR_VERBOSITY` is ``1`` or ``2``, this function calls the
    object's __str__ method. Although this breaks the convention that __repr__
    should return a string which can reconstruct the object, readable reprs are
    invaluable since the Python interpreter calls `repr` to represent all
    objects in the shell. Since PyPhi is often used in the interpreter we want
    to have meaningful and useful representations.

    Args:
        self (obj): The object in question
        attrs (Iterable[str]): Attributes to include in the repr

    Returns:
        str: the ``repr``esentation of the object
    """
    # TODO: change this to a closure so we can do
    # __repr__ = make_repr(attrs) ???

    if config.REPR_VERBOSITY in [MEDIUM, HIGH]:
        return self.__str__()

    elif config.REPR_VERBOSITY is LOW:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(attr + "=" + repr(getattr(self, attr)) for attr in attrs),
        )

    raise ValueError("Invalid value for `config.REPR_VERBOSITY`")


def indent(lines, amount=2, char=" ", newline="\n"):
    r"""Indent a string.

    Prepends whitespace to every line in the passed string. (Lines are
    separated by newline characters.)

    Args:
        lines (str): The string to indent.

    Keyword Args:
        amount (int): The number of columns to indent by.
        char (str): The character to to use as the indentation.

    Returns:
        str: The indented string.

    Example:
        >>> print(indent('line1\nline2', char='*'))
        **line1
        **line2
    """
    lines = str(lines)
    padding = amount * char
    return padding + (newline + padding).join(lines.split(newline))


def margin(text):
    r"""Add a margin to both ends of each line in the string.

    Example:
        >>> margin('line1\nline2')
        '  line1  \n  line2  '
    """
    lines = str(text).split("\n")
    return "\n".join("  {}  ".format(l) for l in lines)


LINES_FORMAT_STR = VERTICAL_SIDE + " {line:<{width}} " + VERTICAL_SIDE


def box(text):
    r"""Wrap a chunk of text in a box.

    Example:
        >>> print(box('line1\nline2'))
        ┌───────┐
        │ line1 │
        │ line2 │
        └───────┘
    """
    lines = text.split("\n")

    w = width(lines)
    top_bar = TOP_LEFT_CORNER + HORIZONTAL_BAR * (2 + w) + TOP_RIGHT_CORNER
    bottom_bar = BOTTOM_LEFT_CORNER + HORIZONTAL_BAR * (2 + w) + BOTTOM_RIGHT_CORNER

    lines = [LINES_FORMAT_STR.format(line=line, width=w) for line in lines]

    return top_bar + "\n" + "\n".join(lines) + "\n" + bottom_bar


def side_by_side(left, right):
    r"""Put two boxes next to each other.

    Assumes that all lines in the boxes are the same width.

    Example:
        >>> left = 'A \nC '
        >>> right = 'B\nD'
        >>> print(side_by_side(left, right))
        A B
        C D
        <BLANKLINE>
    """
    left_lines = list(left.split("\n"))
    right_lines = list(right.split("\n"))

    # Pad the shorter column with whitespace
    diff = abs(len(left_lines) - len(right_lines))
    if len(left_lines) > len(right_lines):
        fill = " " * len(right_lines[0])
        right_lines += [fill] * diff
    elif len(right_lines) > len(left_lines):
        fill = " " * len(left_lines[0])
        left_lines += [fill] * diff

    return "\n".join(a + b for a, b in zip(left_lines, right_lines)) + "\n"


def width(lines):
    """Return the maximum width of the given lines.

    Example:
        >>> width(["abcde", "abc", "", "abcdefg"])
        7
    """
    return max(map(len, lines))


def header(head, text, over_char=None, under_char=None, center=True):
    """Center a head over a block of text.

    The width of the text is the width of the longest line of the text.
    """
    lines = list(text.split("\n"))
    w = width(lines)

    # Center or left-justify
    if center:
        head = head.center(w) + "\n"
    else:
        head = head.ljust(w) + "\n"

    # Underline head
    if under_char:
        head = head + under_char * w + "\n"

    # 'Overline' head
    if over_char:
        head = over_char * w + "\n" + head

    return head + text


def labels(indices, node_labels=None):
    """Get the labels for a tuple of mechanism indices."""
    if node_labels is None:
        return tuple(map(str, indices))
    return node_labels.indices2labels(indices)


def is_multiline(text):
    """Return True if the text contains newlines."""
    return "\n" in text


def align(lines: Iterable[str], direction="<"):
    """Align lines by padding with spaces.

    Examples:
        >>> lines = ["abcde", "abc", "", "abcdefg"]
        >>> align(lines, direction="<")
        ['abcde  ', 'abc    ', '       ', 'abcdefg']
        >>> align(["abcde", "abc\\ndef", "abcdefg"], direction="<")
        ['abcde  ', 'abc    ', 'def    ', 'abcdefg']
        >>> align(lines, direction=">")
        ['  abcde', '    abc', '       ', 'abcdefg']
        >>> align(lines, direction="c")
        [' abcde ', '  abc  ', '       ', 'abcdefg']

    """
    lines = list(
        concat([text.split("\n") if is_multiline(text) else [text] for text in lines])
    )
    w = width(lines)
    if direction == "c":
        return [line.center(w) for line in lines]
    spec = " {direction}{width}".format(direction=direction, width=w)
    return [format(line, spec) for line in lines]


def center(text):
    """Center-align a string."""
    return "\n".join(align(text.split("\n"), direction="c"))


def split_decimal(n):
    """Attempt to split an object into unit and decimal parts, handling
    non-numeric types.

    Examples:
        >>> split_decimal(1.5)
        ['1', '5']
        >>> split_decimal('1.5')
        ['1', '5']
        >>> split_decimal(1)
        ['1', '']
        >>> split_decimal(1.0)
        ['1', '0']
        >>> split_decimal(np.nan)
        ['', 'nan']
        >>> split_decimal(None)
        ['', 'None']
    """
    if n is None:
        return ["", str(None)]
    try:
        if np.isnan(n):
            # nan
            return ["", str(n)]
    except TypeError:
        pass
    if isinstance(n, float):
        # float
        return str(n).split(".")
    try:
        n = float(n)
        if n.is_integer():
            # int
            return [str(int(n)), ""]
        # float
        return str(n).split(".")
    except ValueError:
        pass
    # Assume str
    return ["", str(n)]


def align_decimals(numbers):
    """Align numbers on the decimal point.

    Integers (whether of type `int` or `float`) and floats are aligned. `str`
    elements are aligned to the right of the decimal point.

    Examples:
        >>> numbers = [0.0, 1, 0.99, 100.5, 80.123, 'string']
        >>> align_decimals(numbers)
        ['  0.0     ', '  1      ', '  0.99    ', '100.5     ', ' 80.123   ', '   string']
        >>> align_decimals([0.5] + list(map(str, numbers)))
        ['  0.5     ', '  0      ', '  1      ', '  0.99    ', '100.5     ', ' 80.123   ', '   string']
    """
    units, decimals = zip(*map(split_decimal, numbers))
    points = ["." if unit and decimal else "" for unit, decimal in zip(units, decimals)]
    units = align(units, direction=">")
    decimals = align(decimals, direction="<")
    return ["".join(elements) for elements in zip(units, points, decimals)]


def _multiline_string_to_columns(text):
    return [("", line) for line in text.split("\n")]


def _expand_multiline_strings(left, right):
    """Expand a multiline 'right side' string into a list of columns with empty
    left sides.
    """
    # TODO deal with multiline left?
    if not isinstance(right, str) or not is_multiline(right):
        return [(left, right)]
    columns = _multiline_string_to_columns(right)
    columns[0] = (left, columns[0][1])
    return columns


def align_columns(
    lines,
    delimiter=": ",
    alignment="><",
    types="tn",
    split_columns=False,
):
    """Align columns of text.

    # If a line does not contain the delimiter, it will be assumed to belong to
    # the previous line and will be indented.

    Arguments:
        lines (Iterable): The lines to align.

    Keyword Arguments:
        delimiter (str): The delimiter of the columns.
        alignment (str): A string of ">" and "<", indicating the alignment for
            each column.
        types (str): A string of "t" (text) and "n" (numeric), indicating the
            type of each column.
        split_columns (bool): If True, assume lines are single strings rather
            than tuples, and split them on the delimiter beforehand.

    Examples:
        >>> columns = [
        ...     ('abc', 0.0),
        ...     ('a', 1),
        ...     ('b', 0.999),
        ...     ('c', 100.5),
        ...     ('xy', 80.12),
        ... ]
        >>> align_columns(columns)
        ['abc:   0.0  ', '  a:   1    ', '  b:   0.999', '  c: 100.5  ', ' xy:  80.12 ']
        >>> lines = [
        ...     'abc: 0.0',
        ...     'a: 1',
        ...     'b: 0.999',
        ...     'c: 100.5',
        ...     'xy: 80.12',
        ... ]
        >>> align_columns(lines, split_columns=True)
        ['abc:   0    ', '  a:   1    ', '  b:   0.999', '  c: 100.5  ', ' xy:  80.12 ']
    """
    if split_columns:
        lines = [str(line).split(delimiter) for line in lines]
    # Expand multiline strings into new columns
    lines = concat([_expand_multiline_strings(left, right) for left, right in lines])
    # Reorient into columns
    columns = list(zip(*lines))
    for i, t in enumerate(types):
        if t == "n":
            columns[i] = align_decimals(columns[i])
    alignment = cycle(alignment)
    columns = [align(column, direction=a) for column, a in zip(columns, alignment)]
    return [delimiter.join(line) for line in zip(*columns)]


def fmt_number(p):
    """Format a number.

    It will be printed as a fraction if the denominator isn't too big and as a
    decimal otherwise.

    If formatting fails, return the input unmodified.
    """
    try:
        formatted = format(p, f".{config.PRECISION}f")
    except (ValueError, TypeError):
        return str(p)

    if not config.PRINT_FRACTIONS:
        return formatted

    fraction = Fraction(p)
    nice = fraction.limit_denominator(128)
    return (
        str(nice)
        if (utils.eq(fraction, nice) and nice.denominator in NICE_DENOMINATORS)
        else formatted
    )


def fmt_nodes(nodes, node_labels=None):
    """Format nodes, optionally with labels."""
    return ",".join(labels(nodes, node_labels)) if nodes else EMPTY_SET


def fmt_mechanism(indices, node_labels=None):
    """Format a mechanism or purview."""
    return "[" + fmt_nodes(indices, node_labels=node_labels) + "]"


def fmt_fraction(numer: str, denom: str):
    """Format a fraction.

    Arguments:
        numer (str): The numerator.
        denom (str): The denominator.
    """
    w = max(3, len(numer), len(denom))
    divider = HORIZONTAL_BAR * w

    return ("{numer:^{width}}\n" "{divider}\n" "{denom:^{width}}").format(
        numer=numer, divider=divider, denom=denom, width=w
    )


def fmt_part(part, node_labels=None):
    """Format a |Part|.

    The returned string looks like::

        0,1
        ───
         ∅
    """
    numer = fmt_nodes(part.mechanism, node_labels=node_labels)
    denom = fmt_nodes(part.purview, node_labels=node_labels)

    return fmt_fraction(numer, denom)


def fmt_partition(partition):
    """Format a |Bipartition|.

    The returned string looks like::

        0,1    ∅
        ─── ✕ ───
         2    0,1

    Args:
        partition (Bipartition): The partition in question.

    Returns:
        str: A human-readable string representation of the partition.
    """
    # TODO(4.0) deprecate
    if not partition:
        return ""
    try:
        parts = [
            # TODO(4.0)
            # str(part).split("\n")
            fmt_part(part, node_labels=partition.node_labels).split("\n")
            for part in partition
        ]

        times = ("   ", " {} ".format(MULTIPLY), "   ")
        breaks = ("\n", "\n", "")  # No newline at the end of string
        between = [times] * (len(parts) - 1) + [breaks]

        # Alternate [part, break, part, ..., end]
        elements = chain.from_iterable(zip(parts, between))

        # Transform vertical stacks into horizontal lines
        return "".join(chain.from_iterable(zip(*elements)))
    except TypeError:
        return repr(partition)


def fmt_phi_structure(ps, title="Phi-structure", subsystem=True):
    """Format a PhiStructure."""
    distinctions = len(ps.distinctions)

    if ps.requires_filter_relations:
        relations = sum_phi = sum_phi_r = sii = selectivity = "[requires filter]"
    elif ps.relations is None:
        relations = sum_phi = sum_phi_r = sii = selectivity = "[not computed]"
    else:
        relations = len(ps.relations)
        sum_phi = ps.sum_phi()
        sum_phi_r = ps.relations.sum_phi()
        sii = ps.system_intrinsic_information()
        selectivity = ps.selectivity()

    columns = [
        ("Distinctions", distinctions),
        ("Relations", relations),
        ("Σφ_d", ps.sum_phi_distinctions()),
        ("Σφ_r", sum_phi_r),
        ("Σφ", sum_phi),
        ("Selectivity", selectivity),
        ("S.I.I.", sii),
    ]
    lines = align_columns(columns)
    if subsystem:
        lines = align_columns(
            lines + [f"Subsystem: {ps.subsystem.nodes}"],
            types="tt",
            split_columns=True,
        )
    body = "\n".join(lines)
    if title:
        body = header(title, body, HEADER_BAR_1, HEADER_BAR_1)
    return body


def fmt_partitioned_phi_structure(
    ps,
    title="Partitioned phi-structure",
    subsystem=True,
):
    """Format a PartitionedPhiStructure."""
    if isinstance(ps.partition, (NullCut, CompleteSystemPartition)):
        cut = str(ps.partition)
    else:
        cut = fmt_cut(ps.partition, direction=ps.partition.direction, name=False)
    lines = align_columns(
        fmt_phi_structure(ps, title=None, subsystem=subsystem).split("\n")
        + [f"Partition: {cut}"],
        types="tt",
        split_columns=True,
    )
    body = "\n".join(lines)
    if title:
        body = header(title, body, HEADER_BAR_1, HEADER_BAR_1)
    return body


def fmt_ces(ces, title=None):
    """Format a |CauseEffectStructure|."""
    if title is None:
        title = ces.__class__.__name__
    if not ces:
        return "()\n"

    concepts = center("\n".join(margin(x) for x in ces) + "\n")
    title = "{} ({} distinction{})".format(
        title, len(ces), "" if len(ces) == 1 else "s"
    )

    return header(title, concepts, HEADER_BAR_1, HEADER_BAR_1)


def fmt_concept(concept):
    """Format a |Concept|."""

    def fmt_cause_or_effect(x):  # pylint: disable=missing-docstring
        return indent(str(x), amount=1)

    cause = fmt_cause_or_effect(concept.cause)
    effect = fmt_cause_or_effect(concept.effect)
    ce = side_by_side(cause, effect)

    mechanism = fmt_mechanism(concept.mechanism, concept.node_labels)
    # TODO(4.0) reconsider using Nodes in the mechanism to facilitate access to their state, etc.
    title = "\n".join(
        align(
            [
                f"{concept.__class__.__name__}: mechanism = {mechanism}, state = {list(concept.mechanism_state)}",
                f"{SMALL_PHI} = {fmt_number(concept.phi)}",
            ],
            direction="c",
        )
    )

    # Only center headers for high-verbosity output
    center = config.REPR_VERBOSITY is HIGH
    return header(title, ce, HEADER_BAR_2, HEADER_BAR_2, center=center)


def fmt_ria(ria, verbose=True, mip=False):
    """Format a |RepertoireIrreducibilityAnalysis|."""
    if verbose:
        mechanism = f"Mechanism: {fmt_mechanism(ria.mechanism, ria.node_labels)}"
        direction = f"Direction: {ria.direction}"
    else:
        mechanism = ""
        direction = ""

    # TODO(4.0):  position repertoire and partitioned repertoire side by side
    # TODO(ties) fix state-marking logic
    if config.REPR_VERBOSITY is HIGH:
        partition_name = "MIP" if mip else "Partition"
        partition = f"{partition_name}: "
        if ria.partition:
            partition += f"\n{indent(fmt_partition(ria.partition))}"
        else:
            partition += "empty"
        if ria.specified_state is not None:
            mark_states = [specified.state for specified in ria.specified_state.ties]
        else:
            mark_states = []
        # TODO(refactor)
        if ria.repertoire is not None:
            if ria.repertoire.size == 1:
                repertoire = f"Forward probability:\n    {ria.repertoire}"
                partitioned_repertoire = f"Partitioned forward probability:\n    {ria.partitioned_repertoire}"
            else:
                repertoire = "Repertoire:\n{}".format(
                    indent(fmt_repertoire(ria.repertoire, mark_states=mark_states))
                )
                partitioned_repertoire = "Partitioned repertoire:\n{}".format(
                    indent(
                        fmt_repertoire(
                            ria.partitioned_repertoire,
                            mark_states=mark_states,
                        )
                    )
                )
        else:
            repertoire = ""
            partitioned_repertoire = ""
    else:
        partition = ""
        repertoire = ""
        partitioned_repertoire = ""

    data = (
        [
            f"{SMALL_PHI} = {fmt_number(ria.phi)}",
            f"Normalized {SMALL_PHI} = {fmt_number(ria.normalized_phi)}",
            f"{mechanism}",
            f"Purview: {fmt_mechanism(ria.purview, ria.node_labels)}",
            f"Specified state:\n{ria.specified_state}",
            f"{direction}",
            f"{partition}",
        ]
        + ([f"Selectivity: {ria.selectivity}"] if ria.selectivity is not None else [])
        + [
            f"{repertoire}",
            f"{partitioned_repertoire}",
            f"#(state ties): {ria.num_state_ties}",
            f"#(partition ties): {ria.num_partition_ties}",
        ]
    )
    if hasattr(ria, "num_purview_ties"):
        data.append(f"#(purview ties): {ria.num_purview_ties}")
    if ria.reasons is not None:
        data.append("Reasons: " + ", ".join(map(str, ria.reasons)))
    return "\n".join(data)


def fmt_cut(cut, direction=None, name=True):
    """Format a |Cut|."""
    try:
        if name:
            name = cut.__class__.__name__ + " "
        else:
            name = ""
        return "{name}{from_nodes} {symbol} {to_nodes}".format(
            name=name,
            from_nodes=fmt_mechanism(cut.from_nodes, cut.node_labels),
            symbol=(
                FORWARD_CUT_SYMBOL
                if direction is None
                else CUT_SYMBOLS_BY_DIRECTION[direction]
            ),
            to_nodes=fmt_mechanism(cut.to_nodes, cut.node_labels),
        )
    except AttributeError:
        return str(cut)


def fmt_kcut(cut):
    """Format a |KCut|."""
    return "KCut {}\n{}".format(cut.direction, cut.partition)


def fmt_sia_4(sia, phi_structure=True, title="System irreducibility analysis"):
    """Format an IIT 4.0 |SystemIrreducibilityAnalysis|."""
    if phi_structure:
        body = "\n".join(
            [
                fmt_phi_structure(sia.phi_structure, subsystem=False),
                fmt_phi_structure(
                    sia.partitioned_phi_structure,
                    title="Partitioned phi-structure",
                    subsystem=False,
                ),
            ]
        )
    else:
        body = ""

    selectivity = sia.selectivity
    if selectivity is None:
        selectivity = "[not computed]"
    informativeness = sia.informativeness
    if informativeness is None:
        informativeness = "[not computed]"

    lines = [
        (BIG_PHI, sia.phi),
        ("Selectivity", selectivity),
        ("Informativeness", informativeness),
    ]
    lines = align_columns(lines)
    body = "\n".join(["\n".join(lines), body])

    if isinstance(sia.partition, (NullCut, CompleteSystemPartition)):
        cut = str(sia.partition)
    else:
        cut = fmt_cut(sia.partition, direction=sia.partition.direction, name=False)

    data = [
        sia.subsystem.nodes,
        cut,
    ]
    if sia.reasons:
        data.append("[trivially reducible]\n" + "\n".join(map(str, sia.reasons)))
    data.append("")
    for line in reversed(data):
        body = header(str(line), body)
    body = header(title, body, under_char=HEADER_BAR_2)
    return box(center(body))


def fmt_sia(sia, ces=True, title="System irreducibility analysis"):
    """Format a |SystemIrreducibilityAnalysis|."""
    if ces:
        body = "{ces}\n{partitioned_ces}".format(
            ces=fmt_ces(sia.ces, "Cause-effect structure"),
            partitioned_ces=fmt_ces(
                sia.partitioned_ces, "Partitioned cause-effect structure"
            ),
        )
    else:
        body = ""

    data = [
        f"{BIG_PHI}: {fmt_number(sia.phi)}",
        sia.subsystem,
        sia.cut,
    ]
    for line in reversed(data):
        body = header(str(line), body)
    body = header(title, body, under_char=HEADER_BAR_2)
    return box(center(body))


def fmt_repertoire(r, mark_states=None):
    """Format a repertoire."""
    # TODO: will this get unwieldy with large repertoires?
    if r is None:
        return ""

    r = r.squeeze()

    lines = []

    # Header: 'S      P(S)'
    space = " " * 4
    head = "{S:^{s_width}}{space}Pr({S})".format(S="S", s_width=r.ndim, space=space)
    lines.append(head)

    # Lines: '001     .25'
    for state in utils.all_states(r.ndim):
        state_str = "".join(str(i) for i in state)
        if state in mark_states:
            state_str += " *"
        else:
            state_str += "  "
        lines.append("{0}{1}{2}".format(state_str, space[:-2], fmt_number(r[state])))

    w = width(lines)
    lines.insert(1, DOTTED_HEADER * (w + 1))

    return box("\n".join(lines))


def fmt_relatum(relatum, node_labels=None):
    direction = "Cause" if relatum.direction == Direction.CAUSE else "Effect"
    return (
        direction
        + fmt_mechanism(relatum.mechanism, node_labels=node_labels)
        + "/"
        + fmt_mechanism(relatum.purview, node_labels=node_labels)
    )


def fmt_relata(relata, node_labels=None):
    lines = [fmt_relatum(relatum, node_labels=node_labels) for relatum in relata]
    lines = align_columns(lines, delimiter="/", split_columns=True)
    # TODO(4.0) align purview nodes?
    return "\n".join(lines)


def fmt_relation(relation):
    labels = relation.subsystem.node_labels
    body = fmt_relata(relation.relata, node_labels=labels)
    data = [
        ("φ", relation.phi),
        ("Purview", fmt_mechanism(relation.purview, node_labels=labels)),
        ("Relata", ""),
    ]
    data = "\n".join(align_columns(data))
    body = center(header(data, body))
    return header("Relation", body, over_char=HEADER_BAR_3, under_char=HEADER_BAR_3)


def _fmt_relations(relations, title=None, body="", data=None):
    if title is None:
        title = relations.__class__.__name__
    if data is None:
        data = []
    data = [
        ("#", len(relations)),
        ("Σφ", relations.sum_phi()),
    ] + data
    data = "\n".join(align_columns(data))
    body = header(data, body)
    body = header(title, body, under_char=HEADER_BAR_1)
    return center(body)


def fmt_concrete_relations(relations, title=None):
    body = "\n".join(map(fmt_relation, relations))
    return _fmt_relations(relations, title, body)


def fmt_analytical_relations(relations, title=None):
    body = ""
    return _fmt_relations(relations, title, body)


def fmt_sampled_relations(relations, title=None):
    body = "\n".join(map(fmt_relation, relations.sample))
    return _fmt_relations(
        relations, title, body, data=[("Sampled", len(relations.sample))]
    )


def fmt_extended_purview(extended_purview, node_labels=None):
    """Format an extended purview."""
    if len(extended_purview) == 1:
        return fmt_mechanism(extended_purview[0], node_labels=node_labels)

    purviews = [
        fmt_mechanism(purview, node_labels=node_labels) for purview in extended_purview
    ]
    return "[" + ", ".join(purviews) + "]"


def fmt_causal_link(causal_link):
    """Format a CausalLink."""
    return fmt_ac_ria(causal_link, extended_purview=causal_link.extended_purview)


def fmt_ac_ria(ria, extended_purview=None):
    """Format an AcRepertoireIrreducibilityAnalysis."""
    causality = {
        Direction.CAUSE: (
            (
                fmt_mechanism(ria.purview, ria.node_labels)
                if extended_purview is None
                else fmt_extended_purview(ria.extended_purview, ria.node_labels)
            ),
            ARROW_LEFT,
            fmt_mechanism(ria.mechanism, ria.node_labels),
        ),
        Direction.EFFECT: (
            fmt_mechanism(ria.mechanism, ria.node_labels),
            ARROW_RIGHT,
            (
                fmt_mechanism(ria.purview, ria.node_labels)
                if extended_purview is None
                else fmt_extended_purview(ria.extended_purview, ria.node_labels)
            ),
        ),
    }[ria.direction]
    causality = " ".join(causality)

    return "{ALPHA} = {alpha}  {causality}".format(
        ALPHA=ALPHA, alpha=round(ria.alpha, 4), causality=causality
    )


def fmt_account(account, title=None):
    """Format an Account or a DirectedAccount."""
    if title is None:
        title = account.__class__.__name__  # `Account` or `DirectedAccount`

    title = "{} ({} causal link{})".format(
        title, len(account), "" if len(account) == 1 else "s"
    )

    body = ""
    body += "Irreducible effects\n"
    body += "\n".join(fmt_causal_link(m) for m in account.irreducible_effects)
    body += "\nIrreducible causes\n"
    body += "\n".join(fmt_causal_link(m) for m in account.irreducible_causes)

    return "\n" + header(title, body, under_char="*")


def fmt_ac_sia(ac_sia):
    """Format a AcSystemIrreducibilityAnalysis."""
    body = (
        "{ALPHA} = {alpha}\n"
        "direction: {ac_sia.direction}\n"
        "transition: {ac_sia.transition}\n"
        "before state: {ac_sia.before_state}\n"
        "after state: {ac_sia.after_state}\n"
        "cut:\n{ac_sia.cut}\n"
        "{account}\n"
        "{partitioned_account}".format(
            ALPHA=ALPHA,
            alpha=round(ac_sia.alpha, 4),
            ac_sia=ac_sia,
            account=fmt_account(ac_sia.account, "Account"),
            partitioned_account=fmt_account(
                ac_sia.partitioned_account, "Partitioned Account"
            ),
        )
    )

    return box(
        header("AcSystemIrreducibilityAnalysis", body, under_char=HORIZONTAL_BAR)
    )


def fmt_transition(t):
    """Format a |Transition|."""
    return "Transition({} {} {})".format(
        fmt_mechanism(t.cause_indices, t.node_labels),
        ARROW_RIGHT,
        fmt_mechanism(t.effect_indices, t.node_labels),
    )


def state(state):
    """Format a state."""
    return "(" + ",".join(map(str, state)) + ")"
