# models/fmt.py
"""Helper functions for formatting pretty representations of PyPhi models."""

from collections.abc import Iterable
from collections.abc import Sequence
from fractions import Fraction
from itertools import chain
from itertools import cycle
from typing import Any

import numpy as np
from toolz import concat

from pyphi import utils
from pyphi.conf import config
from pyphi.direction import Direction

from .cuts import CompleteSystemPartition
from .cuts import NullCut

# REPR_VERBOSITY levels
LOW = 0
MEDIUM = 1
HIGH = 2

# Unicode symbols
SMALL_PHI = "\u03c6"
BIG_PHI = "\u03a6"
ALPHA = "\u03b1"
TOP_LEFT_CORNER = "\u250c"
TOP_RIGHT_CORNER = "\u2510"
BOTTOM_LEFT_CORNER = "\u2514"
BOTTOM_RIGHT_CORNER = "\u2518"
HORIZONTAL_BAR = "\u2500"
VERTICAL_SIDE = "\u2502"
HEADER_BAR_1 = "\u2550"
HEADER_BAR_2 = "\u2501"
HEADER_BAR_3 = "\u254d"
DOTTED_HEADER = "\u2574"
LINE = "\u2501"
ARROW_LEFT = "\u25c0" + LINE * 2
ARROW_RIGHT = LINE * 2 + "\u25b6"
BACKWARD_CUT_SYMBOL = ARROW_LEFT + "/ /" + LINE * 2
FORWARD_CUT_SYMBOL = LINE * 2 + "/ /" + ARROW_RIGHT
EMPTY_SET = "\u2205"
MULTIPLY = "\u2715"

CUT_SYMBOLS_BY_DIRECTION = {
    Direction.CAUSE: BACKWARD_CUT_SYMBOL,
    Direction.EFFECT: FORWARD_CUT_SYMBOL,
}

NICE_DENOMINATORS = [*list(range(16)), 16, 32, 64, 128]


def make_repr(self: object, attrs: Iterable[str]) -> str:
    """Construct a repr string.

    If `config.infrastructure.repr_verbosity` is ``1`` or ``2``, this function calls the
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

    if config.infrastructure.repr_verbosity in [MEDIUM, HIGH]:
        return self.__str__()  # type: ignore[attr-defined,unused-ignore]

    if config.infrastructure.repr_verbosity is LOW:
        # Only include attributes that exist on the object
        attr_strs = [
            f"{attr}={getattr(self, attr)!r}" for attr in attrs if hasattr(self, attr)
        ]
        return "{}({})".format(self.__class__.__name__, ", ".join(attr_strs))

    raise ValueError("Invalid value for `config.infrastructure.repr_verbosity`")


def indent(lines: str, amount: int = 2, char: str = " ", newline: str = "\n") -> str:
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


def margin(text: str) -> str:
    r"""Add a margin to both ends of each line in the string.

    Example:
        >>> margin('line1\nline2')
        '  line1  \n  line2  '
    """
    lines = str(text).split("\n")
    return "\n".join(f"  {line}  " for line in lines)


LINES_FORMAT_STR = VERTICAL_SIDE + " {line:<{width}} " + VERTICAL_SIDE


def box(text: str) -> str:
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


def side_by_side(left: str, right: str) -> str:
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

    return "\n".join(a + b for a, b in zip(left_lines, right_lines, strict=False)) + "\n"


def width(lines: Iterable[str]) -> int:
    """Return the maximum width of the given lines.

    Example:
        >>> width(["abcde", "abc", "", "abcdefg"])
        7
    """
    return max(map(len, lines))


def header(
    head: str,
    text: str,
    over_char: str | None = None,
    under_char: str | None = None,
    center: bool = True,
) -> str:
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


def labels(
    indices: tuple[int, ...], node_labels: object | None = None
) -> tuple[str, ...]:
    """Get the labels for a tuple of mechanism indices."""
    if node_labels is None:
        return tuple(map(str, indices))
    return node_labels.indices2labels(indices)  # type: ignore[attr-defined]


def is_multiline(text: str) -> bool:
    """Return True if the text contains newlines."""
    return "\n" in text


def align(lines: Iterable[str], direction: str = "<") -> list[str]:
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
    spec = f" {direction}{w}"
    return [format(line, spec) for line in lines]


def center(text: str) -> str:
    """Center-align a string."""
    return "\n".join(align(text.split("\n"), direction="c"))


def split_decimal(n: Any) -> list[str]:  # noqa: PLR0911
    """Attempt to split an object into unit and decimal parts, handling
    non-numeric types.

    Always returns a 2-element list [units, decimals].

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
        # float - use float() to get base float value for subclasses that
        # override __str__ (e.g., DistanceResult)
        parts = str(float(n)).split(".")
        if len(parts) == 1:
            return [parts[0], ""]
        return parts
    try:
        n = float(n)
        if n.is_integer():
            # int
            return [str(int(n)), ""]
        # float
        parts = str(n).split(".")
        if len(parts) == 1:
            return [parts[0], ""]
        return parts
    except ValueError:
        pass
    # Assume str
    return ["", str(n)]


def align_decimals(numbers: Iterable[Any]) -> list[str]:
    """Align numbers on the decimal point.

    Integers (whether of type `int` or `float`) and floats are aligned. `str`
    elements are aligned to the right of the decimal point.

    Examples:
        >>> numbers = [0.0, 1, 0.99, 100.5, 80.123, 'string']
        >>> align_decimals(numbers)  # doctest: +NORMALIZE_WHITESPACE
        ['  0.0     ', '  1      ', '  0.99    ', '100.5     ',
         ' 80.123   ', '   string']
        >>> mixed = [0.5] + list(map(str, numbers))
        >>> align_decimals(mixed)  # doctest: +NORMALIZE_WHITESPACE
        ['  0.5     ', '  0      ', '  1      ', '  0.99    ', '100.5     ',
         ' 80.123   ', '   string']
        >>> align_decimals([])
        []
    """
    numbers_list = list(numbers)
    if not numbers_list:
        return []
    units_tuple, decimals_tuple = zip(*map(split_decimal, numbers_list), strict=False)
    points = [
        "." if unit and decimal else ""
        for unit, decimal in zip(units_tuple, decimals_tuple, strict=False)
    ]
    units_list = align(units_tuple, direction=">")
    decimals_list = align(decimals_tuple, direction="<")
    return [
        "".join(elements)
        for elements in zip(units_list, points, decimals_list, strict=False)
    ]


def _multiline_string_to_columns(text: str) -> list[tuple[str, str]]:
    return [("", line) for line in text.split("\n")]


def _expand_multiline_strings(left: str, right: str) -> list[tuple[str, str]]:
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
    lines: Iterable[Any],
    delimiter: str = ": ",
    alignment: str = "><",
    types: str = "tn",
    split_columns: bool = False,
) -> list[str]:
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
    columns: list[Any] = list(zip(*lines, strict=False))
    for i, t in enumerate(types):
        if t == "n":
            columns[i] = align_decimals(columns[i])
    alignment_cycle = cycle(alignment)
    columns_aligned = [
        align(column, direction=a)
        for column, a in zip(columns, alignment_cycle, strict=False)
    ]
    return [delimiter.join(line) for line in zip(*columns_aligned, strict=False)]


def fmt_number(p: Any) -> str:
    """Format a number.

    It will be printed as a fraction if the denominator isn't too big and as a
    decimal otherwise.

    If formatting fails, return the input unmodified.
    """
    try:
        formatted = format(p, f".{config.numerics.precision}f")
    except (ValueError, TypeError):
        return str(p)

    if not config.infrastructure.print_fractions:
        return formatted

    fraction = Fraction(p)
    nice = fraction.limit_denominator(128)
    return (
        str(nice)
        if (
            utils.eq(float(fraction), float(nice))
            and nice.denominator in NICE_DENOMINATORS
        )
        else formatted
    )


def fmt_nodes(nodes: tuple[int, ...], node_labels: object | None = None) -> str:
    """Format nodes, optionally with labels."""
    return ",".join(labels(nodes, node_labels)) if nodes else EMPTY_SET


def fmt_mechanism(indices: tuple[int, ...], node_labels: object | None = None) -> str:
    """Format a mechanism or purview."""
    return "[" + fmt_nodes(indices, node_labels=node_labels) + "]"


def fmt_fraction(numer: str, denom: str) -> str:
    """Format a fraction.

    Arguments:
        numer (str): The numerator.
        denom (str): The denominator.
    """
    w = max(3, len(numer), len(denom))
    divider = HORIZONTAL_BAR * w

    return ("{numer:^{width}}\n{divider}\n{denom:^{width}}").format(
        numer=numer, divider=divider, denom=denom, width=w
    )


def fmt_part(part: object, node_labels: object | None = None) -> str:
    """Format a |Part|.

    The returned string looks like::

        0,1
        ───
         ∅
    """
    numer = fmt_nodes(part.mechanism, node_labels=node_labels)  # type: ignore[attr-defined]
    denom = fmt_nodes(part.purview, node_labels=node_labels)  # type: ignore[attr-defined]

    return fmt_fraction(numer, denom)


def fmt_partition(partition: object) -> str:
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
            fmt_part(part, node_labels=partition.node_labels).split("\n")  # type: ignore[attr-defined]
            for part in partition  # type: ignore[attr-defined]
        ]

        times = ("   ", f" {MULTIPLY} ", "   ")
        breaks = ("\n", "\n", "")  # No newline at the end of string
        between = [times] * (len(parts) - 1) + [breaks]

        # Alternate [part, break, part, ..., end]
        elements = chain.from_iterable(zip(parts, between, strict=False))

        # Transform vertical stacks into horizontal lines
        return "".join(chain.from_iterable(zip(*elements, strict=False)))
    except TypeError:
        return repr(partition)


def fmt_phi_structure(
    ps: object, title: str | None = "Phi-structure", system: bool = True
) -> str:
    """Format a CauseEffectStructure."""
    distinctions = len(ps.distinctions)  # type: ignore[attr-defined]

    relations: Any
    if ps.requires_filter_relations:  # type: ignore[attr-defined]
        relations = sum_phi = sum_phi_r = sii = selectivity = "[requires filter]"
    elif ps.relations is None:  # type: ignore[attr-defined]
        relations = sum_phi = sum_phi_r = sii = selectivity = "[not computed]"
    else:
        relations = len(ps.relations)  # type: ignore[attr-defined]
        sum_phi = ps.sum_phi()  # type: ignore[attr-defined]
        sum_phi_r = ps.relations.sum_phi()  # type: ignore[attr-defined]
        sii = ps.system_intrinsic_information()  # type: ignore[attr-defined]
        selectivity = ps.selectivity()  # type: ignore[attr-defined]

    columns = [
        ("Distinctions", distinctions),
        ("Relations", relations),
        ("Σφ_d", ps.sum_phi_distinctions()),  # type: ignore[attr-defined]
        ("Σφ_r", sum_phi_r),
        ("Σφ", sum_phi),
        ("Selectivity", selectivity),
        ("S.I.I.", sii),
    ]
    lines = align_columns(columns)
    if system:
        lines = align_columns(
            [*lines, f"System: {ps.system.nodes}"],  # type: ignore[attr-defined]
            types="tt",
            split_columns=True,
        )
    body = "\n".join(lines)
    if title:
        body = header(title, body, HEADER_BAR_1, HEADER_BAR_1)
    return body


def fmt_partitioned_phi_structure(
    ps: object,
    title: str = "Partitioned phi-structure",
    system: bool = True,
) -> str:
    """Format a PartitionedCauseEffectStructure."""
    if isinstance(ps.partition, (NullCut, CompleteSystemPartition)):  # type: ignore[attr-defined]
        cut = str(ps.partition)  # type: ignore[attr-defined]
    else:
        cut = fmt_cut(ps.partition, direction=ps.partition.direction, name=False)  # type: ignore[attr-defined]
    lines = align_columns(
        [
            *fmt_phi_structure(ps, title=None, system=system).split("\n"),
            f"Partition: {cut}",
        ],
        types="tt",
        split_columns=True,
    )
    body = "\n".join(lines)
    if title:
        body = header(title, body, HEADER_BAR_1, HEADER_BAR_1)
    return body


def fmt_ces(ces: object, title: str | None = None) -> str:
    """Format a |Distinctions|."""
    if title is None:
        title = ces.__class__.__name__
    if not ces:
        return "()\n"

    concepts = center("\n".join(margin(x) for x in ces) + "\n")  # type: ignore[attr-defined]
    title = "{} ({} distinction{})".format(title, len(ces), "" if len(ces) == 1 else "s")  # type: ignore[arg-type]

    return header(title, concepts, HEADER_BAR_1, HEADER_BAR_1)


def fmt_distinction(distinction: object) -> str:
    """Format a :class:`Distinction`."""

    def fmt_cause_or_effect(x: object) -> str:  # pylint: disable=missing-docstring
        return indent(str(x), amount=1)

    cause = fmt_cause_or_effect(distinction.cause)  # type: ignore[attr-defined]
    effect = fmt_cause_or_effect(distinction.effect)  # type: ignore[attr-defined]
    ce = side_by_side(cause, effect)

    mechanism = fmt_mechanism(distinction.mechanism, distinction.node_labels)  # type: ignore[attr-defined]
    # TODO(4.0) reconsider using Nodes in the mechanism to facilitate access
    # to their state, etc.
    mech_state = list(distinction.mechanism_state)  # type: ignore[attr-defined]
    title = "\n".join(
        align(
            [
                f"{distinction.__class__.__name__}: mechanism = {mechanism}, "
                f"state = {mech_state}",
                f"{SMALL_PHI} = {fmt_number(distinction.phi)}",  # type: ignore[attr-defined]
            ],
            direction="c",
        )
    )

    # Only center headers for high-verbosity output
    center_bool = config.infrastructure.repr_verbosity is HIGH
    return header(title, ce, HEADER_BAR_2, HEADER_BAR_2, center=center_bool)


# IIT 3.0 paper terminology calls a distinction a "concept"; the alias
# preserves that vocabulary for IIT 3.0-native callers.
fmt_concept = fmt_distinction


def fmt_ria(ria: object, verbose: bool = True, mip: bool = False) -> str:
    """Format a |RepertoireIrreducibilityAnalysis|."""
    if verbose:
        mechanism = f"Mechanism: {fmt_mechanism(ria.mechanism, ria.node_labels)}"  # type: ignore[attr-defined]
        direction = f"Direction: {ria.direction}"  # type: ignore[attr-defined]
    else:
        mechanism = ""
        direction = ""

    # TODO(4.0):  position repertoire and partitioned repertoire side by side
    # TODO(ties) fix state-marking logic
    if config.infrastructure.repr_verbosity is HIGH:
        partition_name = "MIP" if mip else "Partition"
        partition = f"{partition_name}: "
        if ria.partition:  # type: ignore[attr-defined]
            partition += f"\n{indent(fmt_partition(ria.partition))}"  # type: ignore[attr-defined]
        else:
            partition += "empty"
        if ria.specified_state is not None:  # type: ignore[attr-defined]
            mark_states = [specified.state for specified in ria.specified_state.ties]  # type: ignore[attr-defined]
        else:
            mark_states = []
        # TODO(refactor)
        if ria.repertoire is not None:  # type: ignore[attr-defined]
            if ria.repertoire.size == 1:  # type: ignore[attr-defined]
                repertoire = f"Forward probability:\n    {ria.repertoire}"  # type: ignore[attr-defined]
                partitioned_repertoire = (
                    f"Partitioned forward probability:\n    "
                    f"{ria.partitioned_repertoire}"  # type: ignore[attr-defined]
                )
            else:
                rep_fmt = fmt_repertoire(ria.repertoire, mark_states=mark_states)  # type: ignore[attr-defined]
                repertoire = f"Repertoire:\n{indent(rep_fmt)}"
                partitioned_repertoire = "Partitioned repertoire:\n{}".format(
                    indent(
                        fmt_repertoire(
                            ria.partitioned_repertoire,  # type: ignore[attr-defined]
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
            f"{SMALL_PHI} = {fmt_number(ria.phi)}",  # type: ignore[attr-defined]
            f"Normalized {SMALL_PHI} = {fmt_number(ria.normalized_phi)}",  # type: ignore[attr-defined]
            f"{mechanism}",
            f"Purview: {fmt_mechanism(ria.purview, ria.node_labels)}",  # type: ignore[attr-defined]
            f"Specified state:\n{ria.specified_state}",  # type: ignore[attr-defined]
            f"{direction}",
            f"{partition}",
        ]
        + ([f"Selectivity: {ria.selectivity}"] if ria.selectivity is not None else [])  # type: ignore[attr-defined]
        + [
            f"{repertoire}",
            f"{partitioned_repertoire}",
            f"#(state ties): {ria.num_state_ties}",  # type: ignore[attr-defined]
            f"#(partition ties): {ria.num_partition_ties}",  # type: ignore[attr-defined]
        ]
    )
    if hasattr(ria, "num_purview_ties"):
        data.append(f"#(purview ties): {ria.num_purview_ties}")  # type: ignore[attr-defined]
    if ria.reasons is not None:  # type: ignore[attr-defined]
        data.append("Reasons: " + ", ".join(map(str, ria.reasons)))  # type: ignore[attr-defined]
    return "\n".join(data)


def fmt_cut(cut: object, direction: object | None = None, name: bool = True) -> str:
    """Format a |SystemPartition|."""
    try:
        if name:
            name_str = cut.__class__.__name__ + " "
        else:
            name_str = ""
        return "{name}{from_nodes} {symbol} {to_nodes}".format(
            name=name_str,
            from_nodes=fmt_mechanism(cut.from_nodes, cut.node_labels),  # type: ignore[attr-defined]
            symbol=(
                FORWARD_CUT_SYMBOL
                if direction is None
                else CUT_SYMBOLS_BY_DIRECTION[direction]  # type: ignore[index]
            ),
            to_nodes=fmt_mechanism(cut.to_nodes, cut.node_labels),  # type: ignore[attr-defined]
        )
    except AttributeError:
        return str(cut)


def fmt_kcut(cut: object) -> str:
    """Format a |KCut|."""
    return f"KCut {cut.direction}\n{cut.partition}"  # type: ignore[attr-defined]


def fmt_sia_4(
    sia: object,
    phi_structure: bool = True,
    title: str = "System irreducibility analysis",
) -> str:
    """Format an IIT 4.0 |SystemIrreducibilityAnalysis|."""
    if phi_structure:
        body = "\n".join(
            [
                fmt_phi_structure(sia.phi_structure, system=False),  # type: ignore[attr-defined]
                fmt_phi_structure(
                    sia.partitioned_phi_structure,  # type: ignore[attr-defined]
                    title="Partitioned phi-structure",
                    system=False,
                ),
            ]
        )
    else:
        body = ""

    selectivity = sia.selectivity  # type: ignore[attr-defined]
    if selectivity is None:
        selectivity = "[not computed]"
    informativeness = sia.informativeness  # type: ignore[attr-defined]
    if informativeness is None:
        informativeness = "[not computed]"

    lines_list = [
        (BIG_PHI, sia.phi),  # type: ignore[attr-defined]
        ("Selectivity", selectivity),
        ("Informativeness", informativeness),
    ]
    lines = align_columns(lines_list)
    body = "\n".join(["\n".join(lines), body])

    if isinstance(sia.partition, (NullCut, CompleteSystemPartition)):  # type: ignore[attr-defined]
        cut = str(sia.partition)  # type: ignore[attr-defined]
    else:
        cut = fmt_cut(sia.partition, direction=sia.partition.direction, name=False)  # type: ignore[attr-defined]

    data = [
        sia.system.nodes,  # type: ignore[attr-defined]
        cut,
    ]
    if sia.reasons:  # type: ignore[attr-defined]
        data.append("[trivially reducible]\n" + "\n".join(map(str, sia.reasons)))  # type: ignore[attr-defined]
    data.append("")
    for line in reversed(data):
        body = header(str(line), body)
    body = header(title, body, under_char=HEADER_BAR_2)
    return box(center(body))


def fmt_sia(
    sia: object, ces: bool = True, title: str = "System irreducibility analysis"
) -> str:
    """Format a |SystemIrreducibilityAnalysis|."""
    if ces:
        body = "{ces}\n{partitioned_ces}".format(
            ces=fmt_ces(sia.ces, "Cause-effect structure"),  # type: ignore[attr-defined]
            partitioned_ces=fmt_ces(
                sia.partitioned_ces,  # type: ignore[attr-defined]
                "Partitioned cause-effect structure",
            ),
        )
    else:
        body = ""

    data = [
        f"{BIG_PHI}: {fmt_number(sia.phi)}",  # type: ignore[attr-defined]
        sia.system,  # type: ignore[attr-defined]
        sia.cut,  # type: ignore[attr-defined]
    ]
    for line in reversed(data):
        body = header(str(line), body)
    body = header(title, body, under_char=HEADER_BAR_2)
    return box(center(body))


def fmt_repertoire(r: Any, mark_states: list | None = None) -> str:
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
        if mark_states and state in mark_states:
            state_str += " *"
        else:
            state_str += "  "
        lines.append(f"{state_str}{space[:-2]}{fmt_number(r[state])}")

    w = width(lines)
    lines.insert(1, DOTTED_HEADER * (w + 1))

    return box("\n".join(lines))


def fmt_relatum(relatum: object, node_labels: object | None = None) -> str:
    direction = "Cause" if relatum.direction == Direction.CAUSE else "Effect"  # type: ignore[attr-defined]
    return (
        direction
        + fmt_mechanism(relatum.mechanism, node_labels=node_labels)  # type: ignore[attr-defined]
        + "/"
        + fmt_mechanism(relatum.purview, node_labels=node_labels)  # type: ignore[attr-defined]
    )


def fmt_relata(relata: object, node_labels: object | None = None) -> str:
    lines = [fmt_relatum(relatum, node_labels=node_labels) for relatum in relata]  # type: ignore[attr-defined]
    lines = align_columns(lines, delimiter="/", split_columns=True)
    # TODO(4.0) align purview nodes?
    return "\n".join(lines)


def fmt_relation(relation: object) -> str:
    labels = relation.system.node_labels  # type: ignore[attr-defined]
    body = fmt_relata(relation.relata, node_labels=labels)  # type: ignore[attr-defined]
    data_list = [
        ("φ", relation.phi),  # type: ignore[attr-defined]
        ("Purview", fmt_mechanism(relation.purview, node_labels=labels)),  # type: ignore[attr-defined]
        ("Relata", ""),
    ]
    data = "\n".join(align_columns(data_list))
    body = center(header(data, body))
    return header("Relation", body, over_char=HEADER_BAR_3, under_char=HEADER_BAR_3)


def _fmt_relations(
    relations: object, title: str | None = None, body: str = "", data: list | None = None
) -> str:
    if title is None:
        title = relations.__class__.__name__
    if data is None:
        data = []
    data_list = [
        ("#", len(relations)),  # type: ignore[arg-type]
        ("Σφ", relations.sum_phi()),  # type: ignore[attr-defined]
        *data,
    ]
    data_str = "\n".join(align_columns(data_list))
    body = header(data_str, body)
    body = header(title, body, under_char=HEADER_BAR_1)
    return center(body)


def fmt_concrete_relations(relations: object, title: str | None = None) -> str:
    body = "\n".join(map(fmt_relation, relations))  # type: ignore[call-overload]
    return _fmt_relations(relations, title, body)


def fmt_analytical_relations(relations: object, title: str | None = None) -> str:
    body = ""
    return _fmt_relations(relations, title, body)


def fmt_sampled_relations(relations: object, title: str | None = None) -> str:
    body = "\n".join(map(fmt_relation, relations.sample))  # type: ignore[attr-defined]
    return _fmt_relations(
        relations,
        title,
        body,
        data=[("Sampled", len(relations.sample))],  # type: ignore[attr-defined]
    )


def fmt_extended_purview(
    extended_purview: Sequence, node_labels: object | None = None
) -> str:
    """Format an extended purview."""
    if len(extended_purview) == 1:
        return fmt_mechanism(extended_purview[0], node_labels=node_labels)

    purviews = [
        fmt_mechanism(purview, node_labels=node_labels) for purview in extended_purview
    ]
    return "[" + ", ".join(purviews) + "]"


def fmt_causal_link(causal_link: object) -> str:
    """Format a CausalLink."""
    return fmt_ac_ria(causal_link, extended_purview=causal_link.extended_purview)  # type: ignore[attr-defined]


def fmt_ac_ria(ria: object, extended_purview: object | None = None) -> str:
    """Format an AcRepertoireIrreducibilityAnalysis."""
    causality = {
        Direction.CAUSE: (
            (
                fmt_mechanism(ria.purview, ria.node_labels)  # type: ignore[attr-defined]
                if extended_purview is None
                else fmt_extended_purview(ria.extended_purview, ria.node_labels)  # type: ignore[attr-defined]
            ),
            ARROW_LEFT,
            fmt_mechanism(ria.mechanism, ria.node_labels),  # type: ignore[attr-defined]
        ),
        Direction.EFFECT: (
            fmt_mechanism(ria.mechanism, ria.node_labels),  # type: ignore[attr-defined]
            ARROW_RIGHT,
            (
                fmt_mechanism(ria.purview, ria.node_labels)  # type: ignore[attr-defined]
                if extended_purview is None
                else fmt_extended_purview(ria.extended_purview, ria.node_labels)  # type: ignore[attr-defined]
            ),
        ),
    }[ria.direction]  # type: ignore[attr-defined]
    causality_str = " ".join(causality)

    return f"{ALPHA} = {round(ria.alpha, 4)}  {causality_str}"  # type: ignore[attr-defined]


def fmt_account(account: object, title: str | None = None) -> str:
    """Format an Account or a DirectedAccount."""
    if title is None:
        title = account.__class__.__name__  # `Account` or `DirectedAccount`

    title = "{} ({} causal link{})".format(
        title,
        len(account),  # type: ignore[arg-type]
        "" if len(account) == 1 else "s",  # type: ignore[arg-type]
    )

    body = ""
    body += "Irreducible effects\n"
    body += "\n".join(fmt_causal_link(m) for m in account.irreducible_effects)  # type: ignore[attr-defined]
    body += "\nIrreducible causes\n"
    body += "\n".join(fmt_causal_link(m) for m in account.irreducible_causes)  # type: ignore[attr-defined]

    return "\n" + header(title, body, under_char="*")


def fmt_ac_sia(ac_sia: object) -> str:
    """Format a AcSystemIrreducibilityAnalysis."""
    # Extract attributes explicitly for type checking
    direction_val = ac_sia.direction  # type: ignore[attr-defined]
    transition_val = ac_sia.transition  # type: ignore[attr-defined]
    before_state_val = ac_sia.before_state  # type: ignore[attr-defined]
    after_state_val = ac_sia.after_state  # type: ignore[attr-defined]
    cut_val = ac_sia.cut  # type: ignore[attr-defined]
    alpha_val = round(ac_sia.alpha, 4)  # type: ignore[attr-defined]

    body = (
        "{ALPHA} = {alpha}\n"
        "direction: {direction}\n"
        "transition: {transition}\n"
        "before state: {before_state}\n"
        "after state: {after_state}\n"
        "cut:\n{cut}\n"
        "{account}\n"
        "{partitioned_account}".format(
            ALPHA=ALPHA,
            alpha=alpha_val,
            direction=direction_val,
            transition=transition_val,
            before_state=before_state_val,
            after_state=after_state_val,
            cut=cut_val,
            account=fmt_account(ac_sia.account, "Account"),  # type: ignore[attr-defined]
            partitioned_account=fmt_account(
                ac_sia.partitioned_account,  # type: ignore[attr-defined]
                "Partitioned Account",
            ),
        )
    )

    return box(header("AcSystemIrreducibilityAnalysis", body, under_char=HORIZONTAL_BAR))


def fmt_transition(t: object) -> str:
    """Format a |Transition|."""
    cause = fmt_mechanism(t.cause_indices, t.node_labels)  # type: ignore[attr-defined]
    effect = fmt_mechanism(t.effect_indices, t.node_labels)  # type: ignore[attr-defined]
    return f"Transition({cause} {ARROW_RIGHT} {effect})"


def state(state: tuple[int, ...]) -> str:
    """Format a state."""
    return "(" + ",".join(map(str, state)) + ")"
