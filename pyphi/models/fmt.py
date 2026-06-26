# models/fmt.py
"""Helper functions for formatting pretty representations of PyPhi models."""

from collections.abc import Iterable
from fractions import Fraction
from itertools import chain
from itertools import cycle
from typing import Any

import numpy as np

from pyphi import utils
from pyphi.conf import config

# Unicode symbols
SMALL_PHI = "φ"
BIG_PHI = "Φ"
ALPHA = "\u03b1"
HORIZONTAL_BAR = "─"
DOTTED_HEADER = "╴"
LINE = "━"
ARROW_LEFT = "◀" + LINE * 2
ARROW_RIGHT = LINE * 2 + "▶"
EMPTY_SET = "∅"

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
    LOW = 0
    MEDIUM = 1
    HIGH = 2

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


LINES_FORMAT_STR = "│" + " {line:<{width}} " + "│"


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
    top_bar = "┌" + HORIZONTAL_BAR * (2 + w) + "┐"
    bottom_bar = "└" + HORIZONTAL_BAR * (2 + w) + "┘"

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
        chain.from_iterable(
            [text.split("\n") if "\n" in text else [text] for text in lines]
        )
    )
    w = width(lines)
    if direction == "c":
        return [line.center(w) for line in lines]
    spec = f" {direction}{w}"
    return [format(line, spec) for line in lines]


def center(text: str) -> str:
    """Center-align a string."""
    return "\n".join(align(text.split("\n"), direction="c"))


def _split_decimal(n: Any) -> list[str]:  # noqa: PLR0911
    """Split an object into unit and decimal parts, handling non-numeric types.

    Always returns a 2-element list [units, decimals].
    """
    if n is None:
        return ["", str(None)]
    try:
        if np.isnan(n):
            return ["", str(n)]
    except TypeError:
        pass
    if isinstance(n, float):
        parts = str(float(n)).split(".")
        if len(parts) == 1:
            return [parts[0], ""]
        return parts
    try:
        n = float(n)
        if n.is_integer():
            return [str(int(n)), ""]
        parts = str(n).split(".")
        if len(parts) == 1:
            return [parts[0], ""]
        return parts
    except ValueError:
        pass
    return ["", str(n)]


def _align_decimals(numbers: Iterable[Any]) -> list[str]:
    """Align numbers on the decimal point."""
    numbers_list = list(numbers)
    if not numbers_list:
        return []
    units_tuple, decimals_tuple = zip(*map(_split_decimal, numbers_list), strict=False)
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


def _expand_multiline_strings(left: str, right: str) -> list[tuple[str, str]]:
    """Expand a multiline 'right side' string into a list of columns with empty
    left sides.
    """
    if not isinstance(right, str) or "\n" not in right:
        return [(left, right)]
    columns = [("", line) for line in right.split("\n")]
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
    lines = chain.from_iterable(
        [_expand_multiline_strings(left, right) for left, right in lines]
    )
    # Reorient into columns
    columns: list[Any] = list(zip(*lines, strict=False))
    for i, t in enumerate(types):
        if t == "n":
            columns[i] = _align_decimals(columns[i])
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


def fmt_part(part: object, node_labels: object | None = None) -> str:
    """Format a |Part|.

    The returned string looks like::

        0,1
        ───
         ∅
    """
    numer = fmt_nodes(part.mechanism, node_labels=node_labels)  # type: ignore[attr-defined]
    denom = fmt_nodes(part.purview, node_labels=node_labels)  # type: ignore[attr-defined]

    w = max(3, len(numer), len(denom))
    divider = HORIZONTAL_BAR * w
    return ("{numer:^{width}}\n{divider}\n{denom:^{width}}").format(
        numer=numer, divider=divider, denom=denom, width=w
    )


def fmt_partition(partition: object) -> str:
    """Format a |JointBipartition|.

    The returned string looks like::

        0,1    ∅
        ─── ✕ ───
         2    0,1

    Args:
        partition (JointBipartition): The partition in question.

    Returns:
        str: A human-readable string representation of the partition.
    """
    # TODO: deprecate
    MULTIPLY = "✕"

    if not partition:
        return ""
    try:
        parts = [
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


def fmt_partition_arrow(
    cut: object, direction: object | None = None, name: bool = True
) -> str:
    """Format a |DirectedBipartition| as an arrow expression."""
    from pyphi.direction import Direction

    CUT_SYMBOLS_BY_DIRECTION = {
        Direction.CAUSE: ARROW_LEFT + "/ /" + LINE * 2,
        Direction.EFFECT: LINE * 2 + "/ /" + ARROW_RIGHT,
    }

    try:
        if name:
            name_str = cut.__class__.__name__ + " "
        else:
            name_str = ""
        from_nodes = fmt_mechanism(cut.from_nodes, cut.node_labels)  # type: ignore[attr-defined]
        to_nodes = fmt_mechanism(cut.to_nodes, cut.node_labels)  # type: ignore[attr-defined]
        symbol = (
            ARROW_RIGHT if direction is None else CUT_SYMBOLS_BY_DIRECTION[direction]  # type: ignore[index]
        )
        return f"{name_str}{from_nodes} {symbol} {to_nodes}"
    except AttributeError:
        return str(cut)


def fmt_directed_joint_partition(cut: object) -> str:
    """Format a |DirectedJointPartition|."""
    return f"DirectedJointPartition {cut.direction}\n{cut.partition}"  # type: ignore[attr-defined]


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


def fmt_extended_purview(
    extended_purview: Any, node_labels: object | None = None
) -> str:
    """Format an extended purview."""
    if len(extended_purview) == 1:
        return fmt_mechanism(extended_purview[0], node_labels=node_labels)

    purviews = [
        fmt_mechanism(purview, node_labels=node_labels) for purview in extended_purview
    ]
    return "[" + ", ".join(purviews) + "]"


def fmt_transition(t: object) -> str:
    """Format a |Transition|."""
    cause = fmt_mechanism(t.cause_indices, t.node_labels)  # type: ignore[attr-defined]
    effect = fmt_mechanism(t.effect_indices, t.node_labels)  # type: ignore[attr-defined]
    return f"Transition({cause} {ARROW_RIGHT} {effect})"


def state(state: tuple[int, ...]) -> str:
    """Format a state."""
    return "(" + ",".join(map(str, state)) + ")"
