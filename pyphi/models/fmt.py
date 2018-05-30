#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/fmt.py

"""
Helper functions for formatting pretty representations of PyPhi models.
"""

from fractions import Fraction
from itertools import chain

from .. import Direction, config, constants, utils

# pylint: disable=bad-whitespace

# REPR_VERBOSITY levels
LOW    = 0
MEDIUM = 1
HIGH   = 2

# Unicode symbols
SMALL_PHI           = '\u03C6'
BIG_PHI             = '\u03A6'
ALPHA               = '\u03B1'
TOP_LEFT_CORNER     = '\u250C'
TOP_RIGHT_CORNER    = '\u2510'
BOTTOM_LEFT_CORNER  = '\u2514'
BOTTOM_RIGHT_CORNER = '\u2518'
HORIZONTAL_BAR      = '\u2500'
VERTICAL_SIDE       = '\u2502'
HEADER_BAR_1        = '\u2550'
HEADER_BAR_2        = '\u2501'
HEADER_BAR_3        = '\u254D'
DOTTED_HEADER       = '\u2574'
LINE                = '\u2501'
CUT_SYMBOL          = LINE * 2 + '/ /' + LINE * 2 + '\u27A4'
EMPTY_SET           = '\u2205'
MULTIPLY            = '\u2715'
ARROW_LEFT          = '\u25C0' + LINE * 2
ARROW_RIGHT         = LINE * 2 + '\u25B6'

NICE_DENOMINATORS   = list(range(16)) + [16, 32, 64, 128]

# pylint: enable=bad-whitespace


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
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(attr + '=' + repr(getattr(self, attr))
                      for attr in attrs))

    raise ValueError('Invalid value for `config.REPR_VERBOSITY`')


def indent(lines, amount=2, char=' '):
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
    return padding + ('\n' + padding).join(lines.split('\n'))


def margin(text):
    r"""Add a margin to both ends of each line in the string.

    Example:
        >>> margin('line1\nline2')
        '  line1  \n  line2  '
    """
    lines = str(text).split('\n')
    return '\n'.join('  {}  '.format(l) for l in lines)


LINES_FORMAT_STR = VERTICAL_SIDE + ' {line:<{width}} ' + VERTICAL_SIDE


def box(text):
    r"""Wrap a chunk of text in a box.

    Example:
        >>> print(box('line1\nline2'))
        ┌───────┐
        │ line1 │
        │ line2 │
        └───────┘
    """
    lines = text.split('\n')

    width = max(len(l) for l in lines)
    top_bar = (TOP_LEFT_CORNER + HORIZONTAL_BAR * (2 + width) +
               TOP_RIGHT_CORNER)
    bottom_bar = (BOTTOM_LEFT_CORNER + HORIZONTAL_BAR * (2 + width) +
                  BOTTOM_RIGHT_CORNER)

    lines = [LINES_FORMAT_STR.format(line=line, width=width) for line in lines]

    return top_bar + '\n' + '\n'.join(lines) + '\n' + bottom_bar


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
    left_lines = list(left.split('\n'))
    right_lines = list(right.split('\n'))

    # Pad the shorter column with whitespace
    diff = abs(len(left_lines) - len(right_lines))
    if len(left_lines) > len(right_lines):
        fill = ' ' * len(right_lines[0])
        right_lines += [fill] * diff
    elif len(right_lines) > len(left_lines):
        fill = ' ' * len(left_lines[0])
        left_lines += [fill] * diff

    return '\n'.join(a + b for a, b in zip(left_lines, right_lines)) + '\n'


def header(head, text, over_char=None, under_char=None, center=True):
    """Center a head over a block of text.

    The width of the text is the width of the longest line of the text.
    """
    lines = list(text.split('\n'))
    width = max(len(l) for l in lines)

    # Center or left-justify
    if center:
        head = head.center(width) + '\n'
    else:
        head = head.ljust(width) + '\n'

    # Underline head
    if under_char:
        head = head + under_char * width + '\n'

    # 'Overline' head
    if over_char:
        head = over_char * width + '\n' + head

    return head + text


def labels(indices, node_labels=None):
    """Get the labels for a tuple of mechanism indices."""
    if node_labels is None:
        return tuple(map(str, indices))
    return node_labels.indices2labels(indices)


def fmt_number(p):
    """Format a number.

    It will be printed as a fraction if the denominator isn't too big and as a
    decimal otherwise.
    """
    formatted = '{:n}'.format(p)

    if not config.PRINT_FRACTIONS:
        return formatted

    fraction = Fraction(p)
    nice = fraction.limit_denominator(128)
    return (
        str(nice) if (abs(fraction - nice) < constants.EPSILON and
                      nice.denominator in NICE_DENOMINATORS)
        else formatted
    )


def fmt_mechanism(indices, node_labels=None):
    """Format a mechanism or purview."""
    return '[' + ', '.join(labels(indices, node_labels)) + ']'


def fmt_part(part, node_labels=None):
    """Format a |Part|.

    The returned string looks like::

        0,1
        ───
         ∅
    """
    def nodes(x):  # pylint: disable=missing-docstring
        return ','.join(labels(x, node_labels)) if x else EMPTY_SET

    numer = nodes(part.mechanism)
    denom = nodes(part.purview)

    width = max(3, len(numer), len(denom))
    divider = HORIZONTAL_BAR * width

    return (
        '{numer:^{width}}\n'
        '{divider}\n'
        '{denom:^{width}}'
    ).format(numer=numer, divider=divider, denom=denom, width=width)


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
    if not partition:
        return ''

    parts = [fmt_part(part, partition.node_labels).split('\n')
             for part in partition]

    times = ('   ',
             ' {} '.format(MULTIPLY),
             '   ')
    breaks = ('\n', '\n', '')  # No newline at the end of string
    between = [times] * (len(parts) - 1) + [breaks]

    # Alternate [part, break, part, ..., end]
    elements = chain.from_iterable(zip(parts, between))

    # Transform vertical stacks into horizontal lines
    return ''.join(chain.from_iterable(zip(*elements)))


def fmt_ces(c, title=None):
    """Format a |CauseEffectStructure|."""
    if not c:
        return '()\n'

    if title is None:
        title = 'Cause-effect structure'

    concepts = '\n'.join(margin(x) for x in c) + '\n'
    title = '{} ({} concept{})'.format(
        title, len(c), '' if len(c) == 1 else 's')

    return header(title, concepts, HEADER_BAR_1, HEADER_BAR_1)


def fmt_concept(concept):
    """Format a |Concept|."""

    def fmt_cause_or_effect(x):  # pylint: disable=missing-docstring
        return box(indent(fmt_ria(x.ria, verbose=False, mip=True), amount=1))

    cause = header('MIC', fmt_cause_or_effect(concept.cause))
    effect = header('MIE', fmt_cause_or_effect(concept.effect))
    ce = side_by_side(cause, effect)

    mechanism = fmt_mechanism(concept.mechanism, concept.node_labels)
    title = 'Concept: Mechanism = {}, {} = {}'.format(
        mechanism, SMALL_PHI, fmt_number(concept.phi))

    # Only center headers for high-verbosity output
    center = config.REPR_VERBOSITY is HIGH

    return header(title, ce, HEADER_BAR_2, HEADER_BAR_2, center=center)


def fmt_ria(ria, verbose=True, mip=False):
    """Format a |RepertoireIrreducibilityAnalysis|."""
    if verbose:
        mechanism = 'Mechanism: {}\n'.format(
            fmt_mechanism(ria.mechanism, ria.node_labels))
        direction = '\nDirection: {}'.format(ria.direction)
    else:
        mechanism = ''
        direction = ''

    if config.REPR_VERBOSITY is HIGH:
        partition = '\n{}:\n{}'.format(
            ('MIP' if mip else 'Partition'),
            indent(fmt_partition(ria.partition)))
        repertoire = '\nRepertoire:\n{}'.format(
            indent(fmt_repertoire(ria.repertoire)))
        partitioned_repertoire = '\nPartitioned repertoire:\n{}'.format(
            indent(fmt_repertoire(ria.partitioned_repertoire)))
    else:
        partition = ''
        repertoire = ''
        partitioned_repertoire = ''

    # TODO? print the two repertoires side-by-side
    return (
        '{SMALL_PHI} = {phi}\n'
        '{mechanism}'
        'Purview = {purview}'
        '{direction}'
        '{partition}'
        '{repertoire}'
        '{partitioned_repertoire}').format(
            SMALL_PHI=SMALL_PHI,
            mechanism=mechanism,
            purview=fmt_mechanism(ria.purview, ria.node_labels),
            direction=direction,
            phi=fmt_number(ria.phi),
            partition=partition,
            repertoire=repertoire,
            partitioned_repertoire=partitioned_repertoire)


def fmt_cut(cut):
    """Format a |Cut|."""
    return 'Cut {from_nodes} {symbol} {to_nodes}'.format(
        from_nodes=fmt_mechanism(cut.from_nodes, cut.node_labels),
        symbol=CUT_SYMBOL,
        to_nodes=fmt_mechanism(cut.to_nodes, cut.node_labels))


def fmt_kcut(cut):
    """Format a |KCut|."""
    return 'KCut {}\n{}'.format(cut.direction, cut.partition)


def fmt_sia(sia, ces=True):
    """Format a |SystemIrreducibilityAnalysis|."""
    if ces:
        body = (
            '{ces}'
            '{partitioned_ces}'.format(
                ces=fmt_ces(
                    sia.ces,
                    'Cause-effect structure'),
                partitioned_ces=fmt_ces(
                    sia.partitioned_ces,
                    'Partitioned cause-effect structure')))
        center_header = True
    else:
        body = ''
        center_header = False

    title = 'System irreducibility analysis: {BIG_PHI} = {phi}'.format(
        BIG_PHI=BIG_PHI, phi=fmt_number(sia.phi))

    body = header(str(sia.subsystem), body, center=center_header)
    body = header(str(sia.cut), body, center=center_header)
    return box(header(title, body, center=center_header))


def fmt_repertoire(r):
    """Format a repertoire."""
    # TODO: will this get unwieldy with large repertoires?
    if r is None:
        return ''

    r = r.squeeze()

    lines = []

    # Header: 'S      P(S)'
    space = ' ' * 4
    head = '{S:^{s_width}}{space}Pr({S})'.format(
        S='S', s_width=r.ndim, space=space)
    lines.append(head)

    # Lines: '001     .25'
    for state in utils.all_states(r.ndim):
        state_str = ''.join(str(i) for i in state)
        lines.append('{0}{1}{2}'.format(state_str, space,
                                        fmt_number(r[state])))

    width = max(len(line) for line in lines)
    lines.insert(1, DOTTED_HEADER * (width + 1))

    return box('\n'.join(lines))


def fmt_ac_ria(ria):
    """Format an AcRepertoireIrreducibilityAnalysis."""
    causality = {
        Direction.CAUSE: (fmt_mechanism(ria.purview, ria.node_labels),
                          ARROW_LEFT,
                          fmt_mechanism(ria.mechanism, ria.node_labels)),
        Direction.EFFECT: (fmt_mechanism(ria.mechanism, ria.node_labels),
                           ARROW_RIGHT,
                           fmt_mechanism(ria.purview, ria.node_labels))
    }[ria.direction]
    causality = ' '.join(causality)

    return '{ALPHA} = {alpha}  {causality}'.format(
        ALPHA=ALPHA,
        alpha=round(ria.alpha, 4),
        causality=causality)


def fmt_account(account, title=None):
    """Format an Account or a DirectedAccount."""
    if title is None:
        title = account.__class__.__name__  # `Account` or `DirectedAccount`

    title = '{} ({} causal link{})'.format(
        title, len(account), '' if len(account) == 1 else 's')

    body = ''
    body += 'Irreducible effects\n'
    body += '\n'.join(fmt_ac_ria(m) for m in account.irreducible_effects)
    body += '\nIrreducible causes\n'
    body += '\n'.join(fmt_ac_ria(m) for m in account.irreducible_causes)

    return '\n' + header(title, body, under_char='*')


def fmt_ac_sia(ac_sia):
    """Format a AcSystemIrreducibilityAnalysis."""
    body = (
        '{ALPHA} = {alpha}\n'
        'direction: {ac_sia.direction}\n'
        'transition: {ac_sia.transition}\n'
        'before state: {ac_sia.before_state}\n'
        'after state: {ac_sia.after_state}\n'
        'cut:\n{ac_sia.cut}\n'
        '{account}\n'
        '{partitioned_account}'.format(
            ALPHA=ALPHA,
            alpha=round(ac_sia.alpha, 4),
            ac_sia=ac_sia,
            account=fmt_account(
                ac_sia.account, 'Account'),
            partitioned_account=fmt_account(
                ac_sia.partitioned_account, 'Partitioned Account')))

    return box(header('AcSystemIrreducibilityAnalysis',
                      body,
                      under_char=HORIZONTAL_BAR))


def fmt_transition(t):
    """Format a |Transition|."""
    return "Transition({} {} {})".format(
        fmt_mechanism(t.cause_indices, t.node_labels),
        ARROW_RIGHT,
        fmt_mechanism(t.effect_indices, t.node_labels))
