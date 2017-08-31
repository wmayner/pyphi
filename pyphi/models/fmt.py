#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/fmt.py

'''
Helper functions for formatting pretty representations of PyPhi models.
'''

from fractions import Fraction
from itertools import chain

from .. import config, constants, utils

# pylint: disble=bad-whitespace

# REPR_VERBOSITY levels
LOW    = 0
MEDIUM = 1
HIGH   = 2

# Unicode symbols
SMALL_PHI           = '\u03C6'
BIG_PHI             = '\u03A6'
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
CUT_SYMBOL          = '\u2501' * 2 + '/ /' + '\u2501' * 2 + '\u25B6'
EMPTY_SET           = '\u2205'
MULTIPLY            = '\u2715'

NICE_DENOMINATORS   = list(range(16)) + [16, 32, 64, 128]

# pylint: enable=bad-whitespace


def make_repr(self, attrs):
    '''Construct a repr string.

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
    '''
    # TODO: change this to a closure so we can do
    # __repr__ = make_repr(attrs) ???

    if config.REPR_VERBOSITY in [MEDIUM, HIGH]:
        return self.__str__()

    elif config.REPR_VERBOSITY is LOW:
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join(attr + '=' + repr(getattr(self, attr))
                      for attr in attrs))

    raise ValueError('Invalid `REPR_VERBOSITY` value of {}. Must be one of '
                     '[0, 1, 2]'.format(config.REPR_VERBOSITY))


def indent(lines, amount=2, char=' '):
    '''Indent a string.

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
        >>> print(indent('line1\\nline2', char='*'))
        **line1
        **line2
    '''
    lines = str(lines)
    padding = amount * char
    return padding + ('\n' + padding).join(lines.split('\n'))


def margin(text, amount=2):
    '''Add a margin to both ends of each line in the string.'''
    lines = str(text).split('\n')
    return '\n'.join('  {}  '.format(l) for l in lines)


LINES_FORMAT_STR = VERTICAL_SIDE + ' {line:<{width}} ' + VERTICAL_SIDE


def box(text):
    '''Wrap a chunk of text in a box.

    Example:
        >>> print(box('line1\\nline2'))
        ┌───────┐
        │ line1 │
        │ line2 │
        └───────┘
    '''
    lines = text.split('\n')

    width = max(len(l) for l in lines)
    top_bar = (TOP_LEFT_CORNER + HORIZONTAL_BAR * (2 + width) +
               TOP_RIGHT_CORNER)
    bottom_bar = (BOTTOM_LEFT_CORNER + HORIZONTAL_BAR * (2 + width) +
                  BOTTOM_RIGHT_CORNER)

    lines = [LINES_FORMAT_STR.format(line=line, width=width) for line in lines]

    return top_bar + '\n' + '\n'.join(lines) + '\n' + bottom_bar


def side_by_side(left, right):
    '''Put two boxes next to each other.

    Assumes that all lines in the boxes are the same width.

    Example:
        >>> left = 'A \\nC '
        >>> right = 'B\\nD'
        >>> print(side_by_side(left, right))
        A B
        C D
        <BLANKLINE>
    '''
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
    '''Center a head over a block of text.

    The width of the text is the width of the longest line of the text.
    '''
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


def labels(indices, subsystem=None):
    '''Get the labels for a tuple of mechanism indices.'''
    if subsystem is None:
        return tuple(map(str, indices))
    return subsystem.indices2labels(indices)


def fmt_number(p):
    '''Format a number.

    It will be printed as a fraction if the denominator isn't too big and as a
    decimal otherwise.
    '''
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


def fmt_mechanism(indices, subsystem=None):
    '''Format a mechanism or purview.'''
    return '[' + ', '.join(labels(indices, subsystem)) + ']'


def fmt_part(part, subsystem=None):
    '''Format a |Part|.

    The returned string looks like::

        0,1
        ───
         ∅
    '''
    def nodes(x):  # pylint: disable=missing-docstring
        return ','.join(labels(x, subsystem)) if x else EMPTY_SET

    numer = nodes(part.mechanism)
    denom = nodes(part.purview)

    width = max(3, len(numer), len(denom))
    divider = HORIZONTAL_BAR * width

    return (
        '{numer:^{width}}\n'
        '{divider}\n'
        '{denom:^{width}}'
    ).format(numer=numer, divider=divider, denom=denom, width=width)


def fmt_bipartition(partition, subsystem=None):
    '''Format a |Bipartition|.

    The returned string looks like::

        0,1    ∅
        ─── ✕ ───
         2    0,1

    Args:
        partition (Bipartition): The partition in question.

    Returns:
        str: A human-readable string representation of the partition.
    '''
    if not partition:
        return ''

    parts = [fmt_part(part, subsystem).split('\n') for part in partition]

    times = ('   ',
             ' {} '.format(MULTIPLY),
             '   ')
    breaks = ('\n', '\n', '')  # No newline at the end of string
    between = [times] * (len(parts) - 1) + [breaks]

    # Alternate [part, break, part, ..., end]
    elements = chain.from_iterable(zip(parts, between))

    # Transform vertical stacks into horizontal lines
    return ''.join(chain.from_iterable(zip(*elements)))


def fmt_constellation(c, title=None):
    '''Format a constellation.'''
    if not c:
        return '()\n'

    if title is None:
        title = 'Constellation'

    concepts = '\n'.join(margin(x) for x in c) + '\n'
    title = '{} ({} concept{})'.format(
        title, len(c), '' if len(c) == 1 else 's')

    return header(title, concepts, HEADER_BAR_1, HEADER_BAR_1)


def fmt_concept(concept):
    '''Format a |Concept|.'''

    def fmt_cause_or_effect(x):  # pylint: disable=missing-docstring
        if not x:
            return ''
        return box(indent(fmt_mip(x.mip, verbose=False), amount=1))

    cause = header('Cause', fmt_cause_or_effect(concept.cause))
    effect = header('Effect', fmt_cause_or_effect(concept.effect))
    ce = side_by_side(cause, effect)

    mechanism = fmt_mechanism(concept.mechanism, concept.subsystem)
    title = 'Concept: Mechanism = {}, {} = {}'.format(
        mechanism, SMALL_PHI, fmt_number(concept.phi))

    # Only center headers for high-verbosity output
    center = config.REPR_VERBOSITY is HIGH

    return header(title, ce, HEADER_BAR_2, HEADER_BAR_2, center=center)


def fmt_mip(mip, verbose=True):
    '''Format a |Mip|.'''
    if mip is False or mip is None:  # mips can be Falsy
        return ''

    if verbose:
        mechanism = 'Mechanism: {}\n'.format(
            fmt_mechanism(mip.mechanism, mip.subsystem))
        direction = '\nDirection: {}\n'.format(mip.direction)
    else:
        mechanism = ''
        direction = ''

    if config.REPR_VERBOSITY is HIGH:
        partition = '\nPartition:\n{}'.format(
            indent(fmt_bipartition(mip.partition, mip.subsystem)))
        unpartitioned_repertoire = '\nUnpartitioned Repertoire:\n{}'.format(
            indent(fmt_repertoire(mip.unpartitioned_repertoire)))
        partitioned_repertoire = '\nPartitioned Repertoire:\n{}'.format(
            indent(fmt_repertoire(mip.partitioned_repertoire)))
    else:
        partition = ''
        unpartitioned_repertoire = ''
        partitioned_repertoire = ''

    # TODO? print the two repertoires side-by-side
    return (
        '{SMALL_PHI} = {phi}\n'
        '{mechanism}'
        'Purview = {purview}'
        '{partition}'
        '{direction}'
        '{unpartitioned_repertoire}'
        '{partitioned_repertoire}').format(
            SMALL_PHI=SMALL_PHI,
            mechanism=mechanism,
            purview=fmt_mechanism(mip.purview, mip.subsystem),
            direction=direction,
            phi=fmt_number(mip.phi),
            partition=partition,
            unpartitioned_repertoire=unpartitioned_repertoire,
            partitioned_repertoire=partitioned_repertoire)


def fmt_cut(cut, subsystem=None):
    '''Format a |Cut|.'''
    # Cut indices cannot be converted to labels for macro systems since macro
    # systems are cut at the micro label. Avoid this error by using micro
    # indices directly in the representation.
    # TODO: somehow handle this with inheritance instead of a conditional?
    from ..macro import MacroSubsystem
    if isinstance(subsystem, MacroSubsystem):
        from_nodes = str(cut.from_nodes)
        to_nodes = str(cut.to_nodes)
    else:
        from_nodes = fmt_mechanism(cut.from_nodes, subsystem)
        to_nodes = fmt_mechanism(cut.to_nodes, subsystem)

    return 'Cut {from_nodes} {symbol} {to_nodes}'.format(
        from_nodes=from_nodes, symbol=CUT_SYMBOL, to_nodes=to_nodes)


def fmt_big_mip(big_mip, constellations=True):
    '''Format a |BigMip|.'''
    if constellations:
        body = (
            '{unpartitioned_constellation}'
            '{partitioned_constellation}'.format(
                unpartitioned_constellation=fmt_constellation(
                    big_mip.unpartitioned_constellation,
                    'Unpartitioned Constellation'),
                partitioned_constellation=fmt_constellation(
                    big_mip.partitioned_constellation,
                    'Partitioned Constellation')))
        center_header = True
    else:
        body = ''
        center_header = False

    title = 'Big Mip: {BIG_PHI} = {phi}'.format(
        BIG_PHI=BIG_PHI, phi=fmt_number(big_mip.phi))

    cut = fmt_cut(big_mip.cut, big_mip.subsystem)

    body = header(str(big_mip.subsystem), body, center=center_header)
    body = header(fmt_cut(big_mip.cut), body, center=center_header)
    return box(header(title, body, center=center_header))


def fmt_repertoire(r):
    '''Format a repertoire.'''
    # TODO: will this get unwieldy with large repertoires?
    if r is None:
        return ''

    r = r.squeeze()

    lines = []

    # Header: 'S      P(S)'
    space = ' ' * 4
    head = '{S:^{s_width}}{space}P({S})'.format(
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


def fmt_ac_mip(acmip, verbose=True):
    '''Helper function to format a nice Mip string'''

    if acmip is False or acmip is None:  # mips can be Falsy
        return ''

    mechanism = 'mechanism: {}\t'.format(acmip.mechanism) if verbose else ''
    direction = 'direction: {}\n'.format(acmip.direction) if verbose else ''
    return (
        '{alpha}\t'
        '{mechanism}'
        'purview: {acmip.purview}\t'
        '{direction}'
        'partition:\n{partition}\n'
        'probability:\t{probability}\t'
        'partitioned_probability:\t{partitioned_probability}\n').format(
            alpha='{0:.4f}'.format(round(acmip.alpha, 4)),
            mechanism=mechanism,
            direction=direction,
            acmip=acmip,
            partition=indent(fmt_bipartition(acmip.partition)),
            probability=indent(acmip.probability),
            partitioned_probability=indent(acmip.partitioned_probability))


def fmt_ac_big_mip(ac_big_mip):
    '''Format a AcBigMip.'''
    return (
        '{alpha}\n'
        'direction: {ac_big_mip.direction}\n'
        'context: {ac_big_mip.context}\n'
        'past_state: {ac_big_mip.before_state}\n'
        'current_state: {ac_big_mip.after_state}\n'
        'cut: {ac_big_mip.cut}\n'
        '{unpartitioned_account}'
        '{partitioned_account}'.format(
            alpha='{0:.4f}'.format(round(ac_big_mip.alpha, 4)),
            ac_big_mip=ac_big_mip,
            unpartitioned_account=fmt_account(
                ac_big_mip.unpartitioned_account, 'Unpartitioned Account'),
            partitioned_account=fmt_account(
                ac_big_mip.partitioned_account, 'Partitioned Account')))


def fmt_account(account, title=None):
    '''Format an Account or a DirectedAccount.'''

    if title is None:
        title = account.__class__.__name__  # `Account` or `DirectedAccount`

    title = '{} ({} coefficient{})'.format(
        title, len(account), '' if len(account) == 1 else 's')

    content = '\n'.join(str(m) for m in account)

    return '\n' + header(title, content, under_char='*')


def fmt_actual_cut(cut):
    '''Format an ActualCut.'''
    return (
        '{cut.cause_part1} {symbol} {cut.effect_part2} && '
        '{cut.cause_part2} {symbol} {cut.effect_part1}'
    ).format(cut=cut, symbol=CUT_SYMBOL)
