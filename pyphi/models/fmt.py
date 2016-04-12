#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# fmt.py

"""
Helper functions for formatting pretty representations of PyPhi models.
"""

from .. import config


def make_repr(self, attrs):
    """Construct a repr string.

    If `config.READABLE_REPRS` is True, this function calls out
    to the object's __str__ method. Although this breaks the convention
    that __repr__ should return a string which can reconstruct the object,
    readable reprs are invaluable since the Python interpreter calls
    `repr` to represent all objects in the shell. Since PyPhi is often
    used in the interpreter we want to have meaningful and useful
    representations.

    Args:
        self (obj): The object in question
        attrs (iterable(str)): Attributes to include in the repr

    Returns:
        (str): the `repr`esentation of the object
    """
    # TODO: change this to a closure so we can do
    # __repr__ = make_repr(attrs) ???

    if config.READABLE_REPRS:
        return self.__str__()

    return "{}({})".format(
        self.__class__.__name__,
        ", ".join(attr + '=' + repr(getattr(self, attr)) for attr in attrs))


def indent(lines, amount=2, chr=' '):
    """Indent a string.

    Prepends whitespace to every line in the passed string. (Lines are
    separated by newline characters.)

    Args:
        lines (str): The string to indent.

    Keyword Args:
        amount (int): The number of columns to indent by.
        chr (char): The character to to use as the indentation.

    Returns:
        str: The indented string.
    """
    lines = str(lines)
    padding = amount * chr
    return padding + ('\n' + padding).join(lines.split('\n'))


def fmt_constellation(c):
    """Format a constellation."""
    if not c:
        return "()\n"
    return "\n\n" + "\n".join(map(lambda x: indent(x), c)) + "\n"


def fmt_partition(partition):
    """Format a partition.

    The returned string looks like::

        0,1   []
        --- X ---
         2    0,1

    Args:
        partition (tuple(Part, Part)): The partition in question.

    Returns:
        str: A human-readable string representation of the partition.
    """
    if not partition:
        return ""

    part0, part1 = partition

    def node_repr(x):
        return ','.join(map(str, x)) if x else '[]'

    numer0, denom0 = node_repr(part0.mechanism), node_repr(part0.purview)
    numer1, denom1 = node_repr(part1.mechanism), node_repr(part1.purview)

    width0 = max(len(numer0), len(denom0))
    width1 = max(len(numer1), len(denom1))

    return ("{numer0:^{width0}}   {numer1:^{width1}}\n"
                        "{div0} X {div1}\n"
            "{denom0:^{width0}}   {denom1:^{width1}}").format(
                numer0=numer0, denom0=denom0, width0=width0, div0='-' * width0,
                numer1=numer1, denom1=denom1, width1=width1, div1='-' * width1)


def fmt_concept(concept):
    """Format a |Concept|."""
    return (
        "phi: {concept.phi}\n"
        "mechanism: {concept.mechanism}\n"
        "cause: {cause}\n"
        "effect: {effect}\n".format(
            concept=concept,
            cause=("\n" + indent(fmt_mip(concept.cause.mip, verbose=False))
                   if concept.cause else ""),
            effect=("\n" + indent(fmt_mip(concept.effect.mip, verbose=False))
                    if concept.effect else "")))


def fmt_mip(mip, verbose=True):
    """Format a |Mip|."""
    if mip is False or mip is None:  # mips can be Falsy
        return ""

    mechanism = "mechanism: {}\n".format(mip.mechanism) if verbose else ""
    direction = "direction: {}\n".format(mip.direction) if verbose else ""
    return (
        "phi: {mip.phi}\n"
        "{mechanism}"
        "purview: {mip.purview}\n"
        "partition:\n{partition}\n"
        "{direction}"
        "unpartitioned_repertoire:\n{unpart_rep}\n"
        "partitioned_repertoire:\n{part_rep}").format(
            mechanism=mechanism,
            direction=direction,
            mip=mip,
            partition=indent(fmt_partition(mip.partition)),
            unpart_rep=indent(mip.unpartitioned_repertoire),
            part_rep=indent(mip.partitioned_repertoire))


def fmt_big_mip(big_mip):
    """Format a |BigMip|."""
    return (
        "phi: {big_mip.phi}\n"
        "subsystem: {big_mip.subsystem}\n"
        "cut: {big_mip.cut}\n"
        "unpartitioned_constellation: {unpart_const}"
        "partitioned_constellation: {part_const}".format(
            big_mip=big_mip,
            unpart_const=fmt_constellation(big_mip.unpartitioned_constellation),
            part_const=fmt_constellation(big_mip.partitioned_constellation)))
