#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# partition.py

'''
Functions for generating partitions.
'''

from itertools import chain, product

from .cache import cache


# From stackoverflow.com/questions/19368375/set-partitions-in-python
def partitions(collection):
    '''Generate all set partitions of a collection.

    Example:
        >>> list(partitions(range(3)))  # doctest: +NORMALIZE_WHITESPACE
        [[[0, 1, 2]],
         [[0], [1, 2]],
         [[0, 1], [2]],
         [[1], [0, 2]],
         [[0], [1], [2]]]
    '''
    collection = list(collection)

    # Special cases
    if not collection:
        return []
    if len(collection) == 1:
        yield [collection]
        return

    first = collection[0]
    for smaller in partitions(collection[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n+1:]
        yield [[first]] + smaller


@cache(cache={}, maxmem=None)
def bipartition_indices(N):
    '''Return indices for undirected bipartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        list: A list of tuples containing the indices for each of the two
        parts.

    Example:
        >>> N = 3
        >>> bipartition_indices(N)
        [((), (0, 1, 2)), ((0,), (1, 2)), ((1,), (0, 2)), ((0, 1), (2,))]
    '''
    result = []
    if N <= 0:
        return result

    for i in range(2**(N - 1)):
        part = [[], []]
        for n in range(N):
            bit = (i >> n) & 1
            part[bit].append(n)
        result.append((tuple(part[1]), tuple(part[0])))
    return result


# TODO? rename to `bipartitions`
def bipartition(seq):
    '''Return a list of bipartitions for a sequence.

    Args:
        a (Iterable): The sequence to partition.

    Returns:
        list[tuple[tuple]]: A list of tuples containing each of the two
        partitions.

    Example:
        >>> bipartition((1,2,3))
        [((), (1, 2, 3)), ((1,), (2, 3)), ((2,), (1, 3)), ((1, 2), (3,))]
    '''
    return [(tuple(seq[i] for i in part0_idx), tuple(seq[j] for j in part1_idx))
            for part0_idx, part1_idx in bipartition_indices(len(seq))]


@cache(cache={}, maxmem=None)
def directed_bipartition_indices(N):
    '''Return indices for directed bipartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        list: A list of tuples containing the indices for each of the two
        parts.

    Example:
        >>> N = 3
        >>> directed_bipartition_indices(N)  # doctest: +NORMALIZE_WHITESPACE
        [((), (0, 1, 2)),
         ((0,), (1, 2)),
         ((1,), (0, 2)),
         ((0, 1), (2,)),
         ((2,), (0, 1)),
         ((0, 2), (1,)),
         ((1, 2), (0,)),
         ((0, 1, 2), ())]
    '''
    indices = bipartition_indices(N)
    return indices + [idx[::-1] for idx in indices[::-1]]


# TODO? [optimization] optimize this to use indices rather than nodes
def directed_bipartition(seq, nontrivial=False):
    '''Return a list of directed bipartitions for a sequence.

    Args:
        seq (Iterable): The sequence to partition.

    Returns:
        list[tuple[tuple]]: A list of tuples containing each of the two
        parts.

    Example:
        >>> directed_bipartition((1, 2, 3))  # doctest: +NORMALIZE_WHITESPACE
        [((), (1, 2, 3)),
         ((1,), (2, 3)),
         ((2,), (1, 3)),
         ((1, 2), (3,)),
         ((3,), (1, 2)),
         ((1, 3), (2,)),
         ((2, 3), (1,)),
         ((1, 2, 3), ())]
    '''
    bipartitions = [
        (tuple(seq[i] for i in part0_idx), tuple(seq[j] for j in part1_idx))
        for part0_idx, part1_idx in directed_bipartition_indices(len(seq))
    ]
    if nontrivial:
        # The first and last partitions have a part that is empty; skip them.
        # NOTE: This depends on the implementation of
        # `directed_partition_indices`.
        return bipartitions[1:-1]
    return bipartitions


def bipartition_of_one(seq):
    '''Generate bipartitions where one part is of length 1.'''
    seq = list(seq)
    for i, elt in enumerate(seq):
        yield ((elt,), tuple(seq[:i] + seq[(i + 1):]))


def reverse_elements(seq):
    '''Reverse the elements of a sequence.'''
    for elt in seq:
        yield elt[::-1]


def directed_bipartition_of_one(seq):
    '''Generate directed bipartitions where one part is of length 1.

    Args:
        seq (Iterable): The sequence to partition.

    Returns:
        list[tuple[tuple]]: A list of tuples containing each of the two
        partitions.

    Example:
        >>> partitions = directed_bipartition_of_one((1, 2, 3))
        >>> list(partitions)  # doctest: +NORMALIZE_WHITESPACE
        [((1,), (2, 3)),
         ((2,), (1, 3)),
         ((3,), (1, 2)),
         ((2, 3), (1,)),
         ((1, 3), (2,)),
         ((1, 2), (3,))]
    '''
    bipartitions = list(bipartition_of_one(seq))
    return chain(bipartitions, reverse_elements(bipartitions))


@cache(cache={}, maxmem=None)
def directed_tripartition_indices(N):
    '''Return indices for directed tripartitions of a sequence.

    Args:
        N (int): The length of the sequence.

    Returns:
        list[tuple]: A list of tuples containing the indices for each
        partition.

    Example:
        >>> N = 1
        >>> directed_tripartition_indices(N)
        [((0,), (), ()), ((), (0,), ()), ((), (), (0,))]
    '''

    result = []
    if N <= 0:
        return result

    base = [0, 1, 2]
    for key in product(base, repeat=N):
        part = [[], [], []]
        for i, location in enumerate(key):
            part[location].append(i)

        result.append(tuple(tuple(p) for p in part))

    return result


def directed_tripartition(seq):
    '''Generator over all directed tripartitions of a sequence.

    Args:
        seq (Iterable): a sequence.

    Yields:
        tuple[tuple]: A tripartition of ``seq``.

    Example:
        >>> seq = (2, 5)
        >>> list(directed_tripartition(seq))  # doctest: +NORMALIZE_WHITESPACE
        [((2, 5), (), ()),
         ((2,), (5,), ()),
         ((2,), (), (5,)),
         ((5,), (2,), ()),
         ((), (2, 5), ()),
         ((), (2,), (5,)),
         ((5,), (), (2,)),
         ((), (5,), (2,)),
         ((), (), (2, 5))]
    '''
    for a, b, c in directed_tripartition_indices(len(seq)):
        yield (tuple(seq[i] for i in a),
               tuple(seq[j] for j in b),
               tuple(seq[k] for k in c))


# Knuth's algorithm for k-partitions of a set
# codereview.stackexchange.com/questions/1526/finding-all-k-subset-partitions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# pylint: disable=too-many-arguments,too-many-branches


def _visit(n, a, k, collection):
    ps = [[] for i in range(k)]
    for j in range(n):
        ps[a[j + 1]].append(collection[j])
    return ps


def _f(mu, nu, sigma, n, a, k, collection):  # flake8: noqa
    if mu == 2:
        yield _visit(n, a, k, collection)
    else:
        for v in _f(mu - 1, nu - 1, (mu + sigma) % 2, n, a, k, collection):
            yield v
    if nu == mu + 1:
        a[mu] = mu - 1
        yield _visit(n, a, k, collection)
        while a[nu] > 0:
            a[nu] = a[nu] - 1
            yield _visit(n, a, k, collection)
    elif nu > mu + 1:
        if (mu + sigma) % 2 == 1:
            a[nu - 1] = mu - 1
        else:
            a[mu] = mu - 1
        if (a[nu] + sigma) % 2 == 1:
            for v in _b(mu, nu - 1, 0, n, a, k, collection):
                yield v
        else:
            for v in _f(mu, nu - 1, 0, n, a, k, collection):
                yield v
        while a[nu] > 0:
            a[nu] = a[nu] - 1
            if (a[nu] + sigma) % 2 == 1:
                for v in _b(mu, nu - 1, 0, n, a, k, collection):
                    yield v
            else:
                for v in _f(mu, nu - 1, 0, n, a, k, collection):
                    yield v


def _b(mu, nu, sigma, n, a, k, collection):  # flake8: noqa
    if nu == mu + 1:
        while a[nu] < mu - 1:
            yield _visit(n, a, k, collection)
            a[nu] = a[nu] + 1
        yield _visit(n, a, k, collection)
        a[mu] = 0
    elif nu > mu + 1:
        if (a[nu] + sigma) % 2 == 1:
            for v in _f(mu, nu - 1, 0, n, a, k, collection):
                yield v
        else:
            for v in _b(mu, nu - 1, 0, n, a, k, collection):
                yield v
        while a[nu] < mu - 1:
            a[nu] = a[nu] + 1
            if (a[nu] + sigma) % 2 == 1:
                for v in _f(mu, nu - 1, 0, n, a, k, collection):
                    yield v
            else:
                for v in _b(mu, nu - 1, 0, n, a, k, collection):
                    yield v
        if (mu + sigma) % 2 == 1:
            a[nu - 1] = 0
        else:
            a[mu] = 0
    if mu == 2:
        yield _visit(n, a, k, collection)
    else:
        for v in _b(mu - 1, nu - 1, (mu + sigma) % 2, n, a, k, collection):
            yield v


def k_partitions(collection, k):
    '''Generate all ``k``-partitions of a collection.

    Example:
        >>> list(k_partitions(range(3), 2))
        [[[0, 1], [2]], [[0], [1, 2]], [[0, 2], [1]]]
    '''
    collection = list(collection)
    n = len(collection)

    # Special cases
    if n == 0 or k < 1:
        return []
    if k == 1:
        return [[collection]]

    a = [0] * (n + 1)
    for j in range(1, k + 1):
        a[n - k + j] = j - 1
    return _f(k, n, 0, n, a, k, collection)
