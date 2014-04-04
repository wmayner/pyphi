#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models
~~~~~~

Lightweight containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable


# Equality method that uses np.array_equal when needed
def tuple_eq(a, b):
    if (not (isinstance(a, str) or isinstance(b, str)) and
            isinstance(a, Iterable) and isinstance(b, Iterable)):
        return all(tuple_eq(x, y) for x, y in zip(a,b))
    return a == b


# Connections from the 'severed' set to the 'intact' set are severed, while
# those from 'intact' to 'severed' are left intact
Cut = namedtuple('Cut', ['severed', 'intact'])


Part = namedtuple('Part', ['mechanism', 'purview'])


# TODO refactor to use __eq__ comparison rather than directly using tuple_eq
Mip = namedtuple('Mip', ['direction', 'partition', 'unpartitioned_repertoire',
                         'partitioned_repertoire', 'difference'])


Mice = namedtuple('Mice', ['direction', 'mechanism', 'purview', 'repertoire',
                           'mip', 'phi'])


# A star in concept-space.
Concept = namedtuple('Concept', ['mechanism', 'location', 'size', 'cause',
                                 'effect'])
# The location is given by the probabilities of each state in its cause and
# effect repertoires, i.e.
#     concept.location = array[direction][n_0][n_1]...[n_k]
# where `direction` is either `PAST` or `FUTURE` and the rest of the dimensions
# correspond to a node in the network. The size is the phi_max value. `cause`
# and `effect` are the MICE objects for the past and future, respectively.


# Use tuple_eq to compare namedtuples with numpy arrays in them
Concept.__eq__, Mip.__eq__, Mice.__eq__ = [tuple_eq] * 3


BigMip = namedtuple('BigMip', ['phi', 'partition',
                               'unpartitioned_constellation',
                               'partitioned_constellation'])

BigMip.__lt__ = lambda self, other: self.phi < other.phi
BigMip.__gt__ = lambda self, other: self.phi > other.phi
