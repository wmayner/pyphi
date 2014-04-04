#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models
~~~~~~

Lightweight containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable
import numpy as np
from . import constants


# Cut {{{
#========
# Connections from the 'severed' set to the 'intact' set are severed, while
# those from 'intact' to 'severed' are left intact
Cut = namedtuple('Cut', ['severed', 'intact'])
# }}}


# Part {{{
# ========
# Represents one part of a partition of a system for MIP evaluation
Part = namedtuple('Part', ['mechanism', 'purview'])
# }}}


def numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using all(x == y)
    for comparing numpy arays."""
    if (not (isinstance(a, str) or isinstance(b, str)) and
            isinstance(a, Iterable) and isinstance(b, Iterable)):
        return all(numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


# TODO cross reference PRECISION in docs
def general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to
    ``constants.PRECISION``. All other given attributes are compared with
    :func:`numpy_aware_eq`"""
    if 'phi' in attributes:
        if abs(a.phi - b.phi) > constants.EPSILON:
            return False
        attributes.remove('phi')
    return all(numpy_aware_eq(getattr(a, attr), getattr(b, attr)) for attr in
               attributes)


# Minimum Information Partition {{{
# =================================
mip_attributes = ['direction', 'partition', 'unpartitioned_repertoire',
                  'partitioned_repertoire', 'phi']
Mip = namedtuple('Mip', mip_attributes)
Mip.__eq__ = lambda self, other: general_eq(self, other, mip_attributes)
# }}}


# Maximally Irreducible Cause or Effect {{{
# =========================================
mice_attributes = ['direction', 'mechanism', 'purview', 'repertoire', 'mip',
                   'phi']
Mice = namedtuple('Mice', mice_attributes)
Mice.__eq__ = lambda self, other: general_eq(self, other, mice_attributes)
# }}}


# Concept {{{
# ===========
# A star in concept-space.
#
# The location is given by the probabilities of each state in its cause and
# effect repertoires, i.e.
#     concept.location = array[direction][n_0][n_1]...[n_k]
# where `direction` is either `PAST` or `FUTURE` and the rest of the dimensions
# correspond to a node in the network. The size is the phi_max value. `cause`
# and `effect` are the MICE objects for the past and future, respectively.
Concept = namedtuple('Concept', ['mechanism', 'location', 'size', 'cause',
                                 'effect'])
Concept.__eq__ = numpy_aware_eq
# }}}


# Big Phi MIP {{{
# ===============
bigmip_attributes = ['phi', 'partition', 'unpartitioned_constellation',
                     'partitioned_constellation']
BigMip = namedtuple('BigMip', bigmip_attributes)
BigMip.__eq__ = lambda self, other: general_eq(self, other, bigmip_attributes)
# Order BigMips by their phi values
BigMip.__lt__ = lambda self, other: self.phi < other.phi
BigMip.__gt__ = lambda self, other: self.phi > other.phi
# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
