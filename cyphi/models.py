#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models
~~~~~~

Lightweight containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable
from .utils import phi_eq


# Cut {{{
# =======
# Connections from the 'severed' set to the 'intact' set are severed, while
# those from 'intact' to 'severed' are left intact
Cut = namedtuple('Cut', ['severed', 'intact'])
# }}}


# Part {{{
# ========
# Represents one part of a partition of a system for MIP evaluation
Part = namedtuple('Part', ['mechanism', 'purview'])
# }}}


def _numpy_aware_eq(a, b):
    """Return whether two objects are equal via recursion, using all(x == y)
    for comparing numpy arays."""
    if (not (isinstance(a, str) or isinstance(b, str)) and
            isinstance(a, Iterable) and isinstance(b, Iterable)):
        return all(_numpy_aware_eq(x, y) for x, y in zip(a, b))
    return a == b


# TODO cross reference PRECISION in docs
def _general_eq(a, b, attributes):
    """Return whether two objects are equal up to the given attributes.

    If an attribute is called ``'phi'``, it is compared up to
    ``constants.PRECISION``. All other given attributes are compared with
    :func:`_numpy_aware_eq`"""
    if 'phi' in attributes:
        if phi_eq(a.phi, b.phi):
            return False
        attributes.remove('phi')
    return all(_numpy_aware_eq(getattr(a, attr), getattr(b, attr)) for attr in
               attributes)

# Phi-ordering methods
_phi_lt = lambda self, other: self.phi < other.phi
_phi_gt = lambda self, other: self.phi > other.phi
_phi_le = lambda self, other: self < other or phi_eq(self.phi, other.phi)
_phi_ge = lambda self, other: self > other or phi_eq(self.phi, other.phi)


# Minimum Information Partition {{{
# =================================
_mip_attributes = ['direction', 'partition', 'unpartitioned_repertoire',
                   'partitioned_repertoire', 'phi']
Mip = namedtuple('Mip', _mip_attributes)
Mip.__eq__ = lambda self, other: _general_eq(self, other, _mip_attributes)
# Order by phi value
Mip.__lt__ = _phi_lt
Mip.__gt__ = _phi_gt
Mip.__le__ = _phi_le
Mip.__ge__ = _phi_ge
# }}}


# Maximally Irreducible Cause or Effect {{{
# =========================================
_mice_attributes = ['direction', 'mechanism', 'purview', 'repertoire', 'mip',
                    'phi']
Mice = namedtuple('Mice', _mice_attributes)
Mice.__eq__ = lambda self, other: _general_eq(self, other, _mice_attributes)
# Order by phi value
Mice.__lt__ = _phi_lt
Mice.__gt__ = _phi_gt
Mice.__le__ = _phi_le
Mice.__ge__ = _phi_ge
# }}}


# Concept {{{
# ===========
# A star in concept-space.
#
# The location is given by the probabilities of each state in its cause and
# effect repertoires, i.e.
#     concept.location = array[direction][n_0][n_1]...[n_k]
# where `direction` is either `PAST` or `FUTURE` and the rest of the dimensions
# correspond to a node in the network. `phi` is the small-phi_max value.
# `cause` and `effect` are the MICE objects for the past and future,
# respectively.
_concept_attributes = ['mechanism', 'location', 'phi', 'cause', 'effect']
Concept = namedtuple('Concept', _concept_attributes)
Concept.__eq__ = lambda self, other: _general_eq(self, other,
                                                 _concept_attributes)
# Order by phi value
Concept.__lt__ = _phi_lt
Concept.__gt__ = _phi_gt
Concept.__le__ = _phi_le
Concept.__ge__ = _phi_ge
# }}}


# Big Phi MIP {{{
# ===============
_bigmip_attributes = ['phi', 'partition', 'unpartitioned_constellation',
                      'partitioned_constellation']
BigMip = namedtuple('BigMip', _bigmip_attributes)
BigMip.__eq__ = lambda self, other: _general_eq(self, other,
                                                _bigmip_attributes)
# Order by phi value
BigMip.__lt__ = _phi_lt
BigMip.__gt__ = _phi_gt
BigMip.__le__ = _phi_le
BigMip.__ge__ = _phi_ge
# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
