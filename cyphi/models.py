#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Models
~~~~~~

Lightweight containers for MICE, MIP, cut, partition, and concept data.
"""

from collections import namedtuple, Iterable
from .utils import phi_eq as _phi_eq

# TODO use properties to avoid data duplication
# TODO add proper docstrings with __doc__

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


# Comparison helpers {{{
# ======================
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
        if not _phi_eq(a.phi, b.phi):
            return False
    return all(_numpy_aware_eq(getattr(a, attr), getattr(b, attr)) if attr !=
               'phi' else True for attr in attributes)

# Phi-ordering methods
_phi_lt = lambda self, other: (self.phi < other.phi) if other else False
_phi_gt = lambda self, other: (self.phi > other.phi) if other else True
_phi_le = lambda self, other: ((_phi_lt(self, other) or _phi_eq(self.phi,
                                                                other.phi))
                               if other else False)
_phi_ge = lambda self, other: ((_phi_gt(self, other) or _phi_eq(self.phi,
                                                                other.phi))
                               if other else False)
# }}}


# Minimum Information Partition (small phi MIP) {{{
# =================================================
# TODO! include references to mechanism and purview
_mip_attributes = ['phi', 'direction', 'partition', 'unpartitioned_repertoire',
                   'partitioned_repertoire']
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
_mice_attributes = ['phi', 'direction', 'mechanism', 'purview', 'repertoire',
                    'mip']
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
_concept_attributes = ['phi', 'mechanism', 'location', 'cause', 'effect']
Concept = namedtuple('Concept', _concept_attributes)
Concept.__doc__ = """\
A star in concept-space.

The location is given by the probabilities of each state in its cause and
effect repertoires, i.e.
    concept.location = array[direction][n_0][n_1]...[n_k]
where `direction` is either `PAST` or `FUTURE` and the rest of the dimensions
correspond to a node in the network. `phi` is the small-phi_max value.
`cause` and `effect` are the MICE objects for the past and future,
respectively.
"""
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
                      'partitioned_constellation', 'subsystem']
BigMip = namedtuple('BigMip', _bigmip_attributes)
BigMip.__eq__ = lambda self, other: _general_eq(self, other,
                                                _bigmip_attributes)
# TODO! document comparison methods
# TODO! implement exclusion principle in comparison methods
# Order by phi value, then by subsystem size
def _bigmip_lt(self, other):
    if other:
        return (self.subsystem < other.subsystem if _phi_eq(self.phi, other.phi)
                else _phi_le(self, other))
    else:
        return False

def _bigmip_gt(self, other):
    if other:
        return (self.subsystem > other.subsystem if _phi_eq(self.phi, other.phi)
                else _phi_gt(self, other))
    else:
        return True

def _bigmip_le(self, other):
    return self < other or _phi_eq(self, other) if other else False

def _bigmip_ge(self, other):
    return self > other or _phi_eq(self, other) if other else True

BigMip.__lt__ = _bigmip_lt
BigMip.__gt__ = _bigmip_gt
BigMip.__le__ = _bigmip_le
BigMip.__ge__ = _bigmip_ge
# }}}


# vim: set foldmarker={{{,}}} foldlevel=0  foldmethod=marker :
