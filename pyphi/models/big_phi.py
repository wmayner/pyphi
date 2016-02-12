#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# models/big_phi.py


from .. import utils, config, jsonify
from . import cmp, fmt


_bigmip_attributes = ['phi', 'unpartitioned_constellation',
                      'partitioned_constellation', 'subsystem',
                      'cut_subsystem']


class BigMip(cmp._Orderable):
    """A minimum information partition for |big_phi| calculation.

    BigMips may be compared with the built-in Python comparison operators
    (``<``, ``>``, etc.). First, ``phi`` values are compared. Then, if these
    are equal up to |PRECISION|, the size of the subsystem is compared
    (exclusion principle).

    Attributes:
        phi (float): The |big_phi| value for the subsystem when taken against
            this MIP, *i.e.* the difference between the unpartitioned
            constellation and this MIP's partitioned constellation.
        unpartitioned_constellation (Constellation): The constellation of the
            whole subsystem.
        partitioned_constellation (Constellation): The constellation when the
            subsystem is cut.
        subsystem (Subsystem): The subsystem this MIP was calculated for.
        cut_subsystem (Subsystem): The subsystem with the minimal cut applied.
        time (float): The number of seconds it took to calculate.
        small_phi_time (float): The number of seconds it took to calculate the
            unpartitioned constellation.
    """

    def __init__(self, phi=None, unpartitioned_constellation=None,
                 partitioned_constellation=None, subsystem=None,
                 cut_subsystem=None):
        self.phi = phi
        self.unpartitioned_constellation = unpartitioned_constellation
        self.partitioned_constellation = partitioned_constellation
        self.subsystem = subsystem
        self.cut_subsystem = cut_subsystem
        self.time = None
        self.small_phi_time = None

    def __repr__(self):
        return fmt.make_repr(self, _bigmip_attributes)

    def __str__(self):
        return "\nBigMip\n======\n" + fmt.fmt_big_mip(self)

    @property
    def cut(self):
        """The unidirectional cut that makes the least difference to the
        subsystem.
        """
        return self.cut_subsystem.cut

    @property
    def network(self):
        """The network this |BigMip| belongs to."""
        return self.subsystem.network

    _unorderable_unless_eq = ['network']

    def _order_by(self):
        return [self.phi, len(self.subsystem)]

    def __eq__(self, other):
        return cmp._general_eq(self, other, _bigmip_attributes)

    def __bool__(self):
        """A BigMip is truthy if it is not reducible.

        (That is, if it has a significant amount of |big_phi|.)
        """
        return not utils.phi_eq(self.phi, 0)

    def __hash__(self):
        return hash((self.phi,
                     self.unpartitioned_constellation,
                     self.partitioned_constellation,
                     self.subsystem,
                     self.cut_subsystem))

    def to_json(self):
        return {
            attr: jsonify.jsonify(getattr(self, attr))
            for attr in _bigmip_attributes + ['time', 'small_phi_time']
        }


# TODO document
def _null_bigmip(subsystem):
    """Return a |BigMip| with zero |big_phi| and empty constellations.

    This is the MIP associated with a reducible subsystem.
    """
    return BigMip(subsystem=subsystem, cut_subsystem=subsystem, phi=0.0,
                  unpartitioned_constellation=(), partitioned_constellation=())


def _single_node_bigmip(subsystem):
    """Return a |BigMip| of a single-node with a selfloop.

    Whether these have a nonzero |Phi| value depends on the PyPhi constants.
    """
    if config.SINGLE_NODES_WITH_SELFLOOPS_HAVE_PHI:
        # TODO return the actual concept
        return BigMip(
            phi=0.5,
            unpartitioned_constellation=(),
            partitioned_constellation=(),
            subsystem=subsystem,
            cut_subsystem=subsystem)
    else:
        return _null_bigmip(subsystem)
