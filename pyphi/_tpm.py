#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _tpm.py

"""
Provides the TPM, ExplicitTPM, and ImplicitTPM classes.
"""

import numpy as np

from . import config
from .utils import np_hash, np_immutable
from .tpm import is_state_by_state


class TPM:
    """A transition probability matrix."""
    def __init__(self, tpm):
        self._tpm = self._build_tpm(tpm)

    def __hash__(self):
        return self._hash

    @property
    def tpm(self):
        """Return the underlying `tpm` object."""
        return self._tpm

    def validate(self):
        """Ensure the tpm is well-formed."""
        raise NotImplementedError

    @staticmethod
    def _build_tpm(tpm):
        """Validate the TPM passed by the user and convert to
        multidimensional form."""
        raise NotImplementedError



class ExplicitTPM(TPM):
    """An explicit network TPM in multidimensional form."""
    def __init__(self, tpm):
        super().__init__(tpm)
        self._hash = np_hash(self.tpm)

    @staticmethod
    def _build_tpm(tpm):
        """Validate the TPM passed by the user and convert to
        multidimensional form."""
        tpm = np.array(tpm)

        self.validate(check_independence=config.VALIDATE_CONDITIONAL_INDEPENDENCE)

        # Convert to multidimensional state-by-node form
        if is_state_by_state(tpm):
            tpm = convert.state_by_state2state_by_node(tpm)
        else:
            tpm = convert.to_multidimensional(tpm)

        np_immutable(tpm)

        return tpm

    @property
    def shape(self):
        return self.tpm.shape

    def validate(self, check_independence=True):
        """Validate this TPM.

        The TPM can be in

            * 2-dimensional state-by-state form,
            * 2-dimensional state-by-node form, or
            * multidimensional state-by-node form.
        """
        see_tpm_docs = (
            'See the documentation on TPM conventions and the `pyphi.Network` '
            'object for more information on TPM forms.'
        )
        # Get the number of nodes from the state-by-node TPM.
        tpm = self.tpm
        N = tpm.shape[-1]
        if tpm.ndim == 2:
            if not ((tpm.shape[0] == 2**N and tpm.shape[1] == N) or
                    (tpm.shape[0] == tpm.shape[1])):
                raise ValueError(
                    'Invalid shape for 2-D TPM: {}\nFor a state-by-node TPM, '
                    'there must be ' '2^N rows and N columns, where N is the '
                    'number of nodes. State-by-state TPM must be square. '
                    '{}'.format(tpm.shape, see_tpm_docs))
            if tpm.shape[0] == tpm.shape[1] and check_independence:
                conditionally_independent(tpm)
        elif tpm.ndim == (N + 1):
            if tpm.shape != tuple([2] * N + [N]):
                raise ValueError(
                    'Invalid shape for multidimensional state-by-node TPM: {}\n'
                    'The shape should be {} for {} nodes. {}'.format(
                        tpm.shape, ([2] * N) + [N], N, see_tpm_docs))
        else:
            raise ValueError(
                'Invalid TPM: Must be either 2-dimensional or multidimensional. '
                '{}'.format(see_tpm_docs))
        return True


class ImplicitTPM(TPM):
    """An implicit network TPM consisting of individual node TPMs."""
    def __init__(self, node_tpms):
        super().__init__(node_tpms)
        self._shape = (2,)*len(self.tpm) + (len(self.tpm),)
        self._hash = hash((np_hash(node_tpm) for node_tpm in self.tpm))

    @staticmethod
    def _build_tpm(tpm):
        """Validate the TPM passed by the user and convert to
        multidimensional form."""
        # TODO validate, set immutability flags, etc.
        return tpm
    
    @property
    def shape(self):
        return self._shape