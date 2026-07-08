# timescale.py
"""Functions for manipulating the timescale of TPMs."""

import numpy as np
from scipy.sparse import csc_matrix

from . import convert


def sparse(matrix, threshold=0.1):
    return np.sum(matrix > 0) / matrix.size > threshold


def sparse_time(tpm, time_scale):
    sparse_tpm = csc_matrix(tpm)
    return (sparse_tpm**time_scale).toarray()


def dense_time(tpm, time_scale):
    return np.linalg.matrix_power(tpm, time_scale)


def run_tpm(tpm, time_scale):
    """Iterate a TPM by the specified number of time steps.

    Parameters
    ----------
    tpm : np.ndarray
        A state-by-node TPM.
    time_scale : int
        The number of steps to run the TPM.

    Returns
    -------
    np.ndarray
        The state-by-node TPM advanced by ``time_scale`` steps.

    Notes
    -----
    The TPM is converted to state-by-state form and raised to the
    ``time_scale`` power there, then converted back. A scipy sparse matrix
    power (:func:`sparse_time`) is used when the :func:`sparse` heuristic
    returns true — i.e. when more than 10% of the state-by-node TPM's entries
    are nonzero — and a dense power (:func:`dense_time`) otherwise.
    """
    sbs_tpm = convert.state_by_node2state_by_state(tpm)
    if sparse(tpm):
        tpm = sparse_time(sbs_tpm, time_scale)
    else:
        tpm = dense_time(sbs_tpm, time_scale)
    return convert.state_by_state2state_by_node(tpm)


def run_cm(cm, time_scale):
    """Iterate a connectivity matrix the specified number of steps.

    Raising the connectivity matrix to the ``time_scale`` power counts directed
    paths of that length between nodes; every nonzero entry is then clipped back
    to 1, so the result marks which node pairs are connected by a path of
    ``time_scale`` steps.

    Parameters
    ----------
    cm : np.ndarray
        A connectivity matrix.
    time_scale : int
        The number of steps to run.

    Returns
    -------
    np.ndarray
        The connectivity matrix at the new timescale.
    """
    cm = np.linalg.matrix_power(cm, time_scale)
    # Round non-unitary values back to 1
    cm[cm > 1] = 1
    return cm
