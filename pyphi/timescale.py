#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# timescale.py

"""
Functions for converting the timescale of a TPM.
"""

import numpy as np
from scipy.sparse import csc_matrix

from . import convert


def sparse(matrix, threshold=0.1):
    return np.sum(matrix > 0) / matrix.size > threshold


def sparse_time(tpm, time_scale):
    sparse_tpm = csc_matrix(tpm)
    return (sparse_tpm ** time_scale).toarray()


def dense_time(tpm, time_scale):
    return np.linalg.matrix_power(tpm, time_scale)


def run_tpm(tpm, time_scale):
    """Iterate a TPM by the specified number of time steps.

    Args:
        tpm (np.ndarray): A state-by-node tpm.
        time_scale (int): The number of steps to run the tpm.

    Returns:
        np.ndarray
    """
    sbs_tpm = convert.state_by_node2state_by_state(tpm)
    if sparse(tpm):
        tpm = sparse_time(sbs_tpm, time_scale)
    else:
        tpm = dense_time(sbs_tpm, time_scale)
    return convert.state_by_state2state_by_node(tpm)


def run_cm(cm, time_scale):
    """Iterate a connectivity matrix the specified number of steps.

    Args:
        cm (np.ndarray): A connectivity matrix.
        time_scale (int): The number of steps to run.

    Returns:
        np.ndarray: The connectivity matrix at the new timescale.
    """
    cm = np.linalg.matrix_power(cm, time_scale)
    # Round non-unitary values back to 1
    cm[cm > 1] = 1
    return cm
