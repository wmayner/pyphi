#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/parallel.py

"""
Utilities for parallel computation.
"""

import multiprocessing

from .. import config


def get_num_processes():
    """Return the number of processes to use in parallel."""
    cpu_count = multiprocessing.cpu_count()

    if config.NUMBER_OF_CORES == 0:
        raise ValueError(
            'Invalid NUMBER_OF_CORES; value may not be 0.')

    if config.NUMBER_OF_CORES > cpu_count:
        raise ValueError(
            'Invalid NUMBER_OF_CORES; value must be less than or '
            'equal to the available number of cores ({} for this '
            'system).'.format(cpu_count))

    if config.NUMBER_OF_CORES < 0:
        num = cpu_count + config.NUMBER_OF_CORES + 1
        if num <= 0:
            raise ValueError(
                'Invalid NUMBER_OF_CORES; negative value is too negative: '
                'requesting {} cores, {} available.'.format(num, cpu_count))

        return num

    return config.NUMBER_OF_CORES
