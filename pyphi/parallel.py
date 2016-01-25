#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# parallel.py

import multiprocessing

from . import config


# TODO: can return a negative number if NUMBER_OF_CORES
# is too negative. Handle this
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
        return cpu_count + config.NUMBER_OF_CORES + 1

    return config.NUMBER_OF_CORES
