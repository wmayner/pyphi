#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_parallel.py

from unittest.mock import patch

import pytest

from pyphi import config
from pyphi.compute import parallel


def _mock_cpu_count():
    return 2


@patch('multiprocessing.cpu_count', _mock_cpu_count)
def test_num_processes():

    # Can't have no processes
    with config.override(NUMBER_OF_CORES=0):
        with pytest.raises(ValueError):
            parallel.get_num_processes()

    # Negative numbers
    with config.override(NUMBER_OF_CORES=-1):
        assert parallel.get_num_processes() == 2

    # Too negative
    with config.override(NUMBER_OF_CORES=-3):
        with pytest.raises(ValueError):
            parallel.get_num_processes()

    # Requesting too many cores
    with config.override(NUMBER_OF_CORES=3):
        with pytest.raises(ValueError):
            parallel.get_num_processes()

    # Ok
    with config.override(NUMBER_OF_CORES=1):
        assert parallel.get_num_processes() == 1
