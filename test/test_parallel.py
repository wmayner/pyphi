#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_parallel.py

import pytest
from unittest.mock import patch

from pyphi import config
from pyphi.compute import parallel


@config.override(NUMBER_OF_CORES=0)
def test_num_processes_number_of_cores_cant_be_0():
    with pytest.raises(ValueError):
        parallel.get_num_processes()


def _mock_cpu_count():
    return 2


@config.override(NUMBER_OF_CORES=-1)
@patch('multiprocessing.cpu_count', _mock_cpu_count)
def test_num_processes_with_negative_number_of_cores():
    assert parallel.get_num_processes() == 2


@config.override(NUMBER_OF_CORES=3)
@patch('multiprocessing.cpu_count', _mock_cpu_count)
def test_num_processes_with_too_many_cores():
    with pytest.raises(ValueError):
        parallel.get_num_processes()
