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

    # Requesting more cores than available
    with config.override(NUMBER_OF_CORES=3):
        assert parallel.get_num_processes() == 2

    # Ok
    with config.override(NUMBER_OF_CORES=1):
        assert parallel.get_num_processes() == 1


class MapSquare(parallel.MapReduce):

    def empty_result(self):
        return set()

    @staticmethod
    def compute(num):
        return num ** 2

    def process_result(self, new, previous):
        previous.add(new)
        return previous


def test_map_square():
    engine = MapSquare([1, 2, 3])
    assert engine.run_parallel() == {1, 4, 9}
    assert engine.run_sequential() == {1, 4, 9}


def test_materialize_list_only_when_needed():
    with config.override(PROGRESS_BARS=False):
        engine = MapSquare(iter([1, 2, 3]))
        assert not isinstance(engine.iterable, list)

    with config.override(PROGRESS_BARS=True):
        engine = MapSquare(iter([1, 2, 3]))
        assert isinstance(engine.iterable, list)


class MapError(MapSquare):
    """Raise an exception in the worker process."""
    @staticmethod
    def compute(num):
        raise Exception("I don't wanna!")


def test_parallel_exception_handling():
    with pytest.raises(Exception, match=r"I don't wanna!"):
        MapError([1]).run(parallel=True)
