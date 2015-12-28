
import pytest
from unittest.mock import patch

from pyphi import compute, config


@config.override(NUMBER_OF_CORES=0)
def test_num_processes_number_of_cores_cant_be_0():
    with pytest.raises(ValueError):
        compute.get_num_processes()


def _mock_cpu_count():
    return 2


@config.override(NUMBER_OF_CORES=-1)
@patch('multiprocessing.cpu_count', _mock_cpu_count)
def test_num_processes_with_negative_number_of_cores():
    assert compute.get_num_processes() == 2


@config.override(NUMBER_OF_CORES=3)
@patch('multiprocessing.cpu_count', _mock_cpu_count)
def test_num_processes_with_too_many_cores():
    with pytest.raises(ValueError):
        compute.get_num_processes()
