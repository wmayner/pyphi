
import pytest
from unittest.mock import patch

from pyphi import compute, config, models


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


def test_big_mip_bipartitions():
    with config.override(CUT_ONE_APPROXIMATION=False):
        answer = [models.Cut((1,), (2, 3, 4)),
                  models.Cut((2,), (1, 3, 4)),
                  models.Cut((1, 2), (3, 4)),
                  models.Cut((3,), (1, 2, 4)),
                  models.Cut((1, 3), (2, 4)),
                  models.Cut((2, 3), (1, 4)),
                  models.Cut((1, 2, 3), (4,)),
                  models.Cut((4,), (1, 2, 3)),
                  models.Cut((1, 4), (2, 3)),
                  models.Cut((2, 4), (1, 3)),
                  models.Cut((1, 2, 4), (3,)),
                  models.Cut((3, 4), (1, 2)),
                  models.Cut((1, 3, 4), (2,)),
                  models.Cut((2, 3, 4), (1,))]
        assert compute.big_mip_bipartitions((1, 2, 3, 4)) == answer

    with config.override(CUT_ONE_APPROXIMATION=True):
        answer = [models.Cut((1,), (2, 3, 4)),
                  models.Cut((2,), (1, 3, 4)),
                  models.Cut((3,), (1, 2, 4)),
                  models.Cut((1, 2, 3), (4,)),
                  models.Cut((4,), (1, 2, 3)),
                  models.Cut((1, 2, 4), (3,)),
                  models.Cut((1, 3, 4), (2,)),
                  models.Cut((2, 3, 4), (1,))]
        assert compute.big_mip_bipartitions((1, 2, 3, 4)) == answer
