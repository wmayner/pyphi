import numpy as np
import pytest

from pyphi.metrics import distribution


TEST_DATA = (
    (np.ones((2, 2, 2)) / 8, np.ones((2, 2, 2)) / 8),
    (np.array([[[1.0]], [[0.0]]]), np.array([[[0.25]], [[0.75]]])),
    (np.array([[[0.25]], [[0.75]]]), np.array([[[1.0]], [[0.0]]])),
    (np.array([[[0.25]], [[0.75]]]), np.array([[[0.75]], [[0.25]]])),
    (np.array([[[0.25]]]), np.array([[[0.75]]])),
    (np.array([[9, 12], [4, 5]]) / 30, np.ones(4).reshape((2, 2)) / 4),
    (np.ones(4).reshape((2, 2)) / 4, np.array([[9, 12], [4, 5]]) / 30),
    (np.array([0, 1.0]), np.array([0.5, 0.5])),
    (np.array([0.5, 0.5]), np.array([0, 1.0])),
    (np.array([0.0, 0.0]), np.array([0.0, 0.0])),
    (
        np.array(
            [
                0.60183152,
                0.09236264,
                0.94829783,
                0.3360443,
                0.36940628,
                0.18902953,
                0.00986186,
                0.35108897,
                0.29617025,
                0.19815334,
            ]
        ),
        np.array(
            [
                0.02444988,
                0.11994048,
                0.34864668,
                0.00275309,
                0.01829823,
                0.41558419,
                0.7682337,
                0.7468439,
                0.58376815,
                0.28925836,
            ]
        ),
    ),
    (
        np.array(
            [
                0.67061961,
                0.94509532,
                0.76696495,
                0.80548176,
                0.16206401,
                0.40453888,
                0.78306659,
                0.28769049,
                0.56400422,
                0.34757965,
            ]
        ),
        np.array(
            [
                0.31790194,
                0.04752617,
                0.55990994,
                0.96821618,
                0.63266059,
                0.36250555,
                0.94535072,
                0.76372157,
                0.39726735,
                0.90495261,
            ]
        ),
    ),
)


def test_suppress_np_warnings():
    @distribution.np_suppress()
    def divide_by_zero():
        np.ones((2,)) / np.zeros((2,))

    @distribution.np_suppress()
    def multiply_by_nan():
        np.array([1, 0]) * np.log(0)

    # Try and trigger an error:
    with np.errstate(divide="raise", invalid="raise"):
        divide_by_zero()
        multiply_by_nan()


def test_hamming_matrix():
    # fmt: off
    answer = np.array([
        [0, 1, 1, 2, 1, 2, 2, 3],
        [1, 0, 2, 1, 2, 1, 3, 2],
        [1, 2, 0, 1, 2, 3, 1, 2],
        [2, 1, 1, 0, 3, 2, 2, 1],
        [1, 2, 2, 3, 0, 1, 1, 2],
        [2, 1, 3, 2, 1, 0, 2, 1],
        [2, 3, 1, 2, 1, 2, 0, 1],
        [3, 2, 2, 1, 2, 1, 1, 0],
    ]).astype(float)
    # fmt: on
    assert np.array_equal(distribution._hamming_matrix(3), answer)


def test_large_hamming_matrix():
    n = distribution._NUM_PRECOMPUTED_HAMMING_MATRICES + 1
    distribution._hamming_matrix(n)


def test_hamming_emd_validates_distribution_shapes():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((3, 3, 3)) / 27
    with pytest.raises(ValueError):
        distribution.hamming_emd(a, b)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            0.75,
            0.75,
            0.5,
            0.0,
            0.266667,
            0.266667,
            0.5,
            0.5,
            0.0,
        ],
        # NOTE: The EMD as currently implemented requires the input to have a
        # size that is a power of 2, so we omit the last two tests in TEST_DATA
        # by not providing an expected value (using behavior of `zip`).
    ),
)
def test_hamming_emd(pq, answer):
    assert np.allclose(distribution.hamming_emd(*pq), answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            1.5,
            1.5,
            1.0,
            0.5,
            0.4,
            0.4,
            1.0,
            1.0,
            0.0,
            3.64839424,
            3.4951311999999994,
        ],
    ),
)
def test_l1(pq, answer):
    assert np.allclose(distribution.l1(*pq), answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            0.8112781244591328,
            0.8112781244591328,
            0.0,
            0.18872187554086717,
            0.1317265875938649,
            0.1317265875938649,
            1.0,
            1.0,
            0.0,
            0.6621482247726478,
            0.5216594185459629,
        ],
    ),
)
def test_entropy_difference(pq, answer):
    result = distribution.entropy_difference(*pq)
    if np.isnan(answer):
        assert np.isnan(result)
    else:
        assert np.allclose(result, answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            2.0,
            float("inf"),
            0.7924812503605783,
            -0.3962406251802891,
            0.13172658759386474,
            0.1376866963458108,
            1.0,
            float("inf"),
            0.0,
            6.989064076830283,
            3.866290893689359,
        ],
    ),
)
def test_kld(pq, answer):
    assert np.allclose(distribution.kld(*pq), answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            0.7334585933443496,
            0.7334585933443496,
            0.0,
            0.10845859334434962,
            0.054237587956657674,
            0.054237587956657674,
            1.0,
            1.0,
            0.0,
            0.1833287731088058,
            1.7318178465283829,
        ],
    ),
)
def test_psq2(pq, answer):
    assert np.allclose(distribution.psq2(*pq), answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            4.0,
            float("inf"),
            1.717042709114586,
            -0.13208020839342968,
            0.19958628702442655,
            0.24185916736968496,
            1.0,
            float("inf"),
            np.nan,
            38.83048445250753,
            8.277494548476637,
        ],
    ),
)
def test_mp2q(pq, answer):
    result = distribution.mp2q(*pq)
    if np.isnan(answer):
        assert np.isnan(result)
    else:
        assert np.allclose(result, answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            np.zeros((2, 2, 2)),
            np.array([[[2.0]], [[0.0]]]),
            np.array([[[-0.5]], [[float("inf")]]]),
            np.array([[[-0.39624063]], [[1.18872188]]]),
            np.array([[[-0.39624063]]]),
            np.array([[0.07891032, 0.27122876], [-0.12091875, -0.09749375]]),
            np.array([[-0.0657586, -0.16951798], [0.22672265, 0.14624063]]),
            np.array([0.0, 1.0]),
            np.array([float("inf"), -0.5]),
            np.array([0.0, 0.0]),
            np.array(
                [
                    2.78134052,
                    -0.03481493,
                    1.36893856,
                    2.32927623,
                    1.60153595,
                    -0.21483757,
                    -0.06196741,
                    -0.38232538,
                    -0.28994143,
                    -0.10814048,
                ]
            ),
            np.array(
                [
                    0.72219888,
                    4.07682535,
                    0.3481759,
                    -0.21383711,
                    -0.31843438,
                    0.06402851,
                    -0.21277068,
                    -0.40522047,
                    0.28515812,
                    -0.47983323,
                ]
            ),
        ],
    ),
)
def test_information_density(pq, answer):
    assert np.allclose(distribution.information_density(*pq), answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            2.0,
            float("inf"),
            1.18872188,
            -0.39624063,
            0.27122876204505514,
            0.22672264890212962,
            1.0,
            float("inf"),
            0.0,
            2.7813405239249747,
            4.07682535456649,
        ],
    ),
)
def test_intrinsic_difference(pq, answer):
    assert np.allclose(distribution.intrinsic_difference(*pq), answer)


@pytest.mark.parametrize(
    "pq,answer",
    zip(
        TEST_DATA,
        [
            0.0,
            2.0,
            float("inf"),
            1.18872188,
            0.39624063,
            0.27122876204505514,
            0.22672264890212962,
            1.0,
            float("inf"),
            0.0,
            2.7813405239249747,
            4.07682535456649,
        ],
    ),
)
def test_absolute_intrinsic_difference(pq, answer):
    assert np.allclose(distribution.absolute_intrinsic_difference(*pq), answer)
