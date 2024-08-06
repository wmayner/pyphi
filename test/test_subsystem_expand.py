import numpy as np
import pytest

from pyphi import Direction, compute, config

EPSILON = 10 ** (-config.PRECISION)

CD = (2, 3)
BCD = (1, 2, 3)
ABCD = (0, 1, 2, 3)


def test_expand_cause_repertoire(use_iit_3_config, micro_s_all_off):
    sia = compute.subsystem.sia(micro_s_all_off)
    A = sia.ces[0]
    cause = A.cause_repertoire

    assert np.all(abs(A.expand_cause_repertoire(CD) - cause) < EPSILON)
    assert np.all(
        abs(
            A.expand_cause_repertoire(BCD).flatten(order="F")
            - np.array([1 / 6 if i < 6 else 0 for i in range(8)])
        )
        < EPSILON
    )
    assert np.all(
        abs(
            A.expand_cause_repertoire(ABCD).flatten(order="F")
            - np.array([1 / 12 if i < 12 else 0 for i in range(16)])
        )
        < EPSILON
    )
    assert np.all(
        abs(A.expand_cause_repertoire(ABCD) - A.expand_cause_repertoire()) < EPSILON
    )


def test_expand_effect_repertoire(micro_s_all_off):
    sia = compute.subsystem.sia(micro_s_all_off)
    A = sia.ces[0]
    effect = A.effect_repertoire

    assert np.all(abs(A.expand_effect_repertoire(CD) - effect) < EPSILON)
    assert np.all(
        abs(
            A.expand_effect_repertoire(BCD).flatten(order="F")
            - np.array(
                [0.25725, 0.23275, 0.11025, 0.09975, 0.11025, 0.09975, 0.04725, 0.04275]
            )
        )
        < EPSILON
    )
    assert np.all(
        abs(
            A.expand_effect_repertoire(ABCD).flatten(order="F")
            - np.array(
                [
                    0.13505625,
                    0.12219375,
                    0.12219375,
                    0.11055625,
                    0.05788125,
                    0.05236875,
                    0.05236875,
                    0.04738125,
                    0.05788125,
                    0.05236875,
                    0.05236875,
                    0.04738125,
                    0.02480625,
                    0.02244375,
                    0.02244375,
                    0.02030625,
                ]
            )
        )
        < EPSILON
    )
    assert np.all(
        abs(A.expand_effect_repertoire(ABCD) - A.expand_effect_repertoire()) < EPSILON
    )


def test_expand_repertoire_purview_must_be_subset_of_new_purview(s):
    mechanism = (0, 1)
    purview = (0, 1)
    new_purview = (1,)
    cause_repertoire = s.cause_repertoire(mechanism, purview)
    with pytest.raises(ValueError):
        s.expand_repertoire(Direction.CAUSE, cause_repertoire, new_purview)
