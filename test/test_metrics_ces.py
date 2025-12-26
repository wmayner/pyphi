from unittest.mock import patch
import numpy as np
import pytest

from pyphi import compute, config
from pyphi.metrics.ces import ces_distance, emd_ground_distance
from pyphi import models


def test_emd_ground_distance_must_be_symmetric():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    with config.override(REPERTOIRE_DISTANCE="KLD"):
        with pytest.raises(ValueError):
            emd_ground_distance(a, b)


@pytest.mark.outdated
def test_ces_distances(s):
    with config.override(REPERTOIRE_DISTANCE="EMD"):
        sia = compute.subsystem.sia(s)

    with config.override(CES_DISTANCE="EMD"):
        assert ces_distance(sia.ces, sia.partitioned_ces) == 2.3125

    with config.override(CES_DISTANCE="SUM_SMALL_PHI"):
        assert ces_distance(sia.ces, sia.partitioned_ces) == 1.083333


@pytest.mark.outdated
def test_sia_uses_ces_distances(s):
    with config.override(REPERTOIRE_DISTANCE="EMD", CES_DISTANCE="EMD"):
        sia = compute.subsystem.sia(s)
        assert sia.phi == 2.3125

    with config.override(REPERTOIRE_DISTANCE="EMD", CES_DISTANCE="SUM_SMALL_PHI"):
        sia = compute.subsystem.sia(s)
        assert sia.phi == 1.083333


@patch("pyphi.metrics.ces._emd_simple")
@patch("pyphi.metrics.ces._emd")
@pytest.mark.outdated
def test_ces_distance_uses_simple_vs_emd(mock_emd_distance, mock_simple_distance, s):
    """Quick check that we use the correct EMD distance function for CESs.

    If the two CESs differ only in that some concepts have
    moved to the null concept and all other concepts are the same then
    we use the simple CES distance. Otherwise, use the EMD.
    """
    mock_emd_distance.return_value = float()
    mock_simple_distance.return_value = float()

    make_mice = lambda: models.MaximallyIrreducibleCauseOrEffect(
        models.RepertoireIrreducibilityAnalysis(
            phi=0,
            direction=None,
            mechanism=None,
            purview=None,
            partition=None,
            repertoire=None,
            partitioned_repertoire=None,
        )
    )

    lone_concept = models.Concept(
        cause=make_mice(),
        effect=make_mice(),
        mechanism=(0, 1),
    )
    # lone concept -> null concept
    ces_distance((lone_concept,), ())
    assert mock_emd_distance.called is False
    assert mock_simple_distance.called is True
    mock_simple_distance.reset_mock()

    other_concept = models.Concept(
        cause=make_mice(),
        effect=make_mice(),
        mechanism=(0, 1, 2),
    )
    # different concepts in CES
    ces_distance((lone_concept,), (other_concept,))
    assert mock_emd_distance.called is True
    assert mock_simple_distance.called is False
