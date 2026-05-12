from unittest.mock import patch

import numpy as np
import pytest

from pyphi import config
from pyphi import models
from pyphi.conf import presets
from pyphi.formalism import iit3
from pyphi.metrics.ces import ces_distance
from pyphi.metrics.ces import emd_ground_distance

from .conftest import skip_if_no_pyemd


@pytest.mark.emd
@skip_if_no_pyemd
def test_emd_ground_distance_must_be_symmetric():
    a = np.ones((2, 2, 2)) / 8
    b = np.ones((2, 2, 2)) / 8
    with config.override(mechanism_phi_measure="KLD"), pytest.raises(ValueError):
        emd_ground_distance(a, b)


@pytest.mark.emd
@skip_if_no_pyemd
def test_ces_distances(s):
    """Canonical IIT 3.0 CES distance for the basic substrate (1,0,0)."""
    with config.override(**presets.iit3):
        sia = iit3.sia(s)

    with config.override(**presets.iit3, ces_measure="EMD"):
        assert ces_distance(sia.ces, sia.partitioned_ces, system=s) == pytest.approx(
            2.3125, rel=1e-6
        )

    with config.override(**presets.iit3, ces_measure="SUM_SMALL_PHI"):
        assert ces_distance(sia.ces, sia.partitioned_ces) == pytest.approx(
            1.083333, rel=1e-6
        )


@pytest.mark.emd
@skip_if_no_pyemd
def test_sia_uses_ces_distances(s):
    """Canonical sia.phi under the IIT 3.0 preset (EMD / EMD)."""
    with config.override(**presets.iit3):
        sia = iit3.sia(s)
        assert sia.phi == pytest.approx(2.3125, rel=1e-6)

    with config.override(**presets.iit3, ces_measure="SUM_SMALL_PHI"):
        sia = iit3.sia(s)
        assert sia.phi == pytest.approx(1.083333, rel=1e-6)


@pytest.mark.emd
@skip_if_no_pyemd
@patch("pyphi.metrics.ces._emd_simple")
@patch("pyphi.metrics.ces._emd")
def test_ces_distance_uses_simple_vs_emd(mock_emd_distance, mock_simple_distance, s):
    """Quick check that we use the correct EMD distance function for CESs.

    If the two CESs differ only in that some concepts have
    moved to the null concept and all other concepts are the same then
    we use the simple CES distance. Otherwise, use the EMD.
    """
    mock_emd_distance.return_value = 0.0
    mock_simple_distance.return_value = 0.0

    def make_mice():
        return models.MaximallyIrreducibleCauseOrEffect(
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
    with config.override(ces_measure="EMD"):
        # lone concept -> null concept
        ces_distance((lone_concept,), (), system=s)
        assert mock_emd_distance.called is False
        assert mock_simple_distance.called is True
        mock_simple_distance.reset_mock()

        other_concept = models.Concept(
            cause=make_mice(),
            effect=make_mice(),
            mechanism=(0, 1, 2),
        )
        # different concepts in CES
        ces_distance((lone_concept,), (other_concept,), system=s)
        assert mock_emd_distance.called is True
        assert mock_simple_distance.called is False
