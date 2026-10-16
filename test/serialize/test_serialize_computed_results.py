"""Exact save/load round trips of computed analyses and complexes, and reuse
of a loaded result as the input of a new computation."""

import io

import numpy as np
import pytest

import pyphi
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.system import System
from test.conftest import skip_if_no_emd_backend
from test.hypothesis_utils import random_substrate

FORMATS = ["json", "msgpack"]

BINARY = ({"seed": 52, "n": 3}, (0, 0, 1), None)
KARY = ({"seed": 18, "n": 3, "alphabet": (3, 2, 2)}, (1, 1, 1), None)
SUBSET = ({"seed": 81, "n": 4, "extreme": 0.3}, (1, 0, 1, 0), (0, 1, 2))

ANALYSIS_CASES = [
    pytest.param("IIT_4_0_2026", *BINARY, id="2026-binary"),
    pytest.param("IIT_4_0_2026", *KARY, id="2026-kary"),
    pytest.param("IIT_4_0_2026", *SUBSET, id="2026-subset"),
    pytest.param("IIT_3_0", *BINARY, marks=skip_if_no_emd_backend, id="iit3-binary"),
    pytest.param("IIT_3_0", *KARY, marks=skip_if_no_emd_backend, id="iit3-kary"),
]


def _round_trip(obj, fmt):
    buffer = io.BytesIO()
    pyphi.save(obj, buffer, format=fmt)
    buffer.seek(0)
    return pyphi.load(buffer, format=fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(("name", "kwargs", "state", "subset"), ANALYSIS_CASES)
def test_analysis_round_trips_exactly_and_is_reusable(name, kwargs, state, subset, fmt):
    """An analysis with concretely computed relations loads equal to the one
    saved, with bit-equal repertoires; the loaded substrate recomputes the
    same analysis, and the loaded SIA gives the same structure as the
    original when passed to System.ces."""
    substrate = random_substrate(**kwargs)
    with config.override(**presets.by_name[name], relation_computation="CONCRETE"):
        original = pyphi.analyze(
            substrate, state, subset=subset, parallel=False, progress=False
        )
        loaded = _round_trip(original, fmt)
        loaded_substrate = _round_trip(substrate, fmt)
        recomputed = pyphi.analyze(
            loaded_substrate, state, subset=subset, parallel=False, progress=False
        )
        system = System(substrate, state, subset=subset)
        from_loaded = system.ces(sia=loaded.sia)
        from_original = system.ces(sia=original.sia)
    assert original.phi > 0 and original.ces.distinctions
    assert loaded == original
    assert loaded.sia == original.sia and loaded.ces == original.ces
    for x, y in zip(original.ces.distinctions, loaded.ces.distinctions, strict=True):
        for direction in ("cause", "effect"):
            assert (
                np.asarray(getattr(x, direction).repertoire).tobytes()
                == np.asarray(getattr(y, direction).repertoire).tobytes()
            )
    assert loaded_substrate == substrate
    for i in range(substrate.size):
        np.testing.assert_array_equal(
            np.asarray(loaded_substrate.factored_tpm.factor(i)),
            np.asarray(substrate.factored_tpm.factor(i)),
        )
    assert recomputed == original
    assert from_loaded == from_original


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(
    ("name", "kwargs", "state"),
    [
        pytest.param("IIT_4_0_2026", *SUBSET[:2], id="2026"),
        pytest.param("IIT_3_0", *BINARY[:2], marks=skip_if_no_emd_backend, id="iit3"),
    ],
)
def test_complexes_round_trip_exactly_and_their_sias_are_reusable(
    name, kwargs, state, fmt
):
    """Complexes load equal to the ones saved, margins and maximality
    included, and each loaded complex's SIA gives the structure System.ces()
    computes for its units."""
    substrate = random_substrate(**kwargs)
    with config.override(**presets.by_name[name]):
        found = substrate.complexes(state, parallel=False, progress=False)
        loaded = _round_trip(found, fmt)
        structures = [
            (
                System(substrate, state, subset=c.node_indices).ces(sia=d.sia),
                System(substrate, state, subset=c.node_indices).ces(),
            )
            for c, d in zip(found, loaded, strict=True)
        ]
    assert found and found[0].phi > 0
    assert loaded == found
    assert [c.exclusion_margin for c in loaded] == [c.exclusion_margin for c in found]
    assert [c.is_maximal for c in loaded] == [c.is_maximal for c in found]
    for from_loaded, computed in structures:
        assert from_loaded == computed
