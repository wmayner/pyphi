"""Round-trips for macro-level results: MacroSystem and ComplexesResult."""

import numpy as np
import pytest

from pyphi import serialize
from pyphi.macro.search import ComplexesResult
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import complexes
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import coarse_grain
from pyphi.substrate import Substrate
from test.conftest import IIT_4_CONFIG

FORMATS = ["json", "msgpack"]

MIN_TPM = np.array([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_macro_system():
    sub = Substrate(MIN_TPM, node_labels=("A", "B"))
    unit = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))
    return MacroSystem.from_micro(sub, (unit,), ((0, 0),))


@pytest.mark.parametrize("fmt", FORMATS)
def test_macro_system_round_trips(fmt):
    obj = make_macro_system()
    restored = round_trip(obj, fmt)
    assert type(restored) is MacroSystem
    # MacroSystem equality covers micro substrate, units, history, partition.
    assert restored == obj
    assert hash(restored) == hash(obj)
    assert restored.state == obj.state
    assert restored.substrate == obj.substrate
    # The construction's cause TPM is stored, not recomputed.
    fresh = obj.cause_marginal
    loaded = restored.cause_marginal
    assert loaded.state_space == fresh.state_space
    for i in range(fresh.n_nodes):
        assert np.array_equal(loaded.factor(i), fresh.factor(i))


@pytest.mark.parametrize("fmt", FORMATS)
def test_macro_system_save_load_preserves_type(tmp_path, fmt):
    obj = make_macro_system()
    path = tmp_path / f"macro_system.{fmt}"
    obj.save(path)
    restored = MacroSystem.load(path)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_complexes_result_round_trips(fmt):
    sub = Substrate(MIN_TPM, node_labels=("A", "B"))
    with IIT_4_CONFIG:
        result = complexes(sub, (0, 0), SearchBounds())
        restored = round_trip(result, fmt)
        assert type(restored) is ComplexesResult
        assert len(restored.complexes) == len(result.complexes)
        for fresh, loaded in zip(result.complexes, restored.complexes, strict=True):
            assert float(loaded.phi) == float(fresh.phi)
            assert loaded.node_indices == fresh.node_indices
            assert loaded.units == fresh.units
        assert len(restored.records) == len(result.records)
        for fresh, loaded in zip(result.records, restored.records, strict=True):
            assert loaded.system == fresh.system
            assert loaded.phi == fresh.phi
            assert loaded.ii_ceiling == fresh.ii_ceiling
            assert loaded.gated == fresh.gated
        assert restored.ties == result.ties
        assert float(restored.maximal_complex.phi) == float(result.maximal_complex.phi)
