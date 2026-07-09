"""Round-trip serialization of units-bearing Complex objects."""

import numpy as np

from pyphi import config
from pyphi import serialize
from pyphi.conf import presets
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import complexes
from pyphi.macro.units import MacroUnit
from pyphi.models.complex import ExcludedCandidate
from pyphi.substrate import Substrate


def test_excluded_candidate_units_roundtrip():
    meso = MacroUnit((0, 1), 1, (0, 0, 0, 1))
    unit = MacroUnit((meso, 2), 1, (0, 1, 1, 1))
    e = ExcludedCandidate((0, 1, 2), 0.25, units=(unit,))
    restored = serialize.loads(serialize.dumps(e))
    assert restored.node_indices == (0, 1, 2)
    assert restored.units == (unit,)


def test_excluded_candidate_without_units_roundtrip():
    e = ExcludedCandidate((1, 2), 0.5)
    restored = serialize.loads(serialize.dumps(e))
    assert restored.units is None


def test_macro_complex_roundtrip():
    tpm = np.array([[0.05, 0.05], [0.05, 0.06], [0.06, 0.05], [0.95, 0.95]])
    substrate = Substrate(tpm, node_labels=("A", "B"))
    with config.override(**presets.iit4_2023):
        result = complexes(substrate, (0, 0), SearchBounds())
    top = result.complexes[0]
    restored = serialize.loads(serialize.dumps(top))
    assert restored.node_indices == top.node_indices
    assert restored.units == top.units
    assert restored.is_maximal == top.is_maximal
    assert restored.excluded == top.excluded
