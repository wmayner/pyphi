"""The grain search reports only irreducible complexes, and its result prints
as a compact card."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.macro import SearchBounds
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import micro_unit


def test_grain_search_returns_no_complex_when_every_candidate_is_reducible():
    with pyphi.config.override(**pyphi.iit4_2026):
        for substrate, state in [
            (examples.basic_substrate(), (1, 0, 0)),
            (examples.xor_substrate(), (0, 0, 0)),
        ]:
            assert substrate.complexes(state) == ()
            result = pyphi.analyze(substrate, state, grains=True)
            assert tuple(result.complexes) == ()
            assert result.ties == ()
            assert not result.maximal_complex


def test_an_empty_complexes_result_keeps_its_columns():
    with pyphi.config.override(**pyphi.iit4_2026):
        empty = pyphi.analyze(examples.basic_substrate(), (1, 0, 0), grains=True)
        full = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    df = empty.to_pandas()
    assert len(df) == 0
    assert list(df.columns) == list(full.to_pandas().columns)


def test_grain_search_complexes_all_have_positive_phi():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    assert result.complexes
    assert all(c.phi > 0 for c in result.complexes)


def test_complexes_result_prints_compactly():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    assert len(repr(result)) < 3000
    assert hasattr(result, "_repr_html_")


def test_complexes_result_exports_one_row_per_complex():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (0, 1, 1), grains=True
        )
    df = result.to_pandas()
    assert list(df["node_indices"]) == [c.node_indices for c in result.complexes]
    assert list(df["phi"]) == [c.phi for c in result.complexes]
    assert list(df["grains"]) == [(1,), (1,)]


# A sparse substrate whose pair {0, 1} in state (0, 1) has no predecessor
# with the background unit 2 held at its current value, but is reached from
# the universe state (1, 0, 1). The macro construction must check
# reachability on its cause TPM (Eq. 34), not the effect TPM (Eq. 33).
_CAUSE_REACHABLE_TPM = np.array(
    [
        [0.9, 0, 1],
        [1, 0, 1],
        [1, 0.2, 0.2],
        [0, 0, 0.2],
        [0, 0.5, 0.9],
        [0, 1, 0.2],
        [0, 0, 0],
        [1, 1, 1],
    ]
)

# A deterministic substrate with the same mismatch under the 2023 formalism.
_DETERMINISTIC_TPM = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
        [0, 1, 0],
    ]
)


def _summary(complexes):
    return [(tuple(c.node_indices), float(c.phi)) for c in complexes]


@pytest.mark.parametrize(
    ("preset", "tpm", "state", "expected"),
    [
        (pyphi.iit4_2026, _CAUSE_REACHABLE_TPM, (0, 1, 0), [((0, 1), 0.0148001)]),
        (pyphi.iit4_2023, _DETERMINISTIC_TPM, (1, 0, 1), [((0, 2), 2.0)]),
    ],
)
def test_depth_zero_grain_search_matches_substrate_complexes(
    preset, tpm, state, expected
):
    substrate = pyphi.Substrate(tpm)
    with pyphi.config.override(**preset):
        micro = _summary(substrate.complexes(state))
        searched = _summary(
            pyphi.analyze(substrate, state, grains=SearchBounds(max_depth=0)).complexes
        )
    assert [n for n, _ in searched] == [n for n, _ in micro]
    assert [p for _, p in searched] == pytest.approx([p for _, p in micro])
    assert [n for n, _ in searched] == [n for n, _ in expected]
    assert [p for _, p in searched] == pytest.approx([p for _, p in expected], abs=1e-6)


def test_macro_system_accepts_a_state_reachable_through_the_background():
    substrate = pyphi.Substrate(_CAUSE_REACHABLE_TPM)
    state = (0, 1, 0)
    with pyphi.config.override(**pyphi.iit4_2026):
        system = MacroSystem.from_micro(
            substrate, (micro_unit(0), micro_unit(1)), (state,)
        )
        micro_phi = pyphi.System(substrate, state, subset=(0, 1)).sia().phi
        assert system.sia().phi == pytest.approx(micro_phi)
