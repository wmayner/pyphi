"""Tests for pyphi.macro.search: bounded intrinsic-unit search (Eqs 15-19)."""

import numpy as np
import pytest

from pyphi import config
from pyphi import utils
from pyphi.conf import presets
from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import unit_integration
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import candidate_mappings
from pyphi.macro.search import competing_systems
from pyphi.macro.search import intrinsic_units
from pyphi.macro.search import is_intrinsic_unit
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from test.test_macro_criteria import bu_substrate
from test.test_macro_criteria import min_substrate
from test.test_macro_tpm import _asymmetric_substrate


def dancing_couples(w_v):
    """4 units; P(ON next) = 0.05 + 0.05*self + 0.6*horizontal + w_v*vertical.

    Wiring by unit index: 0 -> h=1, v=2; 1 -> h=0, v=3; 2 -> h=3, v=0;
    3 -> h=2, v=1. The authors' Fig 2 scenarios are w_v = 0.0 (sfn),
    0.01 (sfnn), 0.25 (sfs), all in state (0, 0, 0, 0).
    """
    horizontal = {0: 1, 1: 0, 2: 3, 3: 2}
    vertical = {0: 2, 1: 3, 2: 0, 3: 1}
    n = 4
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = tuple((row >> k) & 1 for k in range(n))
        for i in range(n):
            tpm[row, i] = (
                0.05 + 0.05 * s[i] + 0.6 * s[horizontal[i]] + w_v * s[vertical[i]]
            )
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


def tie_substrate():
    """3 units, exactly symmetric under swapping A and C.

    B couples to A and C identically; A and C couple to B only. Any
    system on footprint {A, B} has an isomorphic twin on {B, C}
    (overlapping at B), forcing exact phi ties.
    """
    n = 3
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = tuple((row >> k) & 1 for k in range(n))
        tpm[row, 0] = 0.05 + 0.05 * s[0] + 0.6 * s[1]
        tpm[row, 1] = 0.05 + 0.05 * s[1] + 0.3 * s[0] + 0.3 * s[2]
        tpm[row, 2] = 0.05 + 0.05 * s[2] + 0.6 * s[1]
    return Substrate(tpm, node_labels=("A", "B", "C"))


SF_STATE = (0, 0, 0, 0)
AC = MacroUnit((0, 2), 1, coarse_grain(2, on_counts={2}))
AB = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))


class TestSearchBounds:
    def test_defaults(self):
        bounds = SearchBounds()
        assert bounds.max_constituents == 4
        assert bounds.max_update_grain == 1
        assert bounds.max_depth == 1
        assert bounds.mappings == "FAMILIES"
        assert bounds.exhaustive_cap == 8
        assert bounds.apportionment == "NONE"
        assert bounds.max_background == 0

    def test_frozen(self):
        bounds = SearchBounds()
        with pytest.raises(AttributeError):
            bounds.max_depth = 2

    def test_max_micro_grain_composes(self):
        assert SearchBounds().max_micro_grain == 1
        assert SearchBounds(max_update_grain=2, max_depth=2).max_micro_grain == 4

    def test_max_constituents_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_constituents"):
            SearchBounds(max_constituents=0)

    def test_max_update_grain_below_one_rejected(self):
        with pytest.raises(ValueError, match="max_update_grain"):
            SearchBounds(max_update_grain=0)

    def test_negative_max_depth_rejected(self):
        with pytest.raises(ValueError, match="max_depth"):
            SearchBounds(max_depth=-1)

    def test_unknown_mappings_policy_rejected(self):
        with pytest.raises(ValueError, match="mappings"):
            SearchBounds(mappings="ALL")

    def test_unknown_apportionment_policy_rejected(self):
        with pytest.raises(ValueError, match="apportionment"):
            SearchBounds(apportionment="ALWAYS")

    def test_enumerate_requires_max_background(self):
        with pytest.raises(ValueError, match="max_background"):
            SearchBounds(apportionment="ENUMERATE")
        assert (
            SearchBounds(apportionment="ENUMERATE", max_background=1).max_background
            == 1
        )


class TestCandidateMappings:
    def test_families_two_constituents_grain_one(self):
        tables = candidate_mappings(2, 1, SearchBounds())
        # Coarse-grainings (canonicalized: complement when the all-OFF
        # state maps to ON), then black-boxings, first-seen order:
        # on_counts {0} -> complement of (1,0,0,0) = at-least-one-ON;
        # {1} -> exactly-one-ON; {2} -> both-ON; {0,1}, {0,2}, {1,2} ->
        # duplicates of the first three; blackbox {0} -> constituent-0;
        # {1} -> constituent-1; {0,1} -> duplicate of both-ON.
        assert tables == (
            (0, 1, 1, 1),
            (0, 1, 1, 0),
            (0, 0, 0, 1),
            (0, 1, 0, 1),
            (0, 0, 1, 1),
        )

    def test_families_count_three_constituents(self):
        assert len(candidate_mappings(3, 1, SearchBounds())) == 13

    def test_families_higher_grain_blackbox_only(self):
        # Coarse-graining is defined at update grain 1 only.
        tables = candidate_mappings(1, 2, SearchBounds(max_update_grain=2))
        assert tables == ((0, 0, 1, 1),)

    def test_exhaustive_min_shape(self):
        tables = candidate_mappings(2, 1, SearchBounds(mappings="EXHAUSTIVE"))
        # 2**(4-1) - 1 = 7 canonical surjective tables.
        assert len(tables) == 7
        assert len(set(tables)) == 7
        for table in tables:
            assert table[0] == 0  # canonical: all-OFF maps to OFF
            assert 1 in table  # surjective
        assert (0, 0, 0, 1) in tables

    def test_exhaustive_cap_exceeded(self):
        with pytest.raises(ValueError, match="exhaustive_cap"):
            candidate_mappings(
                2,
                2,
                SearchBounds(mappings="EXHAUSTIVE", max_update_grain=2),
            )

    def test_all_tables_canonical_and_unique(self):
        for policy in ("FAMILIES", "EXHAUSTIVE"):
            tables = candidate_mappings(2, 1, SearchBounds(mappings=policy))
            assert len(set(tables)) == len(tables)
            assert all(t[0] == 0 for t in tables)


class TestFig2Verdicts:
    """Battery 1: the three dancing-couples scenarios (authors'
    committed values, asserted at 1e-13)."""

    def test_sfn_not_integrated(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.0), AC, SF_STATE)
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_INTEGRATED
        assert verdict.phi == pytest.approx(0.0, abs=1e-13)

    def test_sfn_singleton_anchor(self):
        with config.override(**presets.iit4_2023):
            phi = unit_integration(dancing_couples(0.0), (0,), (SF_STATE,))
        assert phi == pytest.approx(0.02363345634846179, abs=1e-13)

    def test_sfnn_not_maximal(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.01), AC, SF_STATE)
        assert not verdict.valid
        assert verdict.reason is Reason.NOT_MAXIMAL
        assert verdict.phi == pytest.approx(0.004863714555961354, abs=1e-13)
        assert verdict.witness is not None
        assert len(verdict.witness.units) == 1
        assert verdict.witness_phi == pytest.approx(0.023640988356789627, abs=1e-13)
        assert verdict.num_competitors == 2

    def test_sfs_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.25), AC, SF_STATE)
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        assert verdict.phi == pytest.approx(0.16758555077361778, abs=1e-13)
        assert verdict.witness is None
        assert verdict.num_competitors == 2

    def test_sfs_horizontal_pair_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(dancing_couples(0.25), AB, SF_STATE)
        assert verdict.valid
        assert verdict.phi == pytest.approx(0.6728123807299448, abs=1e-13)


class TestMicroExemption:
    def test_micro_unit_trivially_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(min_substrate(), micro_unit(0), (0, 0))
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        # min singletons have phi_s = 0, yet micro units are valid ground.
        assert verdict.phi == 0.0
        assert verdict.num_competitors == 0

    def test_micro_unit_with_unreachable_state_still_valid(self):
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(bu_substrate(), micro_unit(2), (0, 0, 0))
        assert verdict.valid
        assert verdict.phi == 0.0


class TestGrainRaisedSingleton:
    def test_no_competitors_and_gated_by_integration(self):
        # Macroing over updates (Fig 3D): a singleton footprint admits
        # no proper-subset competitors, so the verdict reduces to Eq 15.
        unit = MacroUnit((0,), 2, blackbox(1, 2, (0,)))
        bounds = SearchBounds(max_update_grain=2)
        history = ((1, 0, 1, 0), (1, 0, 1, 0))
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(
                _asymmetric_substrate(), unit, history, bounds
            )
        assert verdict.num_competitors == 0
        assert verdict.valid == utils.is_positive(verdict.phi)
        assert verdict.reason in (Reason.VALID, Reason.NOT_INTEGRATED)


class TestCompetingSystems:
    def test_sfs_competitors_are_the_singletons(self):
        with config.override(**presets.iit4_2023):
            systems = competing_systems(dancing_couples(0.25), AC, SF_STATE)
        assert len(systems) == 2
        footprints = {
            tuple(u.micro_constituents for u in s.units) for s in systems
        }
        assert footprints == {((0,),), ((2,),)}

    def test_own_constituent_system_excluded(self):
        with config.override(**presets.iit4_2023):
            systems = competing_systems(dancing_couples(0.25), AC, SF_STATE)
        own = (micro_unit(0), micro_unit(2))
        assert all(s.units != own for s in systems)

    def test_micro_unit_has_no_competitors(self):
        with config.override(**presets.iit4_2023):
            assert competing_systems(min_substrate(), micro_unit(0), (0, 0)) == ()

    def test_all_member_footprints_proper_subsets(self):
        unit = MacroUnit((0, 1, 2), 1, coarse_grain(3, on_counts={3}))
        with config.override(**presets.iit4_2023):
            systems = competing_systems(bu_substrate(), unit, (0, 0, 0))
        footprint = set(unit.micro_constituents)
        for system in systems:
            for member in system.units:
                assert set(member.micro_constituents) < footprint


class TestVerdictMappingIndependence:
    """Battery 4: Eq 15 mapping-independence -- mapped and grained
    variants of one decomposition share the verdict."""

    def test_variants_share_verdict(self):
        variant_a = MacroUnit((0, 2), 1, coarse_grain(2, on_counts={1, 2}))
        variant_b = MacroUnit((0, 2), 1, blackbox(2, 1, (0,)))
        with config.override(**presets.iit4_2023):
            substrate = dancing_couples(0.25)
            verdicts = [
                is_intrinsic_unit(substrate, unit, SF_STATE)
                for unit in (AC, variant_a, variant_b)
            ]
        for verdict in verdicts[1:]:
            assert verdict.valid == verdicts[0].valid
            assert verdict.reason is verdicts[0].reason
            assert verdict.phi == verdicts[0].phi
            assert verdict.num_competitors == verdicts[0].num_competitors


class TestIntrinsicUnits:
    def test_min_pool_and_verdicts(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        # 2 micro units + 5 canonical FAMILIES variants of (0, 1).
        assert len(result.units) == 7
        grouped = result.units_by_footprint()
        assert set(grouped) == {(0,), (1,), (0, 1)}
        assert {u.mapping for u in grouped[(0, 1)]} == set(
            candidate_mappings(2, 1, SearchBounds())
        )
        assert all(u.constituents == (0, 1) for u in grouped[(0, 1)])
        # One verdict per decomposition (not per variant): 2 micro + 1.
        assert len(result.verdicts) == 3
        pair = [v for v in result.verdicts if v.constituents == (0, 1)]
        assert len(pair) == 1
        assert pair[0].verdict.valid
        assert pair[0].verdict.phi == pytest.approx(
            0.005106576483955726, abs=1e-13
        )
        assert pair[0].verdict.num_competitors == 2

    def test_micro_units_axiomatically_valid(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        micro = [v for v in result.verdicts if len(v.constituents) == 1]
        assert len(micro) == 2
        for verdict in micro:
            assert verdict.verdict.valid
            assert verdict.verdict.phi == 0.0  # valid despite zero phi

    def test_tie_substrate_excludes_unintegrated_footprint(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(tie_substrate(), (0, 0, 0), bounds)
        grouped = result.units_by_footprint()
        # (0, 2) is causally disconnected: NOT_INTEGRATED, no variants.
        assert (0, 2) not in grouped
        assert set(grouped) == {(0,), (1,), (2,), (0, 1), (1, 2)}
        assert len(result.units) == 3 + 5 + 5
        rejected = [v for v in result.verdicts if v.constituents == (0, 2)]
        assert len(rejected) == 1
        assert rejected[0].verdict.reason is Reason.NOT_INTEGRATED

    def test_bu_micro_only_pool(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(bu_substrate(), (0, 0, 0), SearchBounds())
        # Pairs are unintegrated; ABC is beaten by the singleton {A}
        # system at phi 1.0; pool stays micro.
        assert len(result.units) == 3
        full = [v for v in result.verdicts if v.constituents == (0, 1, 2)]
        assert len(full) == 1
        assert full[0].verdict.reason is Reason.NOT_MAXIMAL
        assert full[0].verdict.phi == pytest.approx(0.8300749985576875, abs=1e-13)
        assert full[0].verdict.witness_phi == 1.0
        # Unit C: unreachable state, phi 0, still valid ground.
        unit_c = [v for v in result.verdicts if v.constituents == (2,)]
        assert unit_c[0].verdict.valid
        assert unit_c[0].verdict.phi == 0.0

    def test_history_length_validated(self):
        with pytest.raises(ValueError, match="1 entries"):
            intrinsic_units(min_substrate(), ((0, 0), (0, 0)), SearchBounds())
        with pytest.raises(ValueError, match="bare state"):
            intrinsic_units(
                min_substrate(), (0, 0), SearchBounds(max_update_grain=2)
            )

    def test_result_is_frozen(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        with pytest.raises(AttributeError):
            result.units = ()
