"""Tests for pyphi.macro.search: bounded intrinsic-unit search (Eqs 15-19)."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi import config
from pyphi import numerics
from pyphi.conf import presets
from pyphi.macro.criteria import Reason
from pyphi.macro.criteria import judge_candidate
from pyphi.macro.criteria import unit_integration
from pyphi.macro.search import ComplexesResult
from pyphi.macro.search import SearchBounds
from pyphi.macro.search import candidate_mappings
from pyphi.macro.search import competing_systems
from pyphi.macro.search import complexes
from pyphi.macro.search import intrinsic_units
from pyphi.macro.search import is_intrinsic_unit
from pyphi.macro.search import valid_systems
from pyphi.macro.system import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from test.macro.test_macro_criteria import bu_substrate
from test.macro.test_macro_criteria import min_substrate
from test.macro.test_macro_tpm import CG_TPM
from test.macro.test_macro_tpm import _asymmetric_substrate


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
            SearchBounds(apportionment="ENUMERATE", max_background=1).max_background == 1
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
            verdict = is_intrinsic_unit(_asymmetric_substrate(), unit, history, bounds)
        assert verdict.num_competitors == 0
        assert verdict.valid == numerics.is_positive(verdict.phi)
        assert verdict.reason in (Reason.VALID, Reason.NOT_INTEGRATED)


class TestCompetingSystems:
    def test_sfs_competitors_are_the_singletons(self):
        with config.override(**presets.iit4_2023):
            systems = competing_systems(dancing_couples(0.25), AC, SF_STATE)
        assert len(systems) == 2
        footprints = {tuple(u.micro_constituents for u in s.units) for s in systems}
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


MIN_BOTH_ON = MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2}))


def _spans_footprint(system, footprint):
    """True if ``system`` is a single unit covering all of ``footprint``."""
    return len(system.units) == 1 and set(system.units[0].micro_constituents) == set(
        footprint
    )


class TestWrappingExcludedFromF:
    """Eq 16's competition excludes v^J, the candidate itself.

    Marshall (clarification): the subset condition in f(U^J, W^J) is on
    the total constituents and is not strict, and the comparison is over
    v' != v^J. A single unit spanning all of U^J is the candidate's own
    grain, so it never competes; admitting it would force the candidate
    to beat its own inflated macro phi_s, which the ``min`` example
    refutes. See ``experiment_marshall_f.py``.
    """

    def test_wrapping_never_in_competitors_depth_one(self):
        with config.override(**presets.iit4_2023):
            systems = competing_systems(min_substrate(), MIN_BOTH_ON, (0, 0))
        assert all(not _spans_footprint(s, (0, 1)) for s in systems)

    def test_min_unit_valid_despite_higher_wrapping_phi(self):
        # Constituent-system phi_s is tiny (0.005); the one-unit wrapping
        # is large (0.788). The unit is still VALID because the wrapping
        # is excluded from f.
        with config.override(**presets.iit4_2023):
            verdict = is_intrinsic_unit(min_substrate(), MIN_BOTH_ON, (0, 0))
            wrapping_phi = float(
                MacroSystem.from_micro(min_substrate(), (MIN_BOTH_ON,), ((0, 0),))
                .sia()
                .phi
            )
        assert verdict.valid
        assert verdict.reason is Reason.VALID
        assert verdict.phi == pytest.approx(0.005106576483955726, abs=1e-13)
        assert wrapping_phi == pytest.approx(0.7883339770634886, abs=1e-12)
        assert wrapping_phi > verdict.phi

    def test_admitting_wrapping_would_flip_min_verdict(self):
        # The control: were the wrapping admitted (the naive-literal
        # reading), min's macro unit would fail Eq 16. This is why the
        # exclusion is load-bearing, not a free choice.
        with config.override(**presets.iit4_2023):
            shipped = competing_systems(min_substrate(), MIN_BOTH_ON, (0, 0))
            phi_vJ = float(
                unit_integration(min_substrate(), MIN_BOTH_ON.constituents, (0, 0))
            )
            wrapping = MacroSystem.from_micro(min_substrate(), (MIN_BOTH_ON,), ((0, 0),))
            competitors = [(s, float(s.sia().phi)) for s in shipped]
            with_wrapping = judge_candidate(
                phi_vJ, [*competitors, (wrapping, float(wrapping.sia().phi))]
            )
            without_wrapping = judge_candidate(phi_vJ, competitors)
        assert without_wrapping.reason is Reason.VALID
        assert with_wrapping.reason is Reason.NOT_MAXIMAL

    def test_same_union_meso_reorganizations_compete_depth_two(self):
        # At depth 2, f for a candidate over the whole footprint includes
        # same-U^J reorganizations into several smaller units, and still
        # excludes any single unit spanning U^J.
        unit = MacroUnit((0, 1, 2), 1, coarse_grain(3, on_counts={3}))
        bounds = SearchBounds(max_depth=2, max_constituents=3)
        with config.override(**presets.iit4_2023):
            systems = competing_systems(dancing_couples(0.25), unit, SF_STATE, bounds)
        footprint = (0, 1, 2)
        same_union_multi = [
            s
            for s in systems
            if len(s.units) > 1
            and {i for u in s.units for i in u.micro_constituents} == set(footprint)
        ]
        assert same_union_multi
        assert all(not _spans_footprint(s, footprint) for s in systems)


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
        assert pair[0].verdict.phi == pytest.approx(0.005106576483955726, abs=1e-13)
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
            intrinsic_units(min_substrate(), (0, 0), SearchBounds(max_update_grain=2))

    def test_result_is_frozen(self):
        with config.override(**presets.iit4_2023):
            result = intrinsic_units(min_substrate(), (0, 0), SearchBounds())
        with pytest.raises(AttributeError):
            result.units = ()


def assert_eq18(system):
    """Eq 18: stakes (footprint union apportionment) pairwise disjoint."""
    claimed = set()
    for unit in system.units:
        stake = set(unit.micro_constituents) | set(unit.background_apportionment)
        assert not (claimed & stake)
        claimed |= stake


class TestValidSystems:
    def test_min_count_and_eq18(self):
        with config.override(**presets.iit4_2023):
            systems = valid_systems(min_substrate(), (0, 0), SearchBounds())
        # {A}, {B}, {A,B} plus the 5 one-unit mapped variants.
        assert len(systems) == 8
        for system in systems:
            assert_eq18(system)

    def test_bu_drops_unreachable_singleton(self):
        with config.override(**presets.iit4_2023):
            systems = valid_systems(bu_substrate(), (0, 0, 0), SearchBounds())
        # 7 micro combinations minus the unconstructable {C}.
        assert len(systems) == 6
        assert all(
            tuple(u.micro_constituents for u in s.units) != ((2,),) for s in systems
        )

    def test_tie_substrate_count(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            systems = valid_systems(tie_substrate(), (0, 0, 0), bounds)
        # 7 micro combos + 5 [alpha_AB] + 5 [alpha_AB, C] + 5 [alpha_BC]
        # + 5 [alpha_BC, A].
        assert len(systems) == 27
        for system in systems:
            assert_eq18(system)


class TestMinDriver:
    """Battery 2: min end-to-end with EXHAUSTIVE mappings (7 canonical
    tables after complement dedup)."""

    def test_macro_complex_found(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        assert len(result.complexes) == 1
        winner = result.complexes[0]
        # The argmax mapping is the authors' both-on coarse-graining,
        # in canonical form. Golden recorded at implementation time;
        # sanity: equals the committed both-on macro phi
        # (0.7883339770634886) at 1e-13.
        assert winner.units == (MacroUnit((0, 1), 1, (0, 0, 0, 1)),)
        assert float(winner.phi) == pytest.approx(0.7883339770634884, abs=1e-13)
        assert result.ties == ()

    def test_records_contain_micro_pair_anchor(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        by_units = {r.system.units: r.phi for r in result.records}
        assert by_units[(micro_unit(0), micro_unit(1))] == pytest.approx(
            0.005106576483955726, abs=1e-13
        )

    def test_records_match_independent_recomputation(self):
        # Battery 4: memoized phi equals a fresh evaluation.
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
            for record in result.records[:3]:
                fresh = MacroSystem.from_micro(
                    record.system.micro_substrate,
                    record.system.units,
                    record.system.micro_history,
                )
                assert fresh.sia().phi == pytest.approx(record.phi, abs=1e-13)

    def test_every_record_satisfies_eq18(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        for record in result.records:
            assert_eq18(record.system)

    def test_excluded_records_and_margin_cover_same_footprint_rivals(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        winner = result.complexes[0]
        # The seven losing candidates on the winner's own footprint
        # {0,1} (six rival mappings and the micro pair) plus the two
        # singletons. Golden recorded at implementation time.
        assert len(winner.excluded) == 9
        assert any(
            e.node_indices == (0, 1)
            and e.units == (micro_unit(0), micro_unit(1))
            and e.phi == pytest.approx(0.005106576483955726, abs=1e-13)
            for e in winner.excluded
        )
        # The winning grain beat its best overlapping rival (the
        # (0,1,0,1)/(0,0,1,1) mappings, phi 0.2532971079071088) by:
        assert winner.exclusion_margin == pytest.approx(0.5350368691563798, abs=1e-13)
        assert winner.effectively_tied is False


class TestBuDriver:
    """Battery 3: micro-exemption under the consistent convention (see
    bu_substrate's docstring). The full micro system is admissible and
    reproduces the committed phi, but the singleton systems {A} and {B}
    (phi 1.0) beat it, so they are the complexes -- golden recorded at
    implementation time."""

    def test_micro_system_admissible_and_anchored(self):
        with config.override(**presets.iit4_2023):
            result = complexes(bu_substrate(), (0, 0, 0), SearchBounds())
        by_units = {r.system.units: r.phi for r in result.records}
        full = tuple(micro_unit(i) for i in range(3))
        assert by_units[full] == pytest.approx(0.8300749985576875, abs=1e-13)

    def test_complexes_are_the_strong_singletons(self):
        with config.override(**presets.iit4_2023):
            result = complexes(bu_substrate(), (0, 0, 0), SearchBounds())
        footprints = {
            tuple(u.micro_constituents for u in s.units) for s in result.complexes
        }
        assert footprints == {((0,),), ((1,),)}
        assert all(
            float(s.phi) == pytest.approx(1.0, abs=1e-13) for s in result.complexes
        )
        assert result.ties == ()

    def test_empty_complexes_is_a_result_not_an_error(self):
        # max_depth=0 restricts P(u) to micro systems; the micro pair
        # (phi 0.0051) beats the overlapping singletons (phi 0), so it
        # is the only complex at depth 0.
        bounds = SearchBounds(max_depth=0)
        with config.override(**presets.iit4_2023):
            result = complexes(min_substrate(), (0, 0), bounds)
        assert isinstance(result, ComplexesResult)
        assert len(result.complexes) == 1


class TestTiePath:
    """Battery 5: the exactly-symmetric fixture. Every system on
    footprint {A,B} has a permutation-identical twin on {B,C}, so every
    asymmetric-footprint clique ties at phi (and at big Phi under
    Composition escalation, at precision) and fails exclusion. The
    recursive walk continues past each failed clique -- their units stay
    available -- and condenses onto the substrate's own symmetric
    candidate: the three-singleton system, the only competitive
    candidate that is its own mirror image."""

    def test_tied_twins_fail_and_symmetric_complex_is_accepted(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            result = complexes(tie_substrate(), (0, 0, 0), bounds)
        # The top clique is the (0,1,1,1)-mapped mirror pair.
        top = result.ties[0]
        assert {tuple(u.micro_constituents for u in s.units) for s in top} == {
            ((0, 1),),
            ((1, 2),),
        }
        assert all(s.units[0].mapping == (0, 1, 1, 1) for s in top)
        phis = {r.system: r.phi for r in result.records}
        a, b = top
        assert numerics.eq(phis[a], phis[b])
        assert phis[a] == pytest.approx(0.3881829280978132, abs=1e-13)
        # The accepted complex is the symmetric three-singleton system.
        assert len(result.complexes) == 1
        winner = result.complexes[0]
        assert winner.node_indices == (0, 1, 2)
        assert tuple(u.micro_constituents for u in winner.units) == (
            (0,),
            (1,),
            (2,),
        )
        assert float(winner.phi) == pytest.approx(0.08449862433339383, abs=1e-13)


@pytest.mark.slow
class TestCostGuard:
    """Battery 6: the full default-bounds driver on the cg substrate
    terminates and its record reproduces the SP1-anchored micro panel."""

    def test_default_driver_on_cg(self):
        with config.override(**presets.iit4_2023):
            substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
            result = complexes(substrate, (0, 0, 0, 0))
        by_units = {r.system.units: r.phi for r in result.records}
        panel = {
            (micro_unit(0),): 0.003976279885291341,
            (micro_unit(0), micro_unit(1)): 0.044088890564147803,
            tuple(micro_unit(i) for i in range(4)): 0.02015654077792439,
        }
        for units, expected in panel.items():
            assert by_units[units] == pytest.approx(expected, abs=1e-13)
        for record in result.records:
            assert_eq18(record.system)
        # Driver-outcome golden, recorded at implementation time: the
        # search recovers the paper's Example 1 macro system -- both-on
        # coarse-grainings over (A, B) and (C, D) -- as the unique
        # complex, at SP1's exact-construction phi golden.
        assert len(result.complexes) == 1
        winner = result.complexes[0]
        assert winner.units == (
            MacroUnit((0, 1), 1, (0, 0, 0, 1)),
            MacroUnit((2, 3), 1, (0, 0, 0, 1)),
        )
        phis = {r.system.units: r.phi for r in result.records}
        assert phis[winner.units] == pytest.approx(1.0040208141253277, abs=1e-13)
        assert result.ties == ()


def test_public_surface_importable():
    from pyphi import macro

    for name in (
        "SearchBounds",
        "complexes",
        "intrinsic_units",
        "is_intrinsic_unit",
        "judge_candidate",
        "unit_integration",
        "valid_systems",
    ):
        assert hasattr(macro, name)


class TestMacroParallelConfig:
    def test_option_exists_with_family_defaults(self):
        from collections.abc import Mapping

        from pyphi import config

        option = config.infrastructure.parallel_macro_system_evaluation
        assert isinstance(option, Mapping)
        assert option["parallel"] is False
        assert option["sequential_threshold"] == 2**4
        assert option["chunksize"] == 2**6

    def test_global_switch_gates_the_option(self):
        # With the global switch off, the option's own parallel flag is
        # forced off (an explicit per-call override still wins, matching
        # the rest of the parallel-option family).
        from pyphi import conf
        from pyphi import config

        enabled = {
            "parallel": True,
            "sequential_threshold": 1,
            "chunksize": 1,
            "progress": False,
        }
        with config.override(parallel=False):
            gated = conf.parallel_kwargs(enabled)
            overridden = conf.parallel_kwargs(enabled, parallel=True)
        assert gated["parallel"] is False
        assert overridden["parallel"] is True


class TestEvaluateSystems:
    """The shared batch-evaluation helper that drives parallelism."""

    def _min_systems(self):
        sub = min_substrate()
        state = (0, 0)
        tables = [(0, 0, 0, 1), (0, 1, 1, 1), (0, 1, 1, 0)]
        return [
            MacroSystem.from_micro(sub, (MacroUnit((0, 1), 1, t),), (state,))
            for t in tables
        ]

    def test_in_process_matches_direct_sia_and_order(self):
        from pyphi.macro.search import _evaluate_systems

        systems = self._min_systems()
        with config.override(**presets.iit4_2023):
            reference = [s.sia().phi for s in systems]
            memo = {}
            _evaluate_systems(systems, memo, None)
        assert [memo[s] for s in systems] == reference

    def test_dedups_against_memo_and_within_batch(self):
        from pyphi.macro.search import _evaluate_systems

        systems = self._min_systems()
        with config.override(**presets.iit4_2023):
            memo = {systems[0]: 123.0}  # sentinel: must not recompute
            _evaluate_systems([systems[0], systems[1], systems[1]], memo, None)
        assert memo[systems[0]] == 123.0  # untouched
        assert systems[1] in memo

    def test_empty_input_is_noop(self):
        from pyphi.macro.search import _evaluate_systems

        memo = {}
        _evaluate_systems([], memo, None)
        _evaluate_systems([None, None], memo, None)
        assert memo == {}

    def test_parallel_path_matches_sequential(self):
        from pyphi.macro.search import _evaluate_systems

        systems = self._min_systems()
        enabled = {"parallel": True, "sequential_threshold": 1, "chunksize": 1}
        with config.override(**presets.iit4_2023):
            reference = [s.sia().phi for s in systems]
            memo = {}
            with config.override(parallel=True):
                _evaluate_systems(systems, memo, enabled)
        assert [memo[s] for s in systems] == reference


def _results_equal(a, b):
    """Field-identical ComplexesResult, including record order and
    bitwise phi."""
    assert a.complexes == b.complexes
    assert a.ties == b.ties
    assert [r.system for r in a.records] == [r.system for r in b.records]
    assert [float(r.phi) for r in a.records] == [float(r.phi) for r in b.records]


class TestParallelEquivalenceSweep:
    """The P(u) sweep under the macro parallel option reproduces the
    sequential ComplexesResult exactly."""

    def test_min_exhaustive_driver(self):
        bounds = SearchBounds(mappings="EXHAUSTIVE")
        enabled = {"parallel": True, "sequential_threshold": 1, "chunksize": 1}
        with config.override(**presets.iit4_2023):
            sequential = complexes(min_substrate(), (0, 0), bounds)
            with config.override(parallel=True):
                parallel = complexes(
                    min_substrate(), (0, 0), bounds, parallel_kwargs=enabled
                )
        _results_equal(sequential, parallel)

    def test_tie_path_driver(self):
        bounds = SearchBounds(max_constituents=2)
        enabled = {"parallel": True, "sequential_threshold": 1, "chunksize": 1}
        with config.override(**presets.iit4_2023):
            sequential = complexes(tie_substrate(), (0, 0, 0), bounds)
            with config.override(parallel=True):
                parallel = complexes(
                    tie_substrate(), (0, 0, 0), bounds, parallel_kwargs=enabled
                )
        _results_equal(sequential, parallel)
        assert len(parallel.complexes) == 1
        assert parallel.complexes[0].node_indices == (0, 1, 2)
        assert len(parallel.ties) == 6


class TestParallelEquivalenceRecursion:
    """A full default-bounds driver run (where the recursion does real
    work) under the parallel option reproduces the sequential result."""

    def test_dancing_couples_driver(self):
        enabled = {"parallel": True, "sequential_threshold": 1, "chunksize": 1}
        with config.override(**presets.iit4_2023):
            sequential = complexes(dancing_couples(0.25), SF_STATE)
            with config.override(parallel=True):
                parallel = complexes(
                    dancing_couples(0.25), SF_STATE, parallel_kwargs=enabled
                )
        _results_equal(sequential, parallel)

    def test_intrinsic_units_pool_identical(self):
        enabled = {"parallel": True, "sequential_threshold": 1, "chunksize": 1}
        with config.override(**presets.iit4_2023):
            seq = intrinsic_units(dancing_couples(0.25), SF_STATE, SearchBounds())
            with config.override(parallel=True):
                par = intrinsic_units(
                    dancing_couples(0.25),
                    SF_STATE,
                    SearchBounds(),
                    parallel_kwargs=enabled,
                )
        assert seq.units == par.units
        assert [v.constituents for v in seq.verdicts] == [
            v.constituents for v in par.verdicts
        ]
        assert [v.verdict.phi for v in seq.verdicts] == [
            v.verdict.phi for v in par.verdicts
        ]


class TestParallelGating:
    def test_default_config_runs_in_process(self, monkeypatch):
        # With the global switch off (the default), the driver must not
        # dispatch to a process pool: map_reduce is never called
        # (it is imported lazily from pyphi.parallel inside the helper).
        import pyphi.parallel

        def _boom(*args, **kwargs):
            raise AssertionError("map_reduce should not run under global parallel=False")

        monkeypatch.setattr(pyphi.parallel, "map_reduce", _boom)
        with config.override(**presets.iit4_2023):
            result = complexes(
                min_substrate(), (0, 0), SearchBounds(mappings="EXHAUSTIVE")
            )
        assert len(result.complexes) == 1


@pytest.mark.slow
class TestParallelCostGuard:
    """The default-bounds cg driver under the parallel option matches
    the sequential SP2 golden exactly."""

    def test_cg_driver_parallel_matches_golden(self):
        enabled = {"parallel": True, "sequential_threshold": 1}
        with config.override(**presets.iit4_2023):
            substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
            sequential = complexes(substrate, (0, 0, 0, 0))
            with config.override(parallel=True):
                parallel = complexes(substrate, (0, 0, 0, 0), parallel_kwargs=enabled)
        _results_equal(sequential, parallel)
        assert len(parallel.complexes) == 1
        winner = parallel.complexes[0]
        assert winner.units == (
            MacroUnit((0, 1), 1, (0, 0, 0, 1)),
            MacroUnit((2, 3), 1, (0, 0, 0, 1)),
        )
        phis = {r.system.units: r.phi for r in parallel.records}
        assert phis[winner.units] == pytest.approx(1.0040208141253277, abs=1e-13)


def decaying_chain_substrate():
    """4 units, reciprocal couplings 0.6 (0-1), 0.3 (1-2), 0.15 (2-3).

    The phi landscape is a chain: {A,B} > {B,C} > {C,D} with {B,C}
    overlapping both. Recursive condensation yields {A,B} and {C,D};
    a non-recursive local-maximum predicate would orphan {C,D}.
    """
    n = 4
    weights = np.zeros((n, n))
    weights[0, 1] = weights[1, 0] = 0.6
    weights[1, 2] = weights[2, 1] = 0.3
    weights[2, 3] = weights[3, 2] = 0.15
    for i in range(n):
        weights[i, i] = 0.05
    tpm = np.zeros((2**n, n))
    for row in range(2**n):
        s = np.array([(row >> k) & 1 for k in range(n)])
        tpm[row] = 0.05 + weights @ s
    return Substrate(tpm, node_labels=("A", "B", "C", "D"))


class TestRecursiveCondensation:
    def test_chain_yields_both_disjoint_complexes(self):
        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
        footprints = {c.node_indices for c in result.complexes}
        assert footprints == {(0, 1), (2, 3)}

    def test_winners_are_complex_objects_with_units_and_records(self):
        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
        from pyphi.models.complex import Complex

        top = result.complexes[0]
        assert isinstance(top, Complex)
        assert top.is_maximal
        assert top.node_indices == (0, 1)
        assert top.units is not None and len(top.units) == 2
        assert any(e.node_indices == (1, 2) for e in top.excluded)
        assert result.maximal_complex is top

    def test_matches_micro_door_on_the_chain(self):
        from pyphi.substrate import complexes as micro_complexes

        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            macro = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))
            micro = micro_complexes(substrate, (0, 0, 0, 0))
        assert {c.node_indices for c in macro.complexes} == {
            c.node_indices for c in micro
        }

    def test_iit3_rejected_eagerly(self):
        substrate = decaying_chain_substrate()
        state = (0, 0, 0, 0)
        with config.override(**presets.iit3):
            with pytest.raises(ValueError, match="IIT_3_0"):
                complexes(substrate, state, SearchBounds(max_depth=0))
            with pytest.raises(ValueError, match="IIT_3_0"):
                intrinsic_units(substrate, state, SearchBounds())
            with pytest.raises(ValueError, match="IIT_3_0"):
                valid_systems(substrate, state, SearchBounds())
            with pytest.raises(ValueError, match="IIT_3_0"):
                is_intrinsic_unit(substrate, micro_unit(0), state)
            with pytest.raises(ValueError, match="IIT_3_0"):
                competing_systems(substrate, micro_unit(0), state)

    def test_chain_margins_count_beaten_rivals_only(self):
        from pyphi.substrate import complexes as micro_complexes

        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            found = micro_complexes(substrate, (0, 0, 0, 0))
        by_idx = {c.node_indices: c for c in found}
        ab, cd = by_idx[(0, 1)], by_idx[(2, 3)]
        # {A,B} beat the runner-up {B,C} (phi 0.1041...).
        assert ab.exclusion_margin == pytest.approx(0.21581964583210878, abs=1e-13)
        # {C,D} carries four higher-phi shadows ({B,C} among them); the
        # margin ignores them and measures the gap to the best beaten
        # rival, the singleton {C} (phi 0.0227...).
        shadows = [e for e in cd.excluded if e.phi > float(cd.phi)]
        assert len(shadows) == 4
        assert cd.exclusion_margin == pytest.approx(0.01439865646353308, abs=1e-13)
        assert cd.effectively_tied is False


class TestCrossDoorEquivalence:
    """substrate.complexes and the macro driver at max_depth=0 condense
    the same candidate landscape: every subset of micro units, evaluated
    as identity-unit systems (which reproduce System results exactly)."""

    @settings(max_examples=8, deadline=None)
    @given(st.integers(min_value=0, max_value=10**6))
    def test_doors_agree_at_micro_grain(self, seed):
        from pyphi.substrate import complexes as micro_complexes

        rng = np.random.default_rng(seed)
        n = 3
        tpm = rng.uniform(0.05, 0.95, size=(2**n, n))
        substrate = Substrate(tpm)
        state = tuple(int(v) for v in rng.integers(0, 2, size=n))
        with config.override(**presets.iit4_2023):
            micro = micro_complexes(substrate, state)
            macro = complexes(substrate, state, SearchBounds(max_depth=0))
        assert {c.node_indices for c in macro.complexes} == {
            c.node_indices for c in micro
        }
        micro_phis = {c.node_indices: float(c.phi) for c in micro}
        for c in macro.complexes:
            # identity macroing agrees with the direct System path at
            # config precision, not bit-for-bit: the macro construction
            # performs the same arithmetic in a different order, so the
            # values can differ in the last ulps
            assert float(c.phi) == pytest.approx(micro_phis[c.node_indices], abs=1e-13)

    def test_fingerprint_dedupe_shadow_equality(self, monkeypatch):
        """Forcing full escalation (unique fingerprints) changes nothing."""
        substrate = tie_substrate()
        state = (0, 0, 0)
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            with_skip = complexes(substrate, state, bounds)

        from pyphi import condensation

        monkeypatch.setattr(condensation, "_fingerprint_key", lambda _system: object())
        with config.override(**presets.iit4_2023):
            without_skip = complexes(substrate, state, bounds)

        assert {c.node_indices for c in with_skip.complexes} == {
            c.node_indices for c in without_skip.complexes
        }
        assert [{tuple(s.units) for s in clique} for clique in with_skip.ties] == [
            {tuple(s.units) for s in clique} for clique in without_skip.ties
        ]

    def test_complexes_parallel_equals_sequential_on_the_chain(self):
        substrate = decaying_chain_substrate()
        state = (0, 0, 0, 0)
        enabled = {"parallel": True, "sequential_threshold": 1, "chunksize": 1}
        with config.override(**presets.iit4_2023):
            sequential = complexes(substrate, state, SearchBounds(max_depth=0))
            with config.override(parallel=True):
                parallel = complexes(
                    substrate,
                    state,
                    SearchBounds(max_depth=0),
                    parallel_kwargs=enabled,
                )
        assert [c.node_indices for c in sequential.complexes] == [
            c.node_indices for c in parallel.complexes
        ]
        assert [float(c.phi) for c in sequential.complexes] == [
            float(c.phi) for c in parallel.complexes
        ]

    def test_exclusion_invariants_on_the_chain_sweep(self):
        """Accepted complexes are disjoint; every exclusion record names an
        overlapping, non-accepted candidate."""
        substrate = decaying_chain_substrate()
        with config.override(**presets.iit4_2023):
            result = complexes(substrate, (0, 0, 0, 0), SearchBounds(max_depth=0))

        footprints = [set(c.node_indices) for c in result.complexes]
        for i, a in enumerate(footprints):
            for b in footprints[i + 1 :]:
                assert not (a & b)

        accepted = {c.node_indices for c in result.complexes}
        for c in result.complexes:
            for record in c.excluded:
                # every exclusion record names a candidate that overlaps
                # this complex and was not itself accepted. NOTE: an
                # excluded candidate may carry HIGHER phi than the complex
                # whose record it appears in -- on the chain, {C,D}'s
                # records include {B,C} (phi 0.104 > 0.037), which was
                # carved away by {A,B}. That is the recursive semantics
                # working as intended; do not assert record.phi <= c.phi.
                # (With grains in play a record MAY name an accepted
                # complex's exact footprint -- a rival grain over the same
                # micro units; this sweep is micro-only, so footprints are
                # unique and the containment check below is valid.)
                assert set(record.node_indices) & set(c.node_indices)
                assert record.node_indices not in accepted
        # the chain makes the higher-phi-excluded case concrete:
        by_units = {c.node_indices: c for c in result.complexes}
        assert any(
            record.phi > float(by_units[(2, 3)].phi)
            for record in by_units[(2, 3)].excluded
        )
