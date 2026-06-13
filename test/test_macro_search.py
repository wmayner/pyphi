"""Tests for pyphi.macro.search: bounded intrinsic-unit search (Eqs 15-19)."""

import numpy as np
import pytest

from pyphi import config
from pyphi import utils
from pyphi.conf import presets
from pyphi.macro.criteria import Reason
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
from test.test_macro_criteria import bu_substrate
from test.test_macro_criteria import min_substrate
from test.test_macro_tpm import CG_TPM
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
            tuple(u.micro_constituents for u in s.units) != ((2,),)
            for s in systems
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
        phis = {r.system: r.phi for r in result.records}
        assert phis[winner] == pytest.approx(0.7883339770634884, abs=1e-13)
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
            tuple(u.micro_constituents for u in s.units)
            for s in result.complexes
        }
        assert footprints == {((0,),), ((1,),)}
        phis = {r.system: r.phi for r in result.records}
        assert all(phis[s] == 1.0 for s in result.complexes)
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
    footprint {A,B} has a permutation-identical twin on {B,C}; the top
    pair (the (0,1,1,1)-mapped one-unit systems, measured at
    ~0.3881829280978132 during planning) overlap at B and tie at
    precision, so neither is a complex and nothing else can beat them."""

    def test_no_complex_on_tie(self):
        bounds = SearchBounds(max_constituents=2)
        with config.override(**presets.iit4_2023):
            result = complexes(tie_substrate(), (0, 0, 0), bounds)
        assert result.complexes == ()
        assert len(result.ties) == 1
        a, b = result.ties[0]
        assert {
            tuple(u.micro_constituents for u in s.units) for s in (a, b)
        } == {((0, 1),), ((1, 2),)}
        assert all(s.units[0].mapping == (0, 1, 1, 1) for s in (a, b))
        phis = {r.system: r.phi for r in result.records}
        assert utils.eq(phis[a], phis[b])
        assert phis[a] == pytest.approx(0.3881829280978132, abs=1e-13)


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
        phis = {r.system: r.phi for r in result.records}
        assert phis[winner] == pytest.approx(1.0040208141253277, abs=1e-13)
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
        from pyphi.data_structures.pyphi_float import PyPhiFloat
        from pyphi.macro.search import _evaluate_systems

        systems = self._min_systems()
        with config.override(**presets.iit4_2023):
            memo = {systems[0]: PyPhiFloat(123.0)}  # sentinel: must not recompute
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
        assert parallel.complexes == ()
        assert len(parallel.ties) == 1


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
        # dispatch to a process pool: MapReduce is never constructed
        # (it is imported lazily from pyphi.parallel inside the helper).
        import pyphi.parallel

        def _boom(*args, **kwargs):
            raise AssertionError("MapReduce should not run under global parallel=False")

        monkeypatch.setattr(pyphi.parallel, "MapReduce", _boom)
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
        phis = {r.system: r.phi for r in parallel.records}
        assert phis[winner] == pytest.approx(1.0040208141253277, abs=1e-13)
