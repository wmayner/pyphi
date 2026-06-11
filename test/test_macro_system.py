"""Tests for pyphi.macro.system: the MacroSystem protocol implementer."""

import numpy as np
import pytest

from pyphi import config
from pyphi.conf import presets
from pyphi.core.tpm.factored import FactoredTPM
from pyphi.examples import EXAMPLES
from pyphi.macro import MacroSystem
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import blackbox
from pyphi.macro.units import coarse_grain
from pyphi.macro.units import micro_unit
from pyphi.substrate import Substrate
from pyphi.system import System
from test.test_macro_tpm import CG_TPM
from test.test_macro_tpm import _asymmetric_substrate


def _cg_macro_system():
    substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
    units = (
        MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
        MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
    )
    return MacroSystem.from_micro(substrate, units, ((0, 0, 0, 0),))


class TestConstruction:
    def test_bare_state_wrapped_when_all_grains_one(self):
        substrate = Substrate(CG_TPM)
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
        )
        system = MacroSystem.from_micro(substrate, units, (0, 0, 0, 0))
        assert system.micro_history == ((0, 0, 0, 0),)

    def test_bare_state_rejected_with_higher_grain(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1), 2, blackbox(2, 2, (0,))),)
        with pytest.raises(ValueError, match="history"):
            MacroSystem.from_micro(substrate, units, (1, 0, 1, 0))

    def test_history_length_must_be_max_grain(self):
        substrate = _asymmetric_substrate()
        units = (MacroUnit((0, 1), 2, blackbox(2, 2, (0,))),)
        with pytest.raises(ValueError, match="history"):
            MacroSystem.from_micro(
                substrate,
                units,
                ((1, 0, 1, 0), (1, 0, 1, 0), (1, 0, 1, 0)),
            )

    def test_macro_state_from_history(self):
        system = _cg_macro_system()
        assert system.state == (0, 0)

    def test_eq18_overlap_rejected(self):
        substrate = Substrate(CG_TPM)
        units = (
            MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
            MacroUnit((1, 2), 1, coarse_grain(2, on_counts={2})),
        )
        with pytest.raises(ValueError, match="disjoint"):
            MacroSystem.from_micro(substrate, units, (0, 0, 0, 0))

    def test_apportionment_inside_system_rejected(self):
        substrate = _asymmetric_substrate()
        units = (
            MacroUnit(
                (0, 1),
                1,
                coarse_grain(2, on_counts={2}),
                background_apportionment=(2,),
            ),
            MacroUnit((2,), 1, (0, 1)),
        )
        with pytest.raises(ValueError, match=r"disjoint|background"):
            MacroSystem.from_micro(substrate, units, (1, 0, 1, 0))

    def test_eq12_nested_apportionment_rejected(self):
        substrate = _asymmetric_substrate()
        inner = MacroUnit((0,), 1, (0, 1), background_apportionment=(3,))
        outer = MacroUnit((inner, 1), 1, coarse_grain(2, on_counts={2}))
        with pytest.raises(ValueError, match=r"Eq\. 12"):
            MacroSystem.from_micro(substrate, (outer,), ((1, 0, 1, 0),))

    def test_nonbinary_substrate_rejected(self):
        # A 1-node ternary substrate, built from a uniform factor
        factor = np.full((3, 3), 1 / 3)
        substrate = Substrate.from_factored(FactoredTPM(factors=[factor]))
        with pytest.raises(ValueError, match="binary"):
            MacroSystem.from_micro(substrate, (micro_unit(0),), ((0,),))

    def test_constituent_outside_substrate_rejected(self):
        substrate = _asymmetric_substrate()
        units = (micro_unit(7),)
        with pytest.raises(ValueError, match="substrate"):
            MacroSystem.from_micro(substrate, units, ((1, 0, 1, 0),))

    def test_from_substrate_directs_to_from_micro(self):
        with pytest.raises(TypeError, match="from_micro"):
            MacroSystem.from_substrate(None, None)


class TestProtocolSurface:
    def test_tpms_and_shape(self):
        system = _cg_macro_system()
        assert system.size == 2
        assert system.node_indices == (0, 1)
        assert np.array_equal(system.cm, np.ones((2, 2)))
        for tpm in (system.cause_tpm, system.effect_tpm):
            for k in range(2):
                assert tpm.factor(k).shape == (2, 2, 2)

    def test_proper_tpms_equal_tpms(self):
        system = _cg_macro_system()
        for k in range(2):
            assert np.array_equal(
                system.proper_cause_tpm.factor(k), system.cause_tpm.factor(k)
            )
            assert np.array_equal(
                system.proper_effect_tpm.factor(k), system.effect_tpm.factor(k)
            )

    def test_nodes_use_macro_tpms(self):
        system = _cg_macro_system()
        assert len(system.nodes) == 2

    def test_apply_cut_preserves_type_and_fields(self):
        from pyphi.partition import system_partitions

        system = _cg_macro_system()
        partition = next(
            iter(system_partitions(system.node_indices, system.node_labels))
        )
        cut = system.apply_cut(partition)
        assert isinstance(cut, MacroSystem)
        assert cut.units == system.units
        assert cut.micro_history == system.micro_history
        assert cut.is_partitioned

    def test_equality_and_hash(self):
        a = _cg_macro_system()
        b = _cg_macro_system()
        assert a == b
        assert hash(a) == hash(b)
        substrate = Substrate(CG_TPM, node_labels=("A", "B", "C", "D"))
        c = MacroSystem.from_micro(
            substrate,
            (
                MacroUnit((0, 1), 1, coarse_grain(2, on_counts={1, 2})),
                MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
            ),
            ((0, 0, 0, 0),),
        )
        assert a != c


def _identity_macro(substrate, state, subset, node_labels):
    units = tuple(micro_unit(i) for i in subset)
    return MacroSystem.from_micro(
        substrate, units, (tuple(state),), node_labels=node_labels
    )


def _examples_for_reduction():
    # The macro formalism defines no macro connectivity (cm is all-ones),
    # so micro comparators with explicit sparse cms are rebuilt as
    # all-ones-cm twins via Substrate.from_factored.
    basic = EXAMPLES["substrate"]["basic"]()
    xor = EXAMPLES["substrate"]["xor"]()
    xor_allones = Substrate.from_factored(xor.factored_tpm, node_labels=["A", "B", "C"])
    grid3 = EXAMPLES["substrate"]["grid3"]()
    grid3_allones = Substrate.from_factored(
        grid3.factored_tpm, node_labels=["A", "B", "C"]
    )
    return [
        (basic, (1, 0, 0)),
        (xor_allones, (0, 0, 0)),
        (grid3_allones, (0, 0, 0)),
    ]


class TestMicroReduction:
    """Identity macroing == System, exactly (paper p. 10)."""

    @pytest.mark.parametrize(
        "substrate,state",
        _examples_for_reduction(),
        ids=["basic", "xor", "grid3"],
    )
    def test_full_system_sia_matches(self, substrate, state):
        with config.override(**presets.iit4_2023):
            subset = tuple(range(substrate.size))
            labels = [str(label) for label in substrate.node_labels]
            macro = _identity_macro(substrate, state, subset, labels)
            micro = System(substrate, state)
            macro_sia = macro.sia()
            micro_sia = micro.sia()
            assert macro_sia.phi == micro_sia.phi
            assert macro_sia.partition == micro_sia.partition

    @pytest.mark.parametrize(
        "substrate,state",
        _examples_for_reduction(),
        ids=["basic", "xor", "grid3"],
    )
    def test_full_system_ces_matches(self, substrate, state):
        with config.override(**presets.iit4_2023):
            subset = tuple(range(substrate.size))
            labels = [str(label) for label in substrate.node_labels]
            macro = _identity_macro(substrate, state, subset, labels)
            micro = System(substrate, state)
            macro_ces = macro.ces()
            micro_ces = micro.ces()
            assert macro_ces.sia.phi == micro_ces.sia.phi
            macro_distinctions = list(macro_ces.distinctions)
            micro_distinctions = list(micro_ces.distinctions)
            assert len(macro_distinctions) == len(micro_distinctions)
            for m_dist, u_dist in zip(
                macro_distinctions, micro_distinctions, strict=True
            ):
                assert m_dist.mechanism == u_dist.mechanism
                assert m_dist.phi == u_dist.phi

    def test_subset_system_sia_matches(self):
        """Background path: identity macroing of a proper subset."""
        with config.override(**presets.iit4_2023):
            substrate = EXAMPLES["substrate"]["basic"]()
            state = (1, 0, 0)
            subset = (0, 1)
            macro = _identity_macro(substrate, state, subset, ["A", "B"])
            micro = System(substrate, state, subset)
            assert macro.sia().phi == micro.sia().phi
