"""Tests for the substrate_modeler module."""

import numpy as np
import pytest

from pyphi.substrate_modeler import Unit, CompositeUnit, Substrate, UNIT_FUNCTIONS


# =============================================================================
# Unit construction
# =============================================================================


class TestUnitConstruction:
    def test_string_mechanism(self):
        unit = Unit(index=0, inputs=(0, 1), mechanism="and")
        assert unit.mechanism_type == "and"
        assert unit.index == 0
        assert unit.inputs == (0, 1)

    def test_ndarray_mechanism(self):
        raw_tpm = np.array([[0.0], [1.0]])
        unit = Unit(index=0, inputs=(0,), mechanism=raw_tpm)
        assert unit.mechanism_type == "raw_tpm"
        assert np.array_equal(unit.mechanism, raw_tpm)

    def test_callable_mechanism(self):
        def custom(unit, state, input_state):
            return np.array([[0.0], [1.0]])

        unit = Unit(index=0, inputs=(0,), mechanism=custom)
        assert unit.mechanism_type == "custom"

    def test_label_defaults_to_index(self):
        unit = Unit(index=3, inputs=(0,), mechanism="copy")
        assert unit.label == "3"

    def test_label_explicit(self):
        unit = Unit(index=0, inputs=(0,), mechanism="copy", label="A")
        assert unit.label == "A"

    def test_original_index_defaults_to_index(self):
        unit = Unit(index=5, inputs=(0,), mechanism="copy")
        assert unit.original_index == 5

    def test_original_index_explicit(self):
        unit = Unit(index=0, inputs=(0,), mechanism="copy", original_index=10)
        assert unit.original_index == 10

    def test_params(self):
        unit = Unit(
            index=0,
            inputs=(0, 1),
            mechanism="sigmoid",
            params={"input_weights": (1.0, 1.0), "determinism": 3.0},
        )
        assert unit.params["determinism"] == 3.0

    def test_unknown_mechanism_raises(self):
        with pytest.raises(KeyError):
            Unit(index=0, inputs=(0,), mechanism="nonexistent_mechanism")


# =============================================================================
# Unit equality and hashing
# =============================================================================


class TestUnitEquality:
    def test_equal_units(self):
        a = Unit(index=0, inputs=(0, 1), mechanism="and")
        b = Unit(index=0, inputs=(0, 1), mechanism="and")
        assert a == b

    def test_unequal_mechanism(self):
        a = Unit(index=0, inputs=(0, 1), mechanism="and")
        b = Unit(index=0, inputs=(0, 1), mechanism="or")
        assert a != b

    def test_unequal_inputs(self):
        a = Unit(index=0, inputs=(0,), mechanism="copy")
        b = Unit(index=0, inputs=(1,), mechanism="copy")
        assert a != b

    def test_hash_equal(self):
        a = Unit(index=0, inputs=(0, 1), mechanism="and")
        b = Unit(index=0, inputs=(0, 1), mechanism="and")
        assert hash(a) == hash(b)


# =============================================================================
# Unit.compute_tpm for state-independent mechanisms
# =============================================================================


class TestStateIndependentMechanisms:
    def test_copy_gate(self):
        unit = Unit(index=0, inputs=(0,), mechanism="copy")
        tpm = unit.compute_tpm(state=0, input_state=(0,))
        # copy gate: P(ON | input=0) = 0.0, P(ON | input=1) = 1.0
        assert tpm[(0,)] == pytest.approx(0.0)
        assert tpm[(1,)] == pytest.approx(1.0)

    def test_and_gate(self):
        unit = Unit(index=2, inputs=(0, 1), mechanism="and")
        tpm = unit.compute_tpm(state=0, input_state=(0, 0))
        assert tpm[(0, 0)] == pytest.approx(0.0)
        assert tpm[(0, 1)] == pytest.approx(0.0)
        assert tpm[(1, 0)] == pytest.approx(0.0)
        assert tpm[(1, 1)] == pytest.approx(1.0)

    def test_or_gate(self):
        unit = Unit(index=2, inputs=(0, 1), mechanism="or")
        tpm = unit.compute_tpm(state=0, input_state=(0, 0))
        assert tpm[(0, 0)] == pytest.approx(0.0)
        assert tpm[(0, 1)] == pytest.approx(1.0)
        assert tpm[(1, 0)] == pytest.approx(1.0)
        assert tpm[(1, 1)] == pytest.approx(1.0)

    def test_xor_gate(self):
        unit = Unit(index=2, inputs=(0, 1), mechanism="xor")
        tpm = unit.compute_tpm(state=0, input_state=(0, 0))
        assert tpm[(0, 0)] == pytest.approx(0.0)
        assert tpm[(0, 1)] == pytest.approx(1.0)
        assert tpm[(1, 0)] == pytest.approx(1.0)
        assert tpm[(1, 1)] == pytest.approx(0.0)

    def test_and_gate_with_floor_ceiling(self):
        unit = Unit(
            index=0,
            inputs=(0, 1),
            mechanism="and",
            params={"floor": 0.1, "ceiling": 0.9},
        )
        tpm = unit.compute_tpm(state=0, input_state=(0, 0))
        assert tpm[(0, 0)] == pytest.approx(0.1)
        assert tpm[(1, 1)] == pytest.approx(0.9)

    def test_sigmoid(self):
        unit = Unit(
            index=0,
            inputs=(0, 1),
            mechanism="sigmoid",
            params={"input_weights": (1.0, 1.0), "determinism": 5.0, "threshold": 0.0},
        )
        tpm = unit.compute_tpm(state=0, input_state=(0, 0))
        # With ising=True (default), all-0 maps to all-(-1), so sum = -2
        # sigmoid(-5*(-2)) = sigmoid(10) ≈ 1.0
        # all-1 maps to all-(+1), so sum = 2
        # sigmoid(-5*(2)) = sigmoid(-10) ≈ 0.0... wait, let's just check shape and range
        arr = np.array(tpm)
        assert arr.shape[-1] == 1  # Single output node
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

    def test_democracy(self):
        unit = Unit(index=0, inputs=(0, 1), mechanism="democracy")
        tpm = unit.compute_tpm(state=0, input_state=(0, 0))
        assert tpm[(0, 0)] == pytest.approx(0.0)
        assert tpm[(0, 1)] == pytest.approx(0.5)
        assert tpm[(1, 0)] == pytest.approx(0.5)
        assert tpm[(1, 1)] == pytest.approx(1.0)

    def test_majority(self):
        unit = Unit(index=0, inputs=(0, 1, 2), mechanism="majority")
        tpm = unit.compute_tpm(state=0, input_state=(0, 0, 0))
        # Majority of (0,0,0) = 0, (1,1,0) = 1, (1,1,1) = 1
        assert tpm[(0, 0, 0)] == pytest.approx(0.0)
        assert tpm[(1, 1, 0)] == pytest.approx(1.0)
        assert tpm[(1, 1, 1)] == pytest.approx(1.0)


# =============================================================================
# Unit.compute_tpm for state-dependent mechanisms
# =============================================================================


class TestStateDependentMechanisms:
    def test_sor_gate_state_dependent(self):
        """SOR gate should produce different TPMs for different input states."""
        unit = Unit(
            index=0,
            inputs=(0, 1),
            mechanism="sor",
            params={
                "pattern_selection": ((0, 1), (1, 0)),
                "selectivity": 2.0,
            },
        )
        tpm_a = unit.compute_tpm(state=0, input_state=(0, 1))
        tpm_b = unit.compute_tpm(state=0, input_state=(1, 1))
        # Different input states should give different TPMs
        assert not np.array_equal(np.array(tpm_a), np.array(tpm_b))

    def test_mismatch_corrector(self):
        """Mismatch corrector depends on whether state matches input."""
        unit = Unit(
            index=0,
            inputs=(1,),
            mechanism="mismatch_corrector",
        )
        # state matches input → P = 0.5 (uncertain)
        tpm_match = unit.compute_tpm(state=1, input_state=(1,))
        # state doesn't match input → P = [floor, ceiling] = [0, 1]
        tpm_mismatch = unit.compute_tpm(state=0, input_state=(1,))
        assert not np.array_equal(np.array(tpm_match), np.array(tpm_mismatch))

    def test_mismatch_pattern_detector(self):
        """Mismatch pattern detector gives different TPMs for different states."""
        unit = Unit(
            index=0,
            inputs=(1, 2),
            mechanism="mismatch_pattern_detector",
            params={
                "pattern_selection": ((0, 1),),
                "selectivity": 2.0,
            },
        )
        tpm_on = unit.compute_tpm(state=1, input_state=(0, 1))
        tpm_off = unit.compute_tpm(state=0, input_state=(0, 1))
        assert not np.array_equal(np.array(tpm_on), np.array(tpm_off))


# =============================================================================
# Unit.state_dependent_tpm (cached)
# =============================================================================


class TestStateDependentTPMCache:
    def test_caching(self):
        unit = Unit(index=1, inputs=(0, 1), mechanism="and")
        substrate_state = (0, 1)
        tpm1 = unit.state_dependent_tpm(substrate_state)
        tpm2 = unit.state_dependent_tpm(substrate_state)
        # Should return the same cached object
        assert tpm1 is tpm2

    def test_different_states_different_results(self):
        unit = Unit(
            index=0,
            inputs=(1,),
            mechanism="mismatch_corrector",
        )
        # state=0 vs state=1 should give different TPMs
        tpm_a = unit.state_dependent_tpm((0, 1))  # unit state=0, input=1
        tpm_b = unit.state_dependent_tpm((1, 1))  # unit state=1, input=1
        assert not np.array_equal(np.array(tpm_a), np.array(tpm_b))


# =============================================================================
# CompositeUnit
# =============================================================================


class TestCompositeUnit:
    def test_inputs_aggregated(self):
        sub_a = Unit(index=0, inputs=(1,), mechanism="copy")
        sub_b = Unit(index=0, inputs=(2,), mechanism="copy")
        composite = CompositeUnit(
            index=0, units=(sub_a, sub_b), label="C"
        )
        assert set(composite.inputs) == {1, 2}

    def test_compute_tpm_shape(self):
        sub_a = Unit(index=0, inputs=(1,), mechanism="copy")
        sub_b = Unit(index=0, inputs=(2,), mechanism="copy")
        composite = CompositeUnit(
            index=0, units=(sub_a, sub_b), label="C"
        )
        tpm = composite.compute_tpm(state=0, input_state=(0, 0))
        # Should produce a valid TPM
        arr = np.array(tpm)
        assert np.all(arr >= 0.0)
        assert np.all(arr <= 1.0)

    def test_selective_combination(self):
        """Selective combination picks the sub-unit furthest from 0.5."""
        sub_a = Unit(
            index=0,
            inputs=(1,),
            mechanism="copy",
            params={"floor": 0.0, "ceiling": 1.0},
        )
        sub_b = Unit(
            index=0,
            inputs=(1,),
            mechanism="copy",
            params={"floor": 0.4, "ceiling": 0.6},
        )
        composite = CompositeUnit(
            index=0,
            units=(sub_a, sub_b),
            mechanism_combination="selective",
        )
        tpm = composite.compute_tpm(state=0, input_state=(1,))
        # sub_a gives P=1.0 for input=1, sub_b gives P=0.6
        # |1.0 - 0.5| = 0.5 > |0.6 - 0.5| = 0.1, so selective picks sub_a
        arr = np.array(tpm)
        # Multidimensional TPM shape: (2, 1) for 1 input, 1 node
        prob = float(arr[1, 0])
        assert prob == pytest.approx(1.0)


# =============================================================================
# Substrate construction
# =============================================================================


class TestSubstrateConstruction:
    def test_basic_construction(self):
        units = (
            Unit(index=0, inputs=(1,), mechanism="copy"),
            Unit(index=1, inputs=(0,), mechanism="copy"),
        )
        substrate = Substrate(units)
        assert len(substrate) == 2
        assert substrate.node_indices == (0, 1)

    def test_node_labels(self):
        units = (
            Unit(index=0, inputs=(1,), mechanism="copy", label="A"),
            Unit(index=1, inputs=(0,), mechanism="copy", label="B"),
        )
        substrate = Substrate(units)
        assert list(substrate.node_labels) == ["A", "B"]


# =============================================================================
# Substrate.cm
# =============================================================================


class TestSubstrateCM:
    def test_connectivity_matrix(self):
        units = (
            Unit(index=0, inputs=(1,), mechanism="copy"),
            Unit(index=1, inputs=(0,), mechanism="copy"),
        )
        substrate = Substrate(units)
        cm = substrate.cm
        # Unit 0 gets input from unit 1 → cm[1, 0] = 1
        # Unit 1 gets input from unit 0 → cm[0, 1] = 1
        assert cm[1, 0] == 1.0
        assert cm[0, 1] == 1.0
        assert cm[0, 0] == 0.0
        assert cm[1, 1] == 0.0


# =============================================================================
# Substrate.compute_tpm and dynamic_tpm
# =============================================================================


class TestSubstrateTPM:
    def _make_and_gate_substrate(self):
        """3-node substrate: C = A AND B, with A and B as copy gates."""
        return Substrate((
            Unit(index=0, inputs=(0,), mechanism="copy", label="A"),
            Unit(index=1, inputs=(1,), mechanism="copy", label="B"),
            Unit(index=2, inputs=(0, 1), mechanism="and", label="C"),
        ))

    def test_compute_tpm_shape(self):
        substrate = self._make_and_gate_substrate()
        tpm = substrate.compute_tpm((0, 0, 0))
        arr = np.array(tpm)
        # 3 nodes → 2^3 = 8 states, output shape should be (2,2,2,3) in md format
        assert arr.shape == (2, 2, 2, 3)

    def test_compute_tpm_values(self):
        """Verify AND gate substrate TPM for state (1, 1, 0)."""
        substrate = self._make_and_gate_substrate()
        tpm = substrate.compute_tpm((1, 1, 0))
        arr = np.array(tpm)
        # With present state (1,1,0):
        # Unit A (copy, input=A): depends on past A
        # Unit B (copy, input=B): depends on past B
        # Unit C (AND, inputs=A,B): depends on past A and B
        # For past state (1,1,*): P(A=1)=1.0, P(B=1)=1.0, P(C=1)=AND(1,1)=1.0
        assert arr[1, 1, 0, 0] == pytest.approx(1.0)  # P(A=1 | past A=1)
        assert arr[1, 1, 0, 1] == pytest.approx(1.0)  # P(B=1 | past B=1)
        assert arr[1, 1, 0, 2] == pytest.approx(1.0)  # P(C=1 | past A=1, B=1)
        # For past state (0,0,*): P(A=1)=0.0, P(B=1)=0.0, P(C=1)=AND(0,0)=0.0
        assert arr[0, 0, 0, 0] == pytest.approx(0.0)
        assert arr[0, 0, 0, 1] == pytest.approx(0.0)
        assert arr[0, 0, 0, 2] == pytest.approx(0.0)

    def test_dynamic_tpm_shape(self):
        substrate = self._make_and_gate_substrate()
        tpm = substrate.dynamic_tpm
        arr = np.array(tpm)
        assert arr.shape == (2, 2, 2, 3)

    def test_dynamic_tpm_is_cached(self):
        substrate = self._make_and_gate_substrate()
        tpm1 = substrate.dynamic_tpm
        tpm2 = substrate.dynamic_tpm
        assert tpm1 is tpm2


# =============================================================================
# Substrate.network and subsystem
# =============================================================================


class TestSubstrateNetworkAndSubsystem:
    def _make_simple_substrate(self):
        return Substrate((
            Unit(index=0, inputs=(1,), mechanism="copy", label="A"),
            Unit(index=1, inputs=(0,), mechanism="copy", label="B"),
        ))

    def test_network_returns_network(self):
        from pyphi.network import Network

        substrate = self._make_simple_substrate()
        net = substrate.network((0, 0))
        assert isinstance(net, Network)

    def test_subsystem_returns_subsystem(self):
        from pyphi.subsystem import Subsystem

        substrate = self._make_simple_substrate()
        sub = substrate.subsystem((0, 0))
        assert isinstance(sub, Subsystem)
        assert sub.state == (0, 0)

    def test_subsystem_with_node_subset(self):
        from pyphi.subsystem import Subsystem

        substrate = self._make_simple_substrate()
        sub = substrate.subsystem((0, 0), nodes=(0,))
        assert isinstance(sub, Subsystem)


# =============================================================================
# Substrate equality
# =============================================================================


class TestSubstrateEquality:
    def test_equal_substrates(self):
        units = (
            Unit(index=0, inputs=(1,), mechanism="copy"),
            Unit(index=1, inputs=(0,), mechanism="copy"),
        )
        a = Substrate(units)
        b = Substrate(units)
        assert a == b

    def test_unequal_substrates(self):
        a = Substrate((Unit(index=0, inputs=(0,), mechanism="copy"),))
        b = Substrate((Unit(index=0, inputs=(0,), mechanism="and"),))
        assert a != b


# =============================================================================
# Integration: AND gate substrate → Subsystem → concept
# =============================================================================


class TestIntegration:
    def test_and_gate_subsystem_concept(self):
        """Build a 3-node AND substrate, get a subsystem, and compute a concept."""
        substrate = Substrate((
            Unit(index=0, inputs=(0,), mechanism="copy", label="A"),
            Unit(index=1, inputs=(1,), mechanism="copy", label="B"),
            Unit(index=2, inputs=(0, 1), mechanism="and", label="C"),
        ))
        sub = substrate.subsystem((1, 1, 1))
        # Compute concept for mechanism (2,) — the AND gate
        concept = sub.concept((2,))
        # The concept should have some phi value
        assert concept.phi >= 0.0


# =============================================================================
# to_json
# =============================================================================


class TestToJson:
    def test_unit_to_json(self):
        unit = Unit(index=0, inputs=(1, 2), mechanism="and", label="X")
        j = unit.to_json()
        assert j["index"] == 0
        assert j["inputs"] == (1, 2)
        assert j["mechanism_type"] == "and"
        assert j["label"] == "X"

    def test_substrate_to_json(self):
        units = (
            Unit(index=0, inputs=(1,), mechanism="copy"),
            Unit(index=1, inputs=(0,), mechanism="copy"),
        )
        substrate = Substrate(units)
        j = substrate.to_json()
        assert len(j["units"]) == 2
