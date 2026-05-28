import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi import config
from pyphi import exceptions
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.models import Concept
from pyphi.models import DirectedBipartition
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import MaximallyIrreducibleEffect
from pyphi.models import RepertoireIrreducibilityAnalysis
from pyphi.system import System

from . import example_substrates


@config.override(validate_system_states=True)
def test_system_validation(s):
    # Wrong state length.
    with pytest.raises(ValueError):
        s = System(s.substrate, (0, 0), s.node_indices)
    # Wrong state values.
    with pytest.raises(ValueError):
        s = System(s.substrate, (2, 0, 0), s.node_indices)
    # Disallow impossible states at system level (we don't want to return a
    # phi-value associated with an impossible state).
    net = example_substrates.simple()
    with pytest.raises(exceptions.StateUnreachableError):
        s = System(net, (0, 1, 0), s.node_indices)


def test_validate_partition_nodes_equal_system_nodes(s):
    assert s.node_indices == (0, 1, 2)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))  # A-ok
    System(s.substrate, s.state, s.node_indices, partition=cut)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1,))  # missing node 2 in cut
    with pytest.raises(ValueError):
        System(s.substrate, s.state, s.node_indices, partition=cut)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))  # missing node 2 in system
    with pytest.raises(ValueError):
        System(s.substrate, s.state, (0, 1), partition=cut)


def test_empty_init(s):
    # Empty mechanism
    s = System(s.substrate, s.state, ())
    assert s.nodes == ()


def test_default_nodes(s):
    s = System(s.substrate, s.state)
    assert s.node_indices == (0, 1, 2)


def test_eq(subsys_n0n2, subsys_n1n2):
    assert subsys_n0n2 == subsys_n0n2  # noqa: PLR0124
    assert subsys_n0n2 != subsys_n1n2
    assert subsys_n0n2 is not None
    assert subsys_n1n2 is not None


def test_cmp(subsys_n0n2, subsys_n1n2, s):
    assert s > subsys_n0n2
    assert s > subsys_n1n2
    assert subsys_n0n2 >= subsys_n1n2
    assert s >= subsys_n0n2
    assert subsys_n0n2 < s
    assert subsys_n1n2 < s
    assert subsys_n0n2 <= s
    assert subsys_n0n2 <= subsys_n1n2


def test_len(s, big_subsys_0_thru_3, big_subsys_all):
    assert len(s) == 3
    assert len(big_subsys_0_thru_3) == 4
    assert len(big_subsys_all) == 5


def test_size(s, big_subsys_0_thru_3, big_subsys_all):
    assert s.size == 3
    assert big_subsys_0_thru_3.size == 4
    assert big_subsys_all.size == 5


def test_hash(s):
    print(hash(s))


def test_indices2nodes(s):
    subsys = s  # 3-node system
    assert subsys.indices2nodes(()) == ()
    assert subsys.indices2nodes((1,)) == (subsys.nodes[1],)
    assert subsys.indices2nodes((0, 2)) == (subsys.nodes[0], subsys.nodes[2])


def test_indices2nodes_with_bad_indices(subsys_n1n2):
    with pytest.raises(ValueError):
        subsys_n1n2.indices2nodes((3, 4))  # indices not in substrate
    with pytest.raises(ValueError):
        subsys_n1n2.indices2nodes((0,))  # index n0 in substrate but not subsytem


def test_is_partitioned(s):
    assert s.is_partitioned is False
    s = System(
        s.substrate,
        s.state,
        s.node_indices,
        partition=DirectedBipartition(Direction.EFFECT, (0,), (1, 2)),
    )
    assert s.is_partitioned is True


def test_proper_state(subsys_n0n2, subsys_n1n2):
    # state == (1, 0, 0)
    assert subsys_n0n2.proper_state == (1, 0)
    assert subsys_n1n2.proper_state == (0, 0)


def test_apply_cut(s):
    cut = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    cut_s = s.apply_cut(cut)
    assert s.substrate == cut_s.substrate
    assert s.state == cut_s.state
    assert s.node_indices == cut_s.node_indices
    assert np.array_equal(cut_s.effect_tpm.tpm, s.effect_tpm.tpm)
    assert cut_s.cause_tpm == s.cause_tpm
    assert np.array_equal(cut_s.cm, cut.apply_cut(s.cm))


def test_partition_indices(s, subsys_n1n2):
    assert s.partition_indices == (0, 1, 2)
    assert subsys_n1n2.partition_indices == (1, 2)


def test_partitioned_mechanisms(s):
    assert list(s.partitioned_mechanisms) == []
    assert list(
        s.apply_cut(
            DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
        ).partitioned_mechanisms
    ) == [
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]


def test_partition_node_labels(s):
    assert s.partition_node_labels == s.node_labels


def test_specify_elements_with_labels(standard):
    substrate = Substrate(standard.joint_tpm(), node_labels=("A", "B", "C"))
    system = System(substrate, (0, 0, 0), ("B", "C"))
    assert system.node_indices == (1, 2)
    assert tuple(node.label for node in system.nodes) == ("B", "C")
    assert str(system) == "System(B, C)"


def test_null_concept(s):
    cause = MaximallyIrreducibleCause(
        RepertoireIrreducibilityAnalysis(
            repertoire=s.unconstrained_cause_repertoire(()),
            phi=0,
            direction=Direction.CAUSE,
            mechanism=(),
            purview=(),
            partition=None,
            partitioned_repertoire=None,
        )
    )
    effect = MaximallyIrreducibleEffect(
        RepertoireIrreducibilityAnalysis(
            repertoire=s.unconstrained_effect_repertoire(()),
            phi=0,
            direction=Direction.EFFECT,
            mechanism=(),
            purview=(),
            partition=None,
            partitioned_repertoire=None,
        )
    )
    assert s.null_concept == Concept(
        mechanism=(),
        cause=cause,
        effect=effect,
    )


def test_distinction_no_mechanism(s):
    assert s.distinction(()) == s.null_concept


def test_distinction_nonexistent(s):
    assert not s.distinction((0, 2))


class TestIntrinsicInformationTies:
    """Test that intrinsic_information() correctly tracks tied states."""

    def test_tied_states_are_stored(self):
        """intrinsic_information() should populate .ties when states tie.

        AND-XOR at state (0,1): the cause GID is 0.5 for both purview states
        (1,0) and (0,1), creating a tie. Both should appear in .ties.
        """
        net = example_substrates.and_xor_substrate()
        sub = System(net, (0, 1))
        spec = sub.intrinsic_information(
            Direction.CAUSE,
            (0, 1),
            (0, 1),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
        assert len(spec.ties) == 2
        tied_states = {t.state for t in spec.ties}
        assert tied_states == {(0, 1), (1, 0)}

    def test_tied_states_have_equal_ii(self):
        """All tied StateSpecifications must have the same intrinsic information."""
        net = example_substrates.and_xor_substrate()
        sub = System(net, (0, 1))
        spec = sub.intrinsic_information(
            Direction.CAUSE,
            (0, 1),
            (0, 1),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
        ii_values = {float(t.intrinsic_information) for t in spec.ties}
        assert len(ii_values) == 1, f"Tied states have different II: {ii_values}"

    def test_no_ties_when_unique_max(self):
        """When a single state uniquely maximizes II, ties should have length 1."""
        net = example_substrates.and_xor_substrate()
        sub = System(net, (0, 1))
        spec = sub.intrinsic_information(
            Direction.EFFECT,
            (0, 1),
            (0, 1),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
        # Effect direction should have a unique max (no tie)
        assert len(spec.ties) == 1

    def test_null_sia_resolve_system_state_safe(self):
        """resolve_system_state should be a no-op for NullSIA."""
        from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis

        null_sia = NullSystemIrreducibilityAnalysis()
        null_sia.resolve_system_state()  # Should not raise


def test_proper_cause_tpm_kary_returns_factored_view() -> None:
    """proper_cause_tpm for a k=3 system returns per-system-unit factors
    restricted to the system's node indices."""
    from pyphi.core.tpm.factored import FactoredTPM

    rng = np.random.default_rng(99)
    factors = []
    for _ in range(2):
        arr = rng.uniform(size=(3, 3, 3))
        factors.append(arr / arr.sum(axis=-1, keepdims=True))
    sub = Substrate(marginals=factors)
    with config.override(validate_system_states=False):
        sys = System(sub, state=(0, 0))
        proper = sys.proper_cause_tpm
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(sys.node_indices)
    for i in range(proper.n_nodes):
        assert proper.factor(i).shape[-1] == 3


def test_proper_cause_tpm_binary_matches_legacy_slice(s) -> None:
    """For binary substrates the per-system-unit on-probability slice of
    ``proper_cause_tpm`` matches the substrate cause TPM's on-probability
    slice for the same unit."""
    from pyphi.core.tpm.factored import FactoredTPM

    substrate_cause = s.cause_tpm
    assert isinstance(substrate_cause, FactoredTPM)
    proper = s.proper_cause_tpm
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(s.node_indices)
    for slot, node in enumerate(s.node_indices):
        proper_on = np.squeeze(proper.factor(slot)[..., 1])
        substrate_on = np.squeeze(substrate_cause.factor(node)[..., 1])
        assert np.allclose(proper_on, substrate_on, atol=1e-10)


# external_indices field tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_system_external_indices_default_matches_today(subsys_n1n2) -> None:
    """When external_indices=None (default), resolves to substrate - node_indices."""
    assert subsys_n1n2.external_indices == (0,)


def test_system_external_indices_full_system_has_empty_default(s) -> None:
    """When node_indices covers the whole substrate, default external is ()."""
    assert s.external_indices == ()


def test_system_external_indices_explicit_override_accepted(s) -> None:
    """An explicit override is stored and accepted."""
    overridden = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(1, 2),
        external_indices=(),
    )
    assert overridden.external_indices == ()


def test_system_external_indices_rejects_out_of_range(s) -> None:
    """Indices must be in range(substrate.size)."""
    with pytest.raises(ValueError, match="out of range"):
        System(
            substrate=s.substrate,
            state=s.state,
            node_indices=(0, 1),
            external_indices=(99,),
        )


def test_system_external_indices_rejects_negative(s) -> None:
    """Negative indices are out of range."""
    with pytest.raises(ValueError, match="out of range"):
        System(
            substrate=s.substrate,
            state=s.state,
            node_indices=(0, 1),
            external_indices=(-1,),
        )


def test_system_external_indices_rejects_unsorted(s) -> None:
    """Override must be sorted."""
    with pytest.raises(ValueError, match="sorted"):
        System(
            substrate=s.substrate,
            state=s.state,
            node_indices=(0,),
            external_indices=(2, 1),
        )


def test_system_external_indices_rejects_duplicates(s) -> None:
    """Override must have unique entries."""
    with pytest.raises(ValueError, match="duplicate"):
        System(
            substrate=s.substrate,
            state=s.state,
            node_indices=(0,),
            external_indices=(1, 1),
        )


def test_system_external_indices_included_in_eq(s) -> None:
    """Two Systems differing only in external_indices are not equal."""
    s_default = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
    )
    s_override = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
        external_indices=(),
    )
    # default resolves to (2,), override is (), so they differ.
    assert s_default.external_indices == (2,)
    assert s_override.external_indices == ()
    assert s_default != s_override


def test_system_external_indices_default_equal_to_explicit_same_value(s) -> None:
    """None-default and explicit-default-value resolve to the same field
    value and therefore compare equal."""
    s_implicit = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
    )
    s_explicit = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
        external_indices=(2,),
    )
    assert s_implicit == s_explicit
    assert hash(s_implicit) == hash(s_explicit)


def test_system_external_indices_included_in_hash(s) -> None:
    """Hash distinguishes Systems with different external_indices."""
    s1 = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
    )
    s2 = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
        external_indices=(),
    )
    assert hash(s1) != hash(s2)


def test_system_external_indices_apply_cut_propagates(s) -> None:
    """apply_cut returns new System with external_indices preserved."""
    from pyphi.models.partitions import DirectedBipartition as _DBP

    base = System(
        substrate=s.substrate,
        state=s.state,
        node_indices=(0, 1),
        external_indices=(),
    )
    new_cut = _DBP(Direction.EFFECT, (0,), (1,), s.substrate.node_labels)
    cut = base.apply_cut(new_cut)
    assert cut.external_indices == ()


def test_system_external_indices_overlap_with_node_indices_allowed(s) -> None:
    """The override may overlap with node_indices (the AC use case).

    Validation is disabled because the AC use case (freezing an in-system
    unit at observed state) creates conditioned dynamics that may not
    cover the full state. Reachability is not what this test checks.
    """
    with config.override(validate_system_states=False):
        overlapping = System(
            substrate=s.substrate,
            state=s.state,
            node_indices=(0, 1, 2),
            external_indices=(1,),
        )
    assert overlapping.external_indices == (1,)
