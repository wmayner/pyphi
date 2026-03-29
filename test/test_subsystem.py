import numpy as np
import pytest

from pyphi import Direction
from pyphi import Network
from pyphi import config
from pyphi import exceptions
from pyphi.models import Concept
from pyphi.models import Cut
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import MaximallyIrreducibleEffect
from pyphi.models import RepertoireIrreducibilityAnalysis
from pyphi.subsystem import Subsystem

from . import example_networks


@config.override(VALIDATE_SUBSYSTEM_STATES=True)
def test_subsystem_validation(s):
    # Wrong state length.
    with pytest.raises(ValueError):
        s = Subsystem(s.network, (0, 0), s.node_indices)
    # Wrong state values.
    with pytest.raises(ValueError):
        s = Subsystem(s.network, (2, 0, 0), s.node_indices)
    # Disallow impossible states at subsystem level (we don't want to return a
    # phi-value associated with an impossible state).
    net = example_networks.simple()
    with pytest.raises(exceptions.StateUnreachableError):
        s = Subsystem(net, (0, 1, 0), s.node_indices)


def test_validate_cut_nodes_equal_subsystem_nodes(s):
    assert s.node_indices == (0, 1, 2)

    cut = Cut((0,), (1, 2))  # A-ok
    Subsystem(s.network, s.state, s.node_indices, cut=cut)

    cut = Cut((0,), (1,))  # missing node 2 in cut
    with pytest.raises(ValueError):
        Subsystem(s.network, s.state, s.node_indices, cut=cut)

    cut = Cut((0,), (1, 2))  # missing node 2 in subsystem
    with pytest.raises(ValueError):
        Subsystem(s.network, s.state, (0, 1), cut=cut)


def test_empty_init(s):
    # Empty mechanism
    s = Subsystem(s.network, s.state, ())
    assert s.nodes == ()


def test_default_nodes(s):
    s = Subsystem(s.network, s.state)
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
    subsys = s  # 3-node subsystem
    assert subsys.indices2nodes(()) == ()
    assert subsys.indices2nodes((1,)) == (subsys.nodes[1],)
    assert subsys.indices2nodes((0, 2)) == (subsys.nodes[0], subsys.nodes[2])


def test_indices2nodes_with_bad_indices(subsys_n1n2):
    with pytest.raises(ValueError):
        subsys_n1n2.indices2nodes((3, 4))  # indices not in network
    with pytest.raises(ValueError):
        subsys_n1n2.indices2nodes((0,))  # index n0 in network but not subsytem


def test_is_cut(s):
    assert s.is_cut is False
    s = Subsystem(s.network, s.state, s.node_indices, cut=Cut((0,), (1, 2)))
    assert s.is_cut is True


def test_proper_state(subsys_n0n2, subsys_n1n2):
    # state == (1, 0, 0)
    assert subsys_n0n2.proper_state == (1, 0)
    assert subsys_n1n2.proper_state == (0, 0)


def test_apply_cut(s):
    cut = Cut((0, 1), (2,))
    cut_s = s.apply_cut(cut)
    assert s.network == cut_s.network
    assert s.state == cut_s.state
    assert s.node_indices == cut_s.node_indices
    assert np.array_equal(cut_s.effect_tpm.tpm, s.effect_tpm.tpm)
    assert np.array_equal(cut_s.cause_tpm.tpm, s.cause_tpm.tpm)
    assert np.array_equal(cut_s.cm, cut.apply_cut(s.cm))


def test_cut_indices(s, subsys_n1n2):
    assert s.cut_indices == (0, 1, 2)
    assert subsys_n1n2.cut_indices == (1, 2)


def test_cut_mechanisms(s):
    assert list(s.cut_mechanisms) == []
    assert list(s.apply_cut(Cut((0, 1), (2,))).cut_mechanisms) == [
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]


def test_cut_node_labels(s):
    assert s.cut_node_labels == s.node_labels


def test_specify_elements_with_labels(standard):
    network = Network(standard.tpm.tpm, node_labels=("A", "B", "C"))
    subsystem = Subsystem(network, (0, 0, 0), ("B", "C"))
    assert subsystem.node_indices == (1, 2)
    assert tuple(node.label for node in subsystem.nodes) == ("B", "C")
    assert str(subsystem) == "Subsystem(B, C)"


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


def test_concept_no_mechanism(s):
    assert s.concept(()) == s.null_concept


def test_concept_nonexistent(s):
    assert not s.concept((0, 2))


class TestIntrinsicInformationTies:
    """Test that intrinsic_information() correctly tracks tied states."""

    def test_tied_states_are_stored(self):
        """intrinsic_information() should populate .ties when states tie.

        AND-XOR at state (0,1): the cause GID is 0.5 for both purview states
        (1,0) and (0,1), creating a tie. Both should appear in .ties.
        """
        net = example_networks.and_xor_network()
        sub = Subsystem(net, (0, 1))
        spec = sub.intrinsic_information(Direction.CAUSE, (0, 1), (0, 1))
        assert len(spec.ties) == 2
        tied_states = {t.state for t in spec.ties}
        assert tied_states == {(0, 1), (1, 0)}

    def test_tied_states_have_equal_ii(self):
        """All tied StateSpecifications must have the same intrinsic information."""
        net = example_networks.and_xor_network()
        sub = Subsystem(net, (0, 1))
        spec = sub.intrinsic_information(Direction.CAUSE, (0, 1), (0, 1))
        ii_values = {float(t.intrinsic_information) for t in spec.ties}
        assert len(ii_values) == 1, f"Tied states have different II: {ii_values}"

    def test_no_ties_when_unique_max(self):
        """When a single state uniquely maximizes II, ties should have length 1."""
        net = example_networks.and_xor_network()
        sub = Subsystem(net, (0, 1))
        spec = sub.intrinsic_information(Direction.EFFECT, (0, 1), (0, 1))
        # Effect direction should have a unique max (no tie)
        assert len(spec.ties) == 1
