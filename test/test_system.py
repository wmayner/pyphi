import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi import config
from pyphi import exceptions
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.models import DirectedBipartition
from pyphi.models import Distinction
from pyphi.models import MaximallyIrreducibleCause
from pyphi.models import MaximallyIrreducibleEffect
from pyphi.models import RepertoireIrreducibilityAnalysis
from pyphi.system import System

from . import example_substrates


@config.override(validate_system_states=True)
def test_system_validation(s):
    # Wrong state length.
    with pytest.raises(ValueError):
        s = System(s.substrate, (0, 0), subset=s.node_indices)
    # Wrong state values.
    with pytest.raises(ValueError):
        s = System(s.substrate, (2, 0, 0), subset=s.node_indices)
    # Disallow impossible states at system level (we don't want to return a
    # phi-value associated with an impossible state).
    net = example_substrates.simple()
    with pytest.raises(exceptions.StateUnreachableError):
        s = System(net, (0, 1, 0), subset=s.node_indices)


def test_validate_partition_nodes_equal_system_nodes(s):
    assert s.node_indices == (0, 1, 2)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))  # A-ok
    System(s.substrate, s.state, subset=s.node_indices, partition=cut)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1,))  # missing node 2 in cut
    with pytest.raises(ValueError):
        System(s.substrate, s.state, subset=s.node_indices, partition=cut)

    cut = DirectedBipartition(Direction.EFFECT, (0,), (1, 2))  # missing node 2 in system
    with pytest.raises(ValueError):
        System(s.substrate, s.state, subset=(0, 1), partition=cut)


def test_empty_init(s):
    # Empty mechanism
    s = System(s.substrate, s.state, subset=())
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
        subset=s.node_indices,
        partition=DirectedBipartition(Direction.EFFECT, (0,), (1, 2)),
    )
    assert s.is_partitioned is True


def test_proper_state(subsys_n0n2, subsys_n1n2):
    # state == (1, 0, 0)
    assert subsys_n0n2.proper_state == (1, 0)
    assert subsys_n1n2.proper_state == (0, 0)


def test_apply_partition(s):
    cut = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    cut_s = s.apply_partition(cut)
    assert s.substrate == cut_s.substrate
    assert s.state == cut_s.state
    assert s.node_indices == cut_s.node_indices
    assert cut_s.effect_marginal == s.effect_marginal
    assert cut_s.cause_marginal == s.cause_marginal
    assert np.array_equal(cut_s.cm, cut.apply_cut(s.cm))


def test_apply_partition_shares_partition_independent_caches(s):
    cut = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    # Populate the parent's caches first.
    _ = s.cause_marginal
    _ = s.effect_marginal
    cut_s = s.apply_partition(cut)
    assert cut_s.cause_marginal is s.cause_marginal
    assert cut_s.effect_marginal is s.effect_marginal


def test_apply_partition_shares_caches_even_when_parent_is_cold(s):
    """Sharing must not depend on access order: a cut created before the
    parent's marginals were ever computed still shares them (otherwise
    every cut in a partition search recomputes its own inversion whenever
    the unpartitioned repertoires happen to come from a cross-system
    cache)."""
    cut = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    cut_s = s.apply_partition(cut)  # no prior marginal access on s
    assert cut_s.cause_marginal is s.cause_marginal
    assert cut_s.effect_marginal is s.effect_marginal


def test_apply_partition_partition_dependent_state_still_differs(s):
    cut_a = DirectedBipartition(Direction.EFFECT, (0, 1), (2,))
    cut_b = DirectedBipartition(Direction.EFFECT, (0, 2), (1,))
    sys_a = s.apply_partition(cut_a)
    sys_b = s.apply_partition(cut_b)
    # Genuine difference: the two cuts sever different edges.
    assert not np.array_equal(sys_a.cm, sys_b.cm)
    assert sys_a.nodes != sys_b.nodes


def test_partition_indices(s, subsys_n1n2):
    assert s.partition_indices == (0, 1, 2)
    assert subsys_n1n2.partition_indices == (1, 2)


def test_partitioned_mechanisms(s):
    assert list(s.partitioned_mechanisms) == []
    assert list(
        s.apply_partition(
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
    system = System(substrate, (0, 0, 0), subset=("B", "C"))
    assert system.node_indices == (1, 2)
    assert tuple(node.label for node in system.nodes) == ("B", "C")
    with config.override(repr_verbosity=0):
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
    assert s.null_distinction == Distinction(
        mechanism=(),
        cause=cause,
        effect=effect,
    )


def test_distinction_no_mechanism(s):
    assert s.distinction(()) == s.null_distinction


def test_distinction_nonexistent(s):
    assert not s.distinction((0, 2))


class TestIntrinsicInformationTies:
    """Test that intrinsic_specification() correctly tracks tied states."""

    def test_tied_states_are_stored(self):
        """intrinsic_specification() should populate .ties when states tie.

        AND-XOR at state (0,1): the cause GID is 0.5 for both purview states
        (1,0) and (0,1), creating a tie. Both should appear in .ties.
        """
        net = example_substrates.and_xor_substrate()
        sub = System(net, (0, 1))
        spec = sub.intrinsic_specification(
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
        spec = sub.intrinsic_specification(
            Direction.CAUSE,
            (0, 1),
            (0, 1),
            specification_measure=resolve_mechanism_measure(
                config.formalism.iit.specification_measure
            ),
        )
        ii_values = {float(t.intrinsic_specification) for t in spec.ties}
        assert len(ii_values) == 1, f"Tied states have different II: {ii_values}"

    def test_no_ties_when_unique_max(self):
        """When a single state uniquely maximizes II, ties should have length 1."""
        net = example_substrates.and_xor_substrate()
        sub = System(net, (0, 1))
        spec = sub.intrinsic_specification(
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


def test_proper_cause_marginal_kary_returns_factored_view() -> None:
    """proper_cause_marginal for a k=3 system returns per-system-unit factors
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
        proper = sys.proper_cause_marginal
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(sys.node_indices)
    for i in range(proper.n_nodes):
        assert proper.factor(i).shape[-1] == 3


def test_proper_cause_marginal_binary_matches_legacy_slice(s) -> None:
    """For binary substrates the per-system-unit on-probability slice of
    ``proper_cause_marginal`` matches the substrate cause TPM's on-probability
    slice for the same unit."""
    from pyphi.core.tpm.factored import FactoredTPM
    from pyphi.core.tpm.marginalization import CauseMarginals

    substrate_cause = s.cause_marginal
    assert isinstance(substrate_cause, CauseMarginals)
    proper = s.proper_cause_marginal
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(s.node_indices)
    for slot, node in enumerate(s.node_indices):
        proper_on = np.squeeze(proper.factor(slot)[..., 1])
        substrate_on = np.squeeze(substrate_cause.factor(node)[..., 1])
        assert np.allclose(proper_on, substrate_on, atol=1e-10)


def test_proper_effect_marginal_kary_returns_factored_view() -> None:
    """proper_effect_marginal for a k=3 system returns per-system-unit factors
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
        proper = sys.proper_effect_marginal
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(sys.node_indices)
    for i in range(proper.n_nodes):
        assert proper.factor(i).shape[-1] == 3


def test_proper_effect_marginal_binary_matches_legacy_on_probability(s) -> None:
    """For binary substrates the per-system-unit on-probability slice of
    ``proper_effect_marginal`` reproduces the legacy state-by-node forward
    on-probability column for that unit — the substrate forward factor's
    on-probability as a function of the full input state."""
    from pyphi.core.tpm.factored import FactoredTPM

    proper = s.proper_effect_marginal
    assert isinstance(proper, FactoredTPM)
    assert proper.n_nodes == len(s.node_indices)
    forward = s.substrate.factored_tpm
    for slot, node in enumerate(s.node_indices):
        proper_on = np.squeeze(proper.factor(slot)[..., 1])
        forward_on = np.squeeze(forward.factor(node)[..., 1])
        assert np.allclose(proper_on, forward_on, atol=1e-10)


def _asymmetric_substrate() -> Substrate:
    """A fully connected, asymmetric 3-unit binary substrate."""
    rng = np.random.default_rng(11)
    return Substrate(tpm=rng.random((8, 3)))


def test_proper_marginals_condition_on_external_indices_overlap() -> None:
    """When external_indices overlaps node_indices (the TransitionSystem
    pattern), the proper marginals condition on the external units — not the
    complement of the system — and drop only non-system external input dims."""
    from pyphi.core.tpm.marginalization import _effect_marginal_factored

    substrate = _asymmetric_substrate()
    state = (1, 0, 1)
    with config.override(validate_system_states=False):
        sys_ = System(
            substrate,
            state,
            subset=(0, 1),
            external_indices=(1, 2),
            background_conditioning="CONDITION_CURRENT_STATE",
            background_state=state,
        )
    proper = sys_.proper_effect_marginal
    # Expected: condition on the external units {1: 0, 2: 1}, then drop only
    # the non-system external axis {2}.
    expected = _effect_marginal_factored(substrate.factored_tpm, {1: 0, 2: 1})
    for slot, node in enumerate(sys_.node_indices):
        e = np.squeeze(expected.factor(node), axis=(2,))
        assert proper.factor(slot).shape == e.shape
        assert np.allclose(proper.factor(slot), e)
    # Consistent with effect_marginal restricted to the system factors.
    effect = sys_.effect_marginal
    for slot, node in enumerate(sys_.node_indices):
        assert np.allclose(
            proper.factor(slot), np.squeeze(effect.factor(node), axis=(2,))
        )
    # Cause side: same conditioning and squeeze semantics.
    proper_cause = sys_.proper_cause_marginal
    cause = sys_.cause_marginal
    for slot, node in enumerate(sys_.node_indices):
        e = np.squeeze(cause.factor(node), axis=(2,))
        assert proper_cause.factor(slot).shape == e.shape
        assert np.allclose(proper_cause.factor(slot), e)


def test_proper_marginals_empty_external_no_crash() -> None:
    """external_indices=() (the AC noise-background pattern) must not crash,
    and must not clamp the non-system unit at its observed state."""
    substrate = _asymmetric_substrate()
    state = (1, 0, 1)
    with config.override(validate_system_states=False):
        sys_ = System(
            substrate,
            state,
            subset=(0, 1),
            external_indices=(),
            background_conditioning="CONDITION_CURRENT_STATE",
        )
    proper_cause = sys_.proper_cause_marginal  # raised ValueError before the fix
    proper_effect = sys_.proper_effect_marginal
    for factored in (proper_cause, proper_effect):
        for slot in range(factored.n_nodes):
            assert factored.factor(slot).shape == (2, 2, 2)
    # With no external conditioning, the non-system unit is marginalized
    # uniformly — not clamped at its observed state.
    forward = substrate.factored_tpm
    for slot, node in enumerate(sys_.node_indices):
        expected = forward.factor(node).mean(axis=2)
        assert np.allclose(proper_effect.factor(slot), expected)
        clamped = forward.factor(node)[:, :, state[2], :]
        assert not np.allclose(proper_effect.factor(slot), clamped)


def test_transition_noise_background_proper_cause_marginal() -> None:
    """Public reachability of the empty-external crash:
    Transition(noise_background=True).cause_system.proper_cause_marginal."""
    from pyphi.actual import Transition

    substrate = _asymmetric_substrate()
    transition = Transition(
        substrate,
        before_state=(1, 0, 1),
        after_state=(1, 1, 1),
        cause_indices=(0,),
        effect_indices=(1,),
        noise_background=True,
    )
    proper = transition.cause_system.proper_cause_marginal  # raised before the fix
    assert proper.n_nodes == 2


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
        subset=(1, 2),
        external_indices=(),
    )
    assert overridden.external_indices == ()


def test_system_external_indices_rejects_out_of_range(s) -> None:
    """Indices must be in range(substrate.size)."""
    with pytest.raises(ValueError, match="out of range"):
        System(
            substrate=s.substrate,
            state=s.state,
            subset=(0, 1),
            external_indices=(99,),
        )


def test_system_external_indices_rejects_negative(s) -> None:
    """Negative indices are out of range."""
    with pytest.raises(ValueError, match="out of range"):
        System(
            substrate=s.substrate,
            state=s.state,
            subset=(0, 1),
            external_indices=(-1,),
        )


def test_system_external_indices_rejects_unsorted(s) -> None:
    """Override must be sorted."""
    with pytest.raises(ValueError, match="sorted"):
        System(
            substrate=s.substrate,
            state=s.state,
            subset=(0,),
            external_indices=(2, 1),
        )


def test_system_external_indices_rejects_duplicates(s) -> None:
    """Override must have unique entries."""
    with pytest.raises(ValueError, match="duplicate"):
        System(
            substrate=s.substrate,
            state=s.state,
            subset=(0,),
            external_indices=(1, 1),
        )


def test_system_external_indices_included_in_eq(s) -> None:
    """Two Systems differing only in external_indices are not equal."""
    s_default = System(
        substrate=s.substrate,
        state=s.state,
        subset=(0, 1),
    )
    s_override = System(
        substrate=s.substrate,
        state=s.state,
        subset=(0, 1),
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
        subset=(0, 1),
    )
    s_explicit = System(
        substrate=s.substrate,
        state=s.state,
        subset=(0, 1),
        external_indices=(2,),
    )
    assert s_implicit == s_explicit
    assert hash(s_implicit) == hash(s_explicit)


def test_system_external_indices_included_in_hash(s) -> None:
    """Hash distinguishes Systems with different external_indices."""
    s1 = System(
        substrate=s.substrate,
        state=s.state,
        subset=(0, 1),
    )
    s2 = System(
        substrate=s.substrate,
        state=s.state,
        subset=(0, 1),
        external_indices=(),
    )
    assert hash(s1) != hash(s2)


def test_system_external_indices_apply_partition_propagates(s) -> None:
    """apply_partition returns new System with external_indices preserved."""
    from pyphi.models.partitions import DirectedBipartition as _DBP

    base = System(
        substrate=s.substrate,
        state=s.state,
        subset=(0, 1),
        external_indices=(),
    )
    new_cut = _DBP(Direction.EFFECT, (0,), (1,), s.substrate.node_labels)
    cut = base.apply_partition(new_cut)
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
            subset=(0, 1, 2),
            external_indices=(1,),
        )
    assert overlapping.external_indices == (1,)


def test_iit3_ces_holds_resolved_distinctions():
    """Under IIT 3.0 the structure has no tied states to resolve, so its
    distinctions are ResolvedDistinctions, not the internal
    UnresolvedDistinctions."""
    import pyphi
    from pyphi.conf import presets
    from pyphi.models import CauseEffectStructure
    from pyphi.models import ResolvedDistinctions

    with config.override(**presets.iit3, progress_bars=False):
        ces = System(pyphi.examples.basic_substrate(), (1, 0, 0)).ces()
        analysis = pyphi.analyze(pyphi.examples.basic_substrate(), (1, 0, 0))
    assert isinstance(ces, CauseEffectStructure)
    assert isinstance(ces.distinctions, ResolvedDistinctions)
    assert analysis.ces == ces
    assert len(ces.distinctions) == 4


@pytest.mark.parametrize("method", ["sia", "ces"])
def test_formalism_argument_checks_the_state_under_that_formalism(method):
    """A System built under the default formalism and computed with
    formalism="IIT_3_0" must reject a state IIT 3.0 cannot analyze, as
    pyphi.analyze does, instead of returning a null result."""
    import pyphi
    from pyphi import examples
    from pyphi.conf import presets

    substrate = examples.marshall_2023_fig1_substrate()
    state = (0, 0, 0, 1)
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        pyphi.analyze(substrate, state, subset=(0, 2), formalism="IIT_3_0")
    with config.override(**presets.iit4_2026):
        system = System(substrate, state, subset=(0, 2))
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        getattr(system, method)(formalism="IIT_3_0")


def test_ces_rejects_results_computed_for_another_system():
    """A precomputed analysis or distinctions of other units or another state
    would silently change the cause-effect structure."""
    from pyphi import examples
    from pyphi.conf import presets

    basic = examples.basic_substrate()
    with config.override(**presets.iit4_2023):
        system = System(basic, (1, 0, 0))
        own = system.ces(sia=system.sia(), distinctions=system.distinctions())
        assert own.big_phi == pytest.approx(system.ces().big_phi)
        other_state = System(basic, (1, 1, 0))
        other_units = System(basic, (1, 0, 0), subset=(1, 2))
        with pytest.raises(ValueError, match="state"):
            system.ces(sia=other_state.sia())
        with pytest.raises(ValueError, match="units"):
            system.ces(sia=other_units.sia())
        with pytest.raises(ValueError, match="mechanism state"):
            system.ces(distinctions=other_state.distinctions())
        with pytest.raises(ValueError, match="outside"):
            other_units.ces(distinctions=system.distinctions())


def test_mic_and_mie_accept_purviews_in_any_order_or_as_labels():
    """An unsorted in-system purview returned a null MICE with φ = 0."""
    from pyphi import examples
    from pyphi.conf import presets

    with config.override(**presets.iit4_2023):
        system = System(examples.basic_substrate(), (1, 0, 0))
        for method in (system.mic, system.mie):
            expected = method((0,), purviews=[(1, 2)])
            for spelled in ([(2, 1)], [("C", "B")]):
                found = method(("A",), purviews=spelled)
                assert found.purview == expected.purview
                assert found.phi == pytest.approx(expected.phi)
        found = system.find_mice(Direction.CAUSE, (0,), purviews=[(2, 1)])
        assert found.phi == pytest.approx(system.mic((0,), purviews=[(1, 2)]).phi)
        assert system.distinction(("A",)) == system.distinction((0,))


@pytest.mark.parametrize(
    ("mechanism", "purviews", "match"),
    [
        ((0,), [(5,)], "out of range"),
        ((0, 0), None, "more than once"),
        (("Z",), None, "unknown node labels"),
    ],
)
def test_mic_rejects_unknown_units_with_a_clear_message(mechanism, purviews, match):
    from pyphi import examples

    system = System(examples.basic_substrate(), (1, 0, 0))
    with pytest.raises(ValueError, match=match):
        system.mic(mechanism, purviews=purviews)


def test_mip_queries_accept_a_measure_name():
    import pyphi
    from pyphi import examples

    with config.override(**pyphi.iit4_2026):
        system = System(examples.basic_substrate(), (1, 0, 0))
        named = system.cause_mip(
            (0,), (1,), mechanism_measure="GENERALIZED_INTRINSIC_DIFFERENCE"
        )
        assert named.phi == system.cause_mip((0,), (1,)).phi


@pytest.mark.parametrize(
    ("query", "error", "match"),
    [
        (
            lambda s: s.cause_mip((0,), (1,), mechanism_measure="KLD"),
            "MeasureNotCompatibleError",
            "not compatible",
        ),
        (
            lambda s: s.find_mip(Direction.CAUSE, (0,), (1,), state=(0,)),
            "TypeError",
            "specified state",
        ),
        (
            lambda s: s.find_mice("cause", (0,)),
            "ValueError",
            "Direction.CAUSE or Direction.EFFECT",
        ),
        (
            lambda s: s.sia(partition_scheme="directed_bipartition"),
            "ValueError",
            "unknown partition_scheme",
        ),
    ],
    ids=["incompatible measure", "state", "direction", "partition scheme"],
)
def test_mip_query_options_are_checked(query, error, match):
    import pyphi
    from pyphi import examples

    with config.override(**pyphi.iit4_2026):
        system = System(examples.basic_substrate(), (1, 0, 0))
        with pytest.raises(Exception, match=match) as raised:
            query(system)
    assert type(raised.value).__name__ == error


@pytest.mark.parametrize("value", [0, 1, np.int64(1)])
def test_a_direction_may_be_given_by_value(value):
    """Direction(value) accepts 0 and 1, and so do the mechanism queries."""
    import pyphi
    from pyphi import examples

    with config.override(**pyphi.iit4_2026):
        system = System(examples.basic_substrate(), (1, 0, 0))
        by_value = system.find_mip(value, (0,), (1,))
        assert by_value == system.find_mip(Direction(value), (0,), (1,))
        assert system.find_mice(value, (0,)) == system.find_mice(Direction(value), (0,))


@pytest.mark.parametrize("value", [2, True, "cause", None])
def test_a_direction_other_than_cause_or_effect_is_rejected(value):
    import pyphi
    from pyphi import examples

    with config.override(**pyphi.iit4_2026):
        system = System(examples.basic_substrate(), (1, 0, 0))
        with pytest.raises(ValueError, match=r"Direction\.CAUSE or Direction\.EFFECT"):
            system.find_mip(value, (0,), (1,))


@pytest.mark.parametrize("method", ["sia", "ces"])
def test_an_active_override_checks_the_state_under_that_formalism(method):
    """Computing under config.override(**iit3) a System built under IIT 4.0
    must reject a state IIT 3.0 cannot analyze, as formalism= does."""
    import pyphi
    from pyphi import examples
    from pyphi.conf import presets

    substrate = examples.marshall_2023_fig1_substrate()
    state = (0, 0, 0, 1)
    with config.override(**presets.iit4_2026):
        system = System(substrate, state, subset=(0, 2))
    with (
        config.override(**pyphi.iit3),
        pytest.raises(exceptions.StateUnreachableForwardsError),
    ):
        getattr(system, method)()


def test_a_system_built_without_state_validation_is_not_rechecked():
    """validate_system_states=False at construction opts the System out of
    the check, whether or not the override is still active at compute time."""
    import pyphi
    from pyphi import examples

    substrate = examples.marshall_2023_fig1_substrate()
    state = (0, 0, 0, 1)
    with config.override(validate_system_states=False):
        system = System(substrate, state, subset=(0, 2))
    with config.override(**pyphi.iit3):
        assert system.sia().phi == 0
    with config.override(**pyphi.iit3, validate_system_states=False):
        assert system.sia().phi == 0
