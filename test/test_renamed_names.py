"""Names removed in 2.0 raise errors that give their replacement, and the
arguments renamed for one vocabulary keep their meaning."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi import models


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("SystemIrreducibilityAnalysis", "pyphi.formalism.iit4"),
        ("Bipartition", "JointBipartition"),
        ("Tripartition", "JointTripartition"),
        ("KPartition", "JointPartition"),
        ("KCut", "DirectedJointPartition"),
        ("ActualCut", "DirectedBipartition"),
        ("PhiStructure", "CauseEffectStructure"),
    ],
)
def test_removed_model_names_give_their_replacement(name, replacement):
    with pytest.raises(AttributeError, match=replacement):
        getattr(models, name)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [
        ("cause_tpm", "cause_marginal"),
        ("effect_tpm", "effect_marginal"),
        ("proper_cause_tpm", "proper_cause_marginal"),
        ("proper_effect_tpm", "proper_effect_marginal"),
        ("cut", "partition"),
    ],
)
def test_removed_system_attributes_give_their_replacement(name, replacement):
    with pytest.raises(AttributeError, match=replacement):
        getattr(examples.basic_system(), name)


@pytest.mark.parametrize(
    ("name", "replacement"),
    [("apply_cut", "apply_partition"), ("cut", "partition"), ("network", "substrate")],
)
def test_removed_transition_system_attributes_give_their_replacement(name, replacement):
    system = examples.prevention_transition().cause_system
    with pytest.raises(AttributeError, match=f"`{replacement}`"):
        getattr(system, name)


@pytest.mark.parametrize(
    ("module", "name", "replacement"),
    [
        (pyphi, "distance", "pyphi.measures.distribution"),
        (pyphi.utils, "eq", "pyphi.numerics.eq"),
        (pyphi.utils, "is_positive", "pyphi.numerics.is_positive"),
        (pyphi.utils, "is_nonpositive", "pyphi.numerics.is_nonpositive"),
    ],
)
def test_removed_1x_helpers_give_their_replacement(module, name, replacement):
    with pytest.raises(AttributeError, match=replacement):
        getattr(module, name)


def test_an_unknown_utils_name_gets_the_ordinary_error():
    with pytest.raises(AttributeError) as raised:
        pyphi.utils.no_such_name  # noqa: B018
    assert str(raised.value) == "module 'pyphi.utils' has no attribute 'no_such_name'"


def test_iit4_sia_cut_gives_partition():
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.analyze(
            examples.iit4_2023_fig1a_substrate(), (1, 0, 0), compute="sia"
        )
    with pytest.raises(AttributeError, match="partition"):
        _ = sia.cut


def test_distinctions_hold_no_concepts_attribute():
    with pyphi.config.override(**pyphi.iit4_2026):
        distinctions = pyphi.System(
            examples.iit4_2023_fig1a_substrate(), (1, 0, 0)
        ).distinctions()
    rebuilt = type(distinctions)(distinctions=tuple(distinctions))
    assert rebuilt == distinctions
    with pytest.raises(AttributeError, match="sequence of distinctions"):
        _ = distinctions.concepts
    restored = pyphi.serialize.loads(pyphi.serialize.dumps(distinctions))
    assert restored == distinctions


def test_reachable_systems_takes_state_then_subset():
    from pyphi.substrate import reachable_systems

    substrate = examples.basic_substrate()
    systems = list(reachable_systems(substrate, (1, 0, 0), subset=(0, 1)))
    assert {s.node_indices for s in systems} <= {(0, 1), (0,), (1,)}
    assert all(s.state == (1, 0, 0) for s in systems)
    with pytest.raises(TypeError):
        list(reachable_systems(substrate, (0, 1), (1, 0, 0)))


def test_matching_index_arguments_are_keyword_only():
    from pyphi.matching import PerceptualSystem
    from pyphi.matching import TriggeredTPM
    from pyphi.matching import build_triggered_tpm

    substrate = examples.basic_substrate()
    with pytest.raises(TypeError):
        PerceptualSystem(substrate, (1,), (0,))
    with pytest.raises(TypeError):
        build_triggered_tpm(substrate, (0,), (1,), tau=1, tau_clamp=1)
    with pytest.raises(TypeError):
        TriggeredTPM(np.ones((2, 2)), (0,), (1,), substrate.node_labels)


def test_full_analysis_estimate_reports_none_like_its_argument():
    substrate = examples.basic_substrate()
    estimate = pyphi.estimate_analysis(substrate, (1, 0, 0))
    assert estimate.compute is None
    again = pyphi.estimate_analysis(substrate, (1, 0, 0), compute=estimate.compute)
    assert again == estimate
    assert pyphi.estimate_analysis(substrate, compute="sia").compute == "sia"


def test_dynamics_seed_and_rng_are_keyword_only_and_exclusive():
    from pyphi import dynamics

    tpm = np.array([[[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5]]])
    with pytest.raises(TypeError):
        dynamics.simulate(tpm, (0, 0), 3, None, None, 0)  # type: ignore[misc]
    with pytest.raises(TypeError, match="at most one"):
        dynamics.simulate(tpm, timesteps=3, seed=0, rng=np.random.default_rng(0))
    with pytest.raises(TypeError, match="at most one"):
        dynamics.mean_dynamics(
            tpm, repetitions=1, timesteps=3, seed=0, rng=np.random.default_rng(0)
        )
    with pytest.raises(TypeError):
        dynamics.mean_dynamics(tpm, 1, 0)  # type: ignore[misc]
    first = dynamics.mean_dynamics(tpm, repetitions=2, timesteps=3, seed=1)
    second = dynamics.mean_dynamics(tpm, repetitions=2, timesteps=3, seed=1)
    assert np.array_equal(first, second)
