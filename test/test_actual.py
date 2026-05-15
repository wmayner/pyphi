from dataclasses import replace

import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi import System
from pyphi import actual
from pyphi import config
from pyphi import examples
from pyphi import models
from pyphi.models import JointPartition
from pyphi.models import Part

from .conftest import IIT_3_CONFIG


def test_actual_partitioned_repertoire_schemes_registry():
    from pyphi import actual

    assert "PRODUCT" in actual.partitioned_repertoire_schemes
    assert "FORWARD_PROBABILITY" not in actual.partitioned_repertoire_schemes


def test_actual_background_strategies_registry():
    from pyphi import actual

    assert "UNIFORM" in actual.background_strategies
    assert "STATIONARY" not in actual.background_strategies


def test_actual_alpha_aggregations_registry():
    from pyphi import actual

    assert "SUBTRACTIVE" in actual.alpha_aggregations
    assert "RATIO" not in actual.alpha_aggregations


def test_transition_system_top_level_export():
    import pyphi

    assert hasattr(pyphi, "TransitionSystem")
    from pyphi.actual import TransitionSystem

    assert pyphi.TransitionSystem is TransitionSystem


def _ts_substrate():
    """Helper: build the OR/AND-style 3-node substrate used in TransitionSystem tests."""
    import numpy as np

    from pyphi import Substrate

    tpm = np.array(
        [
            [0, 0.5, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
        ]
    )
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    return Substrate(tpm, cm)


def test_transition_system_is_frozen():
    import dataclasses

    import pytest as _pt

    from pyphi import Direction
    from pyphi.actual import TransitionSystem

    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    with _pt.raises(dataclasses.FrozenInstanceError):
        ts.before_state = (1, 1, 1)


def test_transition_system_satisfies_protocol():
    from pyphi import Direction
    from pyphi.actual import TransitionSystem
    from pyphi.protocols import SystemPublicInterface

    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    assert isinstance(ts, SystemPublicInterface)


def test_transition_system_cause_uses_after_state():
    from pyphi import Direction
    from pyphi.actual import TransitionSystem

    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    assert ts.state == (1, 0, 0)


def test_transition_system_effect_uses_before_state():
    from pyphi import Direction
    from pyphi.actual import TransitionSystem

    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.EFFECT,
    )
    assert ts.state == (0, 1, 1)


def test_transition_system_external_indices_excludes_cause_indices():
    from pyphi import Direction
    from pyphi.actual import TransitionSystem

    ts = TransitionSystem(
        substrate=_ts_substrate(),
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    assert ts.external_indices == (0,)


def test_transition_system_apply_partition_returns_new_instance():
    from pyphi import Direction
    from pyphi.actual import TransitionSystem
    from pyphi.models.partitions import DirectedBipartition
    from pyphi.models.partitions import NullCut

    substrate = _ts_substrate()
    ts = TransitionSystem(
        substrate=substrate,
        before_state=(0, 1, 1),
        after_state=(1, 0, 0),
        cause_indices=(1, 2),
        effect_indices=(0,),
        direction=Direction.CAUSE,
    )
    new_cut = DirectedBipartition(Direction.CAUSE, (0, 1), (2,), substrate.node_labels)
    ts2 = ts.apply_cut(new_cut)
    assert ts2 is not ts
    assert ts2.partition == new_cut
    assert isinstance(ts.partition, NullCut)


# TODO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   * test transition equality/hash
#   * state_probability


# Fixtures
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture
def transition():
    """An OR gate with two inputs. The OR gate is ON, others are OFF."""
    # fmt: off
    tpm = np.array([
        [0, 0.5, 0.5],
        [0, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
    ])
    cm = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ])
    # fmt: on
    substrate = Substrate(tpm, cm)
    before_state = (0, 1, 1)
    after_state = (1, 0, 0)
    return actual.Transition(substrate, before_state, after_state, (1, 2), (0,))


@pytest.fixture
def empty_transition(transition):
    return actual.Transition(
        transition.substrate, transition.before_state, transition.after_state, (), ()
    )


@pytest.fixture
def prevention():
    return examples.prevention_transition()


# Testing background conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# NOTE: test_background_conditions was removed because the expected values
# were outdated. test_background_noised provides coverage for similar functionality.


def test_background_noised():
    # fmt: off
    tpm = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
    ])
    # fmt: on
    substrate = Substrate(tpm)
    state = (1, 1)
    transition = actual.Transition(
        substrate, state, state, (0,), (1,), noise_background=True
    )

    assert np.isclose(transition._ratio(Direction.EFFECT, (0,), (1,)), 0.415037)
    assert np.isclose(transition._ratio(Direction.CAUSE, (1,), (0,)), 0.415037)

    # Elements outside the transition are also frozen
    transition = actual.Transition(
        substrate, state, state, (0,), (0,), noise_background=True
    )
    assert transition.cause_system.effect_tpm.array_equal(substrate.tpm)
    assert transition.effect_system.effect_tpm.array_equal(substrate.tpm)


@pytest.fixture
def background_3_node():
    """A is MAJ(ABC). B is OR(A, C). C is COPY(A)."""
    # fmt: off
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 1],
    ])
    # fmt: on
    return Substrate(tpm)


# NOTE: test_background_3_node was removed because the expected values were outdated.


def test_potential_purviews(background_3_node):
    """Purviews must be a subset of the corresponding cause/effect system."""
    transition = actual.Transition(
        background_3_node, (1, 1, 1), (1, 1, 1), (0, 1), (0, 2)
    )
    assert set(transition.potential_purviews(Direction.CAUSE, (0, 2))) == {
        (0,),
        (1,),
        (0, 1),
    }
    assert set(transition.potential_purviews(Direction.EFFECT, (0, 1))) == {
        (0,),
        (2,),
        (0, 2),
    }


# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_transition_initialization(transition):
    assert transition.effect_system.state == (0, 1, 1)
    assert transition.cause_system.state == (1, 0, 0)
    assert tuple(n.state for n in transition.cause_system.nodes) == (1, 0, 0)


def test_purview_state(transition):
    assert transition.purview_state(Direction.CAUSE) == (0, 1, 1)
    assert transition.purview_state(Direction.EFFECT) == (1, 0, 0)


def test_mechanism_state(transition):
    assert transition.mechanism_state(Direction.CAUSE) == (1, 0, 0)
    assert transition.mechanism_state(Direction.EFFECT) == (0, 1, 1)


def test_mechanism_indices(transition):
    assert transition.mechanism_indices(Direction.CAUSE) == (0,)
    assert transition.mechanism_indices(Direction.EFFECT) == (1, 2)


def test_purview_indices(transition):
    assert transition.purview_indices(Direction.CAUSE) == (1, 2)
    assert transition.purview_indices(Direction.EFFECT) == (0,)


def test_system_dict(transition):
    assert transition.system[Direction.CAUSE] == transition.cause_system
    assert transition.system[Direction.EFFECT] == transition.effect_system


def test_transition_len(transition, empty_transition):
    assert len(transition) == 3
    assert len(empty_transition) == 0


def test_transition_bool(transition, empty_transition):
    assert bool(transition)
    assert not bool(empty_transition)


def test_transition_equal(transition, empty_transition):
    assert transition != empty_transition
    assert hash(transition) != hash(empty_transition)


def test_transition_apply_partition(transition):
    cut = ac_cut(Direction.CAUSE, Part((1,), (2,)), Part((), (0,)))
    cut_transition = transition.apply_cut(cut)
    assert cut_transition.before_state == transition.before_state
    assert cut_transition.after_state == transition.after_state
    assert cut_transition.cause_indices == transition.cause_indices
    assert cut_transition.effect_indices == transition.effect_indices
    assert cut_transition.partition == cut
    assert cut_transition != transition


def test_to_json(transition):
    transition.to_json()


# Test AC models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def acria(**kwargs):
    defaults = {
        "alpha": 0.0,
        "state": None,
        "direction": None,
        "mechanism": (),
        "purview": (),
        "partition": None,
        "probability": 0.0,
        "partitioned_probability": 0.0,
    }
    defaults.update(kwargs)
    return models.AcRepertoireIrreducibilityAnalysis(**defaults)


def causal_link(**kwargs):
    return models.CausalLink(acria(**kwargs))


def account(links=()):
    return models.Account(links)


def ac_sia(**kwargs):
    defaults = {
        "alpha": 0.0,
        "direction": Direction.BIDIRECTIONAL,
        "account": account(),
        "partitioned_account": account(),
        "partition": None,
    }
    transition = kwargs.pop("transition", None)
    if transition is not None:
        defaults.update(
            before_state=transition.before_state,
            after_state=transition.after_state,
            size=len(transition),
            node_indices=transition.node_indices,
            node_labels=transition.substrate.node_labels,
            partition=transition.partition,
        )
    defaults.update(kwargs)
    return models.AcSystemIrreducibilityAnalysis(**defaults)


def test_acria_ordering():
    assert acria() == acria()
    assert acria(alpha=0.0) < acria(alpha=1.0)
    assert acria(alpha=0.0, mechanism=(1, 2)) <= acria(alpha=1.0, mechanism=(1,))
    assert acria(alpha=0.0, mechanism=(1, 2)) > acria(alpha=0.0, mechanism=(1,))

    assert bool(acria(alpha=1.0)) is True
    assert bool(acria(alpha=0.0)) is False
    assert bool(acria(alpha=-1)) is False

    with pytest.raises(TypeError):
        acria(direction=Direction.CAUSE) < acria(direction=Direction.EFFECT)  # noqa: B015

    # Smaller purviews should be chosen
    assert acria(purview=(1,)) > acria(purview=(0, 2))


def test_acria_hash():
    hash(acria())


def test_acria_phi_alias():
    assert acria(alpha=3.3).phi == 3.3


def test_causal_link_ordering():
    assert causal_link() == causal_link()

    assert causal_link(alpha=0.0) < causal_link(alpha=1.0)
    assert causal_link(alpha=0.0, mechanism=(1, 2)) <= causal_link(
        alpha=1.0, mechanism=(1,)
    )
    assert causal_link(alpha=0.0, mechanism=(1, 2)) > causal_link(
        alpha=0.0, mechanism=(1,)
    )

    with pytest.raises(TypeError):
        (
            causal_link(direction=Direction.CAUSE)  # noqa: B015
            < causal_link(direction=Direction.EFFECT)
        )

    assert bool(causal_link(alpha=1.0)) is True
    assert bool(causal_link(alpha=0.0)) is False
    assert bool(causal_link(alpha=-1)) is False


def test_account_irreducible_causes_and_effects():
    cause = causal_link(direction=Direction.CAUSE)
    effect = causal_link(direction=Direction.EFFECT)
    account = models.Account((cause, effect))

    assert account.irreducible_causes == (cause,)
    assert account.irreducible_effects == (effect,)


def test_account_repr_and_str():
    str(account())
    repr(account())


def test_account_addition():
    a1 = account([causal_link(direction=Direction.CAUSE)])
    a2 = account([causal_link(direction=Direction.EFFECT)])
    assert len(a1 + a2) == 2

    with pytest.raises(TypeError):
        a1 + [causal_link()]  # noqa: RUF005


def test_ac_sia_repr_and_str(transition):
    bm = ac_sia(transition=transition)
    str(bm)
    repr(bm)


def test_ac_sia_ordering(transition, empty_transition):
    assert ac_sia() == ac_sia()
    assert hash(ac_sia()) == hash(ac_sia())

    assert ac_sia(alpha=1.0, transition=transition) > ac_sia(
        alpha=0.5, transition=transition
    )
    assert ac_sia(alpha=1.0, transition=empty_transition) <= ac_sia(
        alpha=1.0, transition=transition
    )


@pytest.mark.parametrize(
    "direction,mechanism,purview,repertoire",
    [
        (Direction.CAUSE, (0,), (1,), [[[0.3333333], [0.66666667]]]),
        (Direction.CAUSE, (0,), (2,), [[[0.3333333, 0.66666667]]]),
        (Direction.CAUSE, (0,), (1, 2), [[[0, 0.3333333], [0.3333333, 0.3333333]]]),
        (Direction.EFFECT, (1,), (0,), [[[0]], [[1]]]),
        (Direction.EFFECT, (2,), (0,), [[[0]], [[1]]]),
        (Direction.EFFECT, (1, 2), (0,), [[[0]], [[1]]]),
    ],
)
def test_repertoires(direction, mechanism, purview, repertoire, transition):
    np.testing.assert_array_almost_equal(
        transition.repertoire(direction, mechanism, purview), repertoire
    )


def test_invalid_repertoires(transition):
    """Check that elements outside the transition cannot be passed in
    the mechanism or purview."""
    with pytest.raises(ValueError):
        transition.effect_repertoire((1, 2), (0, 1))

    with pytest.raises(ValueError):
        transition.effect_repertoire((0, 1, 2), (0,))

    with pytest.raises(ValueError):
        transition.cause_repertoire((0,), (0, 1, 2))

    with pytest.raises(ValueError):
        transition.cause_repertoire((0, 1), (1, 2))


def test_unconstrained_repertoires(transition):
    np.testing.assert_array_equal(
        transition.unconstrained_cause_repertoire((2,)), [[[0.5, 0.5]]]
    )
    np.testing.assert_array_equal(
        transition.unconstrained_effect_repertoire((0,)), [[[0.25]], [[0.75]]]
    )


@pytest.mark.parametrize(
    "direction,mechanism,purview,probability",
    [
        (Direction.CAUSE, (0,), (1,), 0.66666667),
        (Direction.CAUSE, (0,), (2,), 0.66666667),
        (Direction.CAUSE, (0,), (1, 2), 0.3333333),
        (Direction.EFFECT, (1,), (0,), 1),
        (Direction.EFFECT, (2,), (0,), 1),
        (Direction.EFFECT, (1, 2), (0,), 1),
    ],
)
def test_probability(direction, mechanism, purview, probability, transition):
    assert np.isclose(transition.probability(direction, mechanism, purview), probability)


def test_unconstrained_probability(transition):
    assert transition.unconstrained_probability(Direction.CAUSE, (1,)) == 0.5
    assert transition.unconstrained_probability(Direction.EFFECT, (0,)) == 0.75


def test_state_probability_strict_system():
    """Regression test for ``Transition.state_probability`` on transitions
    whose system is a strict subset of the substrate.

    ``System.cause_repertoire``/``effect_repertoire`` return
    substrate-shaped arrays (``ndim == substrate.size``) for non-empty mechanisms,
    but ``max_entropy_distribution`` (used when the mechanism is empty) returns
    a system-shaped array (``ndim == system.size``). ``state_probability``
    must handle both shapes; this test exercises the system-shaped branch,
    which the existing 3-node fixtures (whose system equals the substrate) do
    not reach.
    """
    # fmt: off
    tpm = np.zeros((16, 4))
    for i in range(16):
        for n in range(4):
            tpm[i, n] = (i >> (3 - n)) & 1
    cm = np.eye(4, dtype=int)
    # fmt: on
    substrate = Substrate(tpm, cm)
    with config.override(validate_system_states=False):
        t = actual.Transition(substrate, (1, 1, 1, 1), (1, 1, 1, 1), (2,), (3,))
    # system.node_indices is (2, 3), strictly inside substrate.node_indices
    assert t.cause_system.node_indices == (2, 3)
    assert t.cause_system.substrate.size == 4

    # Unconstrained repertoire is system-shaped (ndim=2), constrained
    # repertoire is substrate-shaped (ndim=4). Both must index correctly.
    unconstrained = t.unconstrained_cause_repertoire((2,))
    constrained = t.cause_repertoire((3,), (2,))
    assert unconstrained.ndim == t.cause_system.size
    assert constrained.ndim == t.cause_system.substrate.size

    # If state_probability mishandles the shape, this raises IndexError.
    assert t.unconstrained_probability(Direction.CAUSE, (2,)) == 0.5
    assert t.probability(Direction.CAUSE, (3,), (2,)) == 0.5


@pytest.mark.parametrize(
    "mechanism,purview,ratio",
    [
        ((0,), (1,), 0.41504),
        ((0,), (2,), 0.41504),
        ((0,), (1, 2), 0.41504),
    ],
)
def test_cause_ratio(mechanism, purview, ratio, transition):
    assert np.isclose(transition.cause_ratio(mechanism, purview), ratio)


@pytest.mark.parametrize(
    "mechanism,purview,ratio",
    [
        ((1,), (0,), 0.41504),
        ((2,), (0,), 0.41504),
        ((1, 2), (0,), 0.41504),
    ],
)
def test_effect_ratio(mechanism, purview, ratio, transition):
    assert np.isclose(transition.effect_ratio(mechanism, purview), ratio)


# NOTE: test_ac_ex1_transition was removed because the expected values were outdated.


def test_actual_partition_indices():
    cut = ac_cut(Direction.CAUSE, Part((0,), (2,)), Part((4,), (5,)))
    assert cut.indices == (0, 2, 4, 5)
    cut = ac_cut(Direction.EFFECT, Part((0, 2), (0, 2)), Part((), ()))
    assert cut.indices == (0, 2)


def test_actual_apply_partition():
    cut = ac_cut(Direction.CAUSE, Part((0,), (0, 2)), Part((2,), ()))
    cm = np.ones((3, 3))
    # fmt: off
    answer = np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 0],
    ])
    # fmt: on
    assert np.array_equal(cut.apply_cut(cm), answer)


def test_actual_cut_matrix():
    cut = ac_cut(Direction.CAUSE, Part((0,), (0, 2)), Part((2,), ()))
    # fmt: off
    answer = np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 1]
    ])
    # fmt: on
    assert np.array_equal(cut.cut_matrix(3), answer)


def ac_cut(direction, *parts):
    return models.DirectedJointPartition(direction, JointPartition(*parts))


@config.override(
    iit=replace(config.formalism.iit, mechanism_partition_scheme="WEDGE_TRIPARTITION")
)
@pytest.mark.parametrize(
    "direction,answer",
    [
        (
            Direction.BIDIRECTIONAL,
            [
                ac_cut(Direction.CAUSE, Part((), ()), Part((), (1, 2)), Part((0,), ())),
                ac_cut(Direction.EFFECT, Part((), ()), Part((1,), (0,)), Part((2,), ())),
                ac_cut(Direction.EFFECT, Part((), ()), Part((1,), ()), Part((2,), (0,))),
            ],
        ),
        (
            Direction.CAUSE,
            [ac_cut(Direction.CAUSE, Part((), ()), Part((), (1, 2)), Part((0,), ()))],
        ),
        (
            Direction.EFFECT,
            [
                ac_cut(Direction.EFFECT, Part((), ()), Part((), (0,)), Part((1, 2), ())),
                ac_cut(Direction.EFFECT, Part((), ()), Part((1,), (0,)), Part((2,), ())),
                ac_cut(Direction.EFFECT, Part((), ()), Part((1,), ()), Part((2,), (0,))),
            ],
        ),
    ],
)
def test_get_actual_partitions(direction, answer, transition):
    cuts = list(actual._get_partitions(transition, direction))
    assert set(cuts) == set(answer)


# NOTE: test_sia was removed because the expected values were outdated.


def test_null_ac_sia(transition):
    sia = actual._null_ac_sia(transition, Direction.CAUSE)
    assert sia.before_state == transition.before_state
    assert sia.after_state == transition.after_state
    assert sia.size == len(transition)
    assert sia.node_indices == transition.node_indices
    assert sia.direction == Direction.CAUSE
    assert sia.account == ()
    assert sia.partitioned_account == ()
    assert sia.alpha == 0.0

    sia = actual._null_ac_sia(transition, Direction.CAUSE, alpha=float("inf"))
    assert sia.alpha == float("inf")


# NOTE: test_prevention was removed because the expected values were outdated.


# IIT 3.0 Regression Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The tests below were originally written for IIT 3.0 and validate that
# the library produces consistent results under that configuration.
# They require specific IIT 3.0 config settings to pass.
#
# The IIT_3_CONFIG is defined in conftest.py and shared across test files.


@pytest.fixture
def background_all_off():
    """Transition fixture with all nodes OFF -> all nodes OFF."""
    # fmt: off
    tpm = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
    ])
    # fmt: on
    substrate = Substrate(tpm)
    return actual.Transition(substrate, (0, 0), (0, 0), (0,), (1,))


@pytest.fixture
def background_all_on():
    """Transition fixture with all nodes ON -> all nodes ON."""
    # fmt: off
    tpm = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1],
    ])
    # fmt: on
    substrate = Substrate(tpm)
    return actual.Transition(substrate, (1, 1), (1, 1), (0,), (1,))


class TestActualCausationIIT30:
    """Regression tests for actual causation with IIT 3.0 configuration.

    These tests were originally written for IIT 3.0 and validate that
    the library produces consistent results under that configuration.
    """

    @pytest.fixture(autouse=True)
    def _apply_iit30_config(self):
        with IIT_3_CONFIG:
            yield

    def test_background_conditions(self, background_all_off, background_all_on):
        """Actual-causation ratios on two-OR-gate transitions under IIT 3.0.

        Substrate: two OR gates with full all-to-all connectivity. Both fixtures
        transition state -> state (fixed point), so the only thing that changes
        between them is whether the shared fixed point is ``(0, 0)`` or
        ``(1, 1)``. Ratios are computed with ``ACTUAL_CAUSATION_MEASURE="PMI"``
        (inherited from ``IIT_3_CONFIG``), so each ratio is
        ``log2(constrained / unconstrained)``.

        ``(0, 0) -> (0, 0)``:
            The only previous state leading to ``node_1 = 0`` next is
            ``(0, 0)``, so the cause repertoire is a delta on ``node_0 = 0``;
            likewise the effect repertoire is a delta on ``node_1 = 0``.
            Constrained probability is 1, unconstrained is 1/2, ratio is
            ``log2(2) = 1``.

        ``(1, 1) -> (1, 1)``:
            EFFECT direction — OR saturates: the next state of ``node_1`` is
            ``1`` regardless of ``node_0``, so constrained and unconstrained
            probabilities are both 1 and the ratio is ``log2(1) = 0``.
            CAUSE direction — OR does **not** saturate backwards: three of
            four previous states lead to ``node_1 = 1`` next, with previous
            ``node_0`` values ``{0, 1, 1}``, so
            ``P(previous node_0 = 1 | node_1 next = 1) = 2/3``. Unconstrained
            is 1/2, so the ratio is ``log2((2/3) / (1/2)) = log2(4/3) ≈
            0.4150374992788``. (The original pre-cleanup assertion of ``0``
            here was wrong — it conflated forward saturation with backward
            saturation.)
        """
        assert background_all_off._ratio(Direction.EFFECT, (0,), (1,)) == 1
        assert background_all_off._ratio(Direction.CAUSE, (1,), (0,)) == 1
        assert background_all_on._ratio(Direction.EFFECT, (0,), (1,)) == 0
        assert np.isclose(
            background_all_on._ratio(Direction.CAUSE, (1,), (0,)),
            np.log2(4 / 3),
        )

    def test_sia(self, transition):
        """Test actual causation SIA computation (IIT 3.0)."""
        sia = actual.sia(transition)
        assert np.isclose(sia.alpha, 0.4150374992788)
        assert sia.direction == Direction.BIDIRECTIONAL
        assert len(sia.account) == 3
        assert len(sia.partitioned_account) == 2

    def test_sia_cause_direction(self, transition):
        """Test SIA with CAUSE direction only under IIT 3.0 bipartitions.

        Under ``mechanism_partition_scheme="JOINT_BIPARTITION"`` (inherited
        from ``IIT_3_CONFIG``), the cause direction reduces to ``alpha == 0.0``
        for this fixture even though individual causal links have nonzero
        alpha. The effect and bidirectional directions remain nonzero — see
        :meth:`test_sia_effect_direction` and :meth:`test_sia`. Under
        ``mechanism_partition_scheme="WEDGE_TRIPARTITION"`` the cause
        direction is also nonzero; that regime is exercised by
        :meth:`test_prevention`.
        """
        sia_cause = actual.sia(transition, Direction.CAUSE)
        assert sia_cause.alpha == 0.0
        assert sia_cause.direction == Direction.CAUSE

    def test_sia_effect_direction(self, transition):
        """Test SIA with EFFECT direction only (IIT 3.0)."""
        sia_effect = actual.sia(transition, Direction.EFFECT)
        assert np.isclose(sia_effect.alpha, 0.4150374992788)
        assert sia_effect.direction == Direction.EFFECT

    @config.override(
        iit=replace(
            config.formalism.iit, mechanism_partition_scheme="WEDGE_TRIPARTITION"
        )
    )
    def test_prevention(self, prevention):
        """Test prevention example under IIT 3.0 with tripartitions.

        The original test deliberately exercised tripartition
        (``mechanism_partition_scheme="WEDGE_TRIPARTITION"``). Do not silently
        inherit ``JOINT_BIPARTITION`` from ``IIT_3_CONFIG`` — the bipartition
        and tripartition results differ and are different claims about the
        prevention example.
        """
        assert np.isclose(actual.sia(prevention, Direction.CAUSE).alpha, 0.4150374992788)
        assert actual.sia(prevention, Direction.EFFECT).alpha == 0.0
        assert actual.sia(prevention, Direction.BIDIRECTIONAL).alpha == 0.0

    def test_ac_ex1_transition(self, transition):
        """Full account regression for the OR gate transition (IIT 3.0).

        Verifies cause and effect accounts including each RIA's mechanism,
        purview, direction, state, alpha, probability, partitioned probability,
        and partition. Values verified against current code under
        ``IIT_3_CONFIG`` with ``ACTUAL_CAUSATION_MEASURE="PMI"``.
        """
        cause_account = actual.account(transition, Direction.CAUSE)
        assert len(cause_account) == 1
        cria = cause_account[0].ria
        assert cria.mechanism == (0,)
        assert cria.purview == (1,)
        assert cria.direction == Direction.CAUSE
        assert cria.state == (1, 0, 0)
        assert np.isclose(cria.alpha, 0.4150374992788)
        assert cria.probability == 2 / 3
        assert cria.partitioned_probability == 0.5
        assert cria.partition == models.JointBipartition(
            models.Part((), (1,)),
            models.Part((0,), ()),
        )

        effect_account = actual.account(transition, Direction.EFFECT)
        assert len(effect_account) == 2
        eria0 = effect_account[0].ria
        eria1 = effect_account[1].ria

        assert eria0.mechanism == (1,)
        assert eria0.purview == (0,)
        assert eria0.direction == Direction.EFFECT
        assert eria0.state == (0, 1, 1)
        assert np.isclose(eria0.alpha, 0.4150374992788)
        assert eria0.probability == 1.0
        assert eria0.partitioned_probability == 0.75
        assert eria0.partition == models.JointBipartition(
            models.Part((), (0,)),
            models.Part((1,), ()),
        )

        assert eria1.mechanism == (2,)
        assert eria1.purview == (0,)
        assert eria1.direction == Direction.EFFECT
        assert eria1.state == (0, 1, 1)
        assert np.isclose(eria1.alpha, 0.4150374992788)
        assert eria1.probability == 1.0
        assert eria1.partitioned_probability == 0.75
        assert eria1.partition == models.JointBipartition(
            models.Part((), (0,)),
            models.Part((2,), ()),
        )

    @pytest.mark.slow
    def test_causal_nexus(self, standard):
        """Test causal nexus computation (IIT 3.0)."""
        nexus = actual.causal_nexus(standard, (0, 0, 1), (1, 1, 0))
        assert np.isclose(nexus.alpha, 2.0)
        assert nexus.direction == Direction.BIDIRECTIONAL
        assert nexus.cause_indices == (0, 1)
        assert nexus.effect_indices == (2,)

    @pytest.mark.slow
    @pytest.mark.skip(
        reason=(
            "Major-complex selection at current_state=(0,0,1) is genuinely "
            "tied: SIAs over (1,2) and (0,2) both yield phi=1.0. The IIT 3.0 "
            "paper (Albantakis et al. 2019, p.17) explicitly says symmetric "
            "ties of this form are 'undetermined' — neither winner is "
            "paper-canonical. PyPhi must pick something deterministically, "
            "but that choice is a policy decision, not a math one. PyPhi 1.x "
            "picked (1,2) by iteration order; the current cascade picks "
            "(0,2) via partition.lex_key(), at which there are 0 true "
            "events. The downstream regression target — true_events output "
            "shape on a known irreducible complex — is covered by "
            "test_true_events_on_known_complex below. This test stays "
            "skipped until PyPhi adopts an explicit policy for symmetric "
            "major-complex ties (ROADMAP P11.95c)."
        )
    )
    def test_true_events(self, standard):
        """Full regression for true events with default major-complex selection."""
        states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # previous, current, next
        events = actual.true_events(standard, *states)

        assert len(events) == 2

        true_cause1, true_effect1 = events[0]
        assert events[0].mechanism == (1,)

        assert true_cause1.alpha == 1.0
        assert true_cause1.mechanism == (1,)
        assert true_cause1.purview == (2,)
        assert true_cause1.direction == Direction.CAUSE

        assert true_effect1.alpha == 1.0
        assert true_effect1.mechanism == (1,)
        assert true_effect1.purview == (2,)
        assert true_effect1.direction == Direction.EFFECT

        true_cause2, true_effect2 = events[1]
        assert events[1].mechanism == (2,)

        assert true_cause2.alpha == 1.0
        assert true_cause2.mechanism == (2,)
        assert true_cause2.purview == (1,)
        assert true_cause2.direction == Direction.CAUSE

        assert true_effect2.alpha == 1.0
        assert true_effect2.mechanism == (2,)
        assert true_effect2.purview == (1,)
        assert true_effect2.direction == Direction.EFFECT

    @pytest.mark.slow
    def test_true_events_on_known_complex(self, standard):
        """Event detection on a known-irreducible IIT 3.0 complex.

        Sibling to ``test_true_events`` that pins ``indices=(1, 2)`` so
        the regression target (``true_events`` output shape, per-event
        ``RIA`` mechanism / purview / alpha / direction) is covered
        without depending on ``maximal_complex`` tie-break. At
        ``current_state=(0,0,1)`` the SIA over ``(1, 2)`` is one of two
        equally-irreducible candidates (phi=1.0), so this test exercises
        event detection on it directly.
        """
        states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # previous, current, next
        events = actual.true_events(standard, *states, indices=(1, 2))

        assert len(events) == 2

        true_cause1, true_effect1 = events[0]
        assert events[0].mechanism == (1,)
        assert true_cause1.alpha == 1.0
        assert true_cause1.mechanism == (1,)
        assert true_cause1.purview == (2,)
        assert true_cause1.direction == Direction.CAUSE
        assert true_effect1.alpha == 1.0
        assert true_effect1.mechanism == (1,)
        assert true_effect1.purview == (2,)
        assert true_effect1.direction == Direction.EFFECT

        true_cause2, true_effect2 = events[1]
        assert events[1].mechanism == (2,)
        assert true_cause2.alpha == 1.0
        assert true_cause2.mechanism == (2,)
        assert true_cause2.purview == (1,)
        assert true_cause2.direction == Direction.CAUSE
        assert true_effect2.alpha == 1.0
        assert true_effect2.mechanism == (2,)
        assert true_effect2.purview == (1,)
        assert true_effect2.direction == Direction.EFFECT

    def test_true_ces(self, standard):
        """Regression for true_ces on the standard substrate (IIT 3.0).

        Computes the true cause-effect structure for the
        ``(1,0,0) -> (0,0,1) -> (1,1,0)`` transition triple and verifies
        the cause/effect mechanisms and purviews of both concepts. Values
        verified against current code under ``IIT_3_CONFIG``.
        """
        previous_state = (1, 0, 0)
        current_state = (0, 0, 1)
        next_state = (1, 1, 0)
        system = System(standard, current_state, standard.node_indices)

        ces = actual.true_ces(system, previous_state, next_state)

        assert len(ces) == 2
        actual_cause, actual_effect = ces

        assert actual_cause.purview == (0, 1)
        assert actual_cause.mechanism == (2,)

        assert actual_effect.purview == (1,)
        assert actual_effect.mechanism == (2,)

    def test_extrinsic_events(self, standard):
        """Full regression for extrinsic events on the standard substrate (IIT 3.0).

        Verifies the single extrinsic event's mechanism and the cause/effect
        RIA mechanism, purview, alpha, and direction. Values verified against
        current code under ``IIT_3_CONFIG``.
        """
        states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # previous, current, next

        events = actual.extrinsic_events(standard, *states)

        assert len(events) == 1

        true_cause, true_effect = events[0]
        assert events[0].mechanism == (2,)

        assert true_cause.alpha == 1.0
        assert true_cause.mechanism == (2,)
        assert true_cause.purview == (0, 1)
        assert true_cause.direction == Direction.CAUSE

        assert true_effect.alpha == 1.0
        assert true_effect.mechanism == (2,)
        assert true_effect.purview == (1,)
        assert true_effect.direction == Direction.EFFECT


# Paper-fixture acceptance tests: pin alpha values from
# Albantakis, Marshall, Hoel, Tononi 2019 ("What Caused What?", Entropy 21:459).


@pytest.fixture
def or_and_substrate():
    """Two-element substrate from 2019 Figs 1-6: OR and AND gates over (A, B).

    Both gates take both nodes as inputs. OR is on if either is on; AND is
    on only if both are on.
    """
    tpm = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 1],
        ]
    )
    cm = np.array([[1, 1], [1, 1]])
    return Substrate(tpm, cm, node_labels=("OR", "AND"))


def test_paper_fig5_first_order_or_alpha(or_and_substrate):
    """2019 Fig 5/6: {OR_{t-1}=1}<->{OR_t=1} has alpha approx 0.415 bits."""
    transition = actual.Transition(
        or_and_substrate,
        before_state=(1, 0),
        after_state=(1, 0),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    cause = transition.find_actual_cause((0,))
    assert cause.alpha == pytest.approx(0.415, abs=1e-2)


def test_paper_fig6_second_order_or_and_cause_alpha(or_and_substrate):
    """2019 Fig 6: {(OR,AND)_{t-1}=10}<-{(OR,AND)_t=10} has alpha approx 0.170 bits."""
    transition = actual.Transition(
        or_and_substrate,
        before_state=(1, 0),
        after_state=(1, 0),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    cause = transition.find_actual_cause((0, 1))
    assert cause.alpha == pytest.approx(0.170, abs=1e-2)


@pytest.fixture
def conjunction_substrate():
    """3-node substrate from 2019 Fig 7B: A, B independent inputs to AND-gate D.

    A and B have no inputs (uniform random under causal marginalization);
    D fires when both A_{t-1} and B_{t-1} are on.
    """
    tpm = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
        ]
    )
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(tpm, cm, node_labels=("A", "B", "D"))


def test_paper_fig7b_conjunction_alpha(conjunction_substrate):
    """2019 Fig 7B: {AB=11}<-{D=1} has alpha_c^max = 2.0 bits."""
    transition = actual.Transition(
        conjunction_substrate,
        before_state=(1, 1, 0),
        after_state=(1, 1, 1),
        cause_indices=(0, 1),
        effect_indices=(2,),
    )
    cause = transition.find_actual_cause((2,))
    assert cause.alpha == pytest.approx(2.0, abs=1e-2)
    assert tuple(sorted(cause.purview)) == (0, 1)


@pytest.fixture
def biconditional_substrate():
    """3-node substrate from 2019 Fig 7C: A, B independent inputs to XNOR-gate E.

    E fires when A_{t-1} == B_{t-1} (both 0 or both 1).
    """
    tpm = np.array(
        [
            [0.5, 0.5, 1],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
            [0.5, 0.5, 1],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 1],
        ]
    )
    cm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    return Substrate(tpm, cm, node_labels=("A", "B", "E"))


def test_paper_fig7c_biconditional_alpha(biconditional_substrate):
    """2019 Fig 7C: {AB=11}<-{E=1} has alpha = 1.0 bits via second-order purview.

    First-order purviews {A} and {B} have alpha = 0 (their cause information
    is zero); only the joint occurrence specifies a constraint.
    """
    transition = actual.Transition(
        biconditional_substrate,
        before_state=(1, 1, 0),
        after_state=(1, 1, 1),
        cause_indices=(0, 1),
        effect_indices=(2,),
    )
    cause = transition.find_actual_cause((2,))
    assert cause.alpha == pytest.approx(1.0, abs=1e-2)
    assert tuple(sorted(cause.purview)) == (0, 1)


@pytest.fixture
def majority_substrate():
    """5-node substrate from 2019 Fig 8A: A, B, C, D random inputs to majority-gate M.

    M fires when sum(A_{t-1}, B_{t-1}, C_{t-1}, D_{t-1}) >= 3.
    """
    n_nodes = 5
    n_states = 2**n_nodes
    tpm = np.zeros((n_states, n_nodes))
    for idx in range(n_states):
        bits = [(idx >> i) & 1 for i in range(n_nodes)]
        a, b, c, d, _m = bits
        tpm[idx, 0:4] = 0.5
        tpm[idx, 4] = 1.0 if (a + b + c + d) >= 3 else 0.0
    cm = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
    )
    return Substrate(tpm, cm, node_labels=("A", "B", "C", "D", "M"))


def test_paper_fig8a_majority_alpha(majority_substrate):
    """2019 Fig 8A: {ABC=111}<-{M=1} has alpha_c^max = 1.678 bits."""
    transition = actual.Transition(
        majority_substrate,
        before_state=(1, 1, 1, 0, 0),
        after_state=(1, 1, 1, 0, 1),
        cause_indices=(0, 1, 2, 3),
        effect_indices=(4,),
    )
    cause = transition.find_actual_cause((4,))
    assert cause.alpha == pytest.approx(1.678, abs=1e-2)
    # Minimality: actual cause is the third-order occurrence (3 of 4 inputs)
    assert len(cause.purview) == 3


@pytest.mark.skip(
    reason="non-binary substrate (3-state voters, 4-state W); needs non-binary units"
)
def test_paper_fig11_three_candidate_alpha():
    pass


@pytest.fixture
def probabilistic_substrate():
    """2-node substrate from 2019 Fig 12: A is a random input; N is noisy COPY of A.

    p(N_t=1 | A_{t-1}=1) = 0.9; p(N_t=1 | A_{t-1}=0) = 0.1.
    """
    tpm = np.array(
        [
            [0.5, 0.1],
            [0.5, 0.9],
            [0.5, 0.1],
            [0.5, 0.9],
        ]
    )
    cm = np.array([[0, 1], [0, 0]])
    return Substrate(tpm, cm, node_labels=("A", "N"))


def test_paper_fig12_probabilistic_alpha(probabilistic_substrate):
    """2019 Fig 12: {A=1}<->{N=1} has alpha_c^max = alpha_e^max approx 0.848 bits."""
    transition = actual.Transition(
        probabilistic_substrate,
        before_state=(1, 0),
        after_state=(1, 1),
        cause_indices=(0,),
        effect_indices=(1,),
    )
    cause = transition.find_actual_cause((1,))
    effect = transition.find_actual_effect((0,))
    assert cause.alpha == pytest.approx(0.848, abs=1e-2)
    assert effect.alpha == pytest.approx(0.848, abs=1e-2)
