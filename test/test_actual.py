import numpy as np
import pytest

from pyphi import (Direction, Network, Subsystem, actual, config, examples,
                   models)
from pyphi.models import KPartition, Part

# TODO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   * test transition equality/hash
#   * state_probability


# Fixtures
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.fixture
def transition():
    """An OR gate with two inputs. The OR gate is ON, others are OFF."""
    tpm = np.array([
        [0, 0.5, 0.5],
        [0, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5]
    ])
    cm = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0]
    ])
    network = Network(tpm, cm)
    before_state = (0, 1, 1)
    after_state = (1, 0, 0)
    return actual.Transition(network, before_state, after_state, (1, 2), (0,))


@pytest.fixture
def empty_transition(transition):
    return actual.Transition(transition.network, transition.before_state,
                             transition.after_state, (), ())


@pytest.fixture
def prevention():
    return examples.prevention()


# Testing background conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

@pytest.fixture
def background_all_on():
    """Two OR gates, both ON.

    If we look at the transition A -> B, then B should be frozen at t-1, and
    A should have no effect on B.
    """
    tpm = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    network = Network(tpm)
    state = (1, 1)
    return actual.Transition(network, state, state, (0,), (1,))


@pytest.fixture
def background_all_off():
    """Two OR gates, both OFF."""
    tpm = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    network = Network(tpm)
    state = (0, 0)
    return actual.Transition(network, state, state, (0,), (1,))


@pytest.mark.parametrize('transition,direction,mechanism,purview,ratio', [
    (pytest.lazy_fixture('background_all_off'), Direction.EFFECT, (0,), (1,), 1),
    (pytest.lazy_fixture('background_all_off'), Direction.CAUSE, (1,), (0,), 1),
    (pytest.lazy_fixture('background_all_on'), Direction.EFFECT, (0,), (1,), 0),
    (pytest.lazy_fixture('background_all_on'), Direction.CAUSE, (1,), (0,), 0)])
def test_background_conditions(transition, direction, mechanism, purview,
                               ratio):
    assert transition._ratio(direction, mechanism, purview) == ratio


def test_background_noised():
    tpm = np.array([
        [0, 0],
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    network = Network(tpm)
    state = (1, 1)
    transition = actual.Transition(network, state, state, (0,), (1,),
                                   noise_background=True)

    assert transition._ratio(Direction.EFFECT, (0,), (1,)) == 0.415037
    assert transition._ratio(Direction.CAUSE, (1,), (0,)) == 0.415037

    # Elements outside the transition are also frozen
    transition = actual.Transition(network, state, state, (0,), (0,),
                                   noise_background=True)
    assert np.array_equal(transition.cause_system.tpm, network.tpm)
    assert np.array_equal(transition.effect_system.tpm, network.tpm)


@pytest.fixture
def background_3_node():
    """A is MAJ(ABC). B is OR(A, C). C is COPY(A)."""
    tpm = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
        [1, 1, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])
    return Network(tpm)


@pytest.mark.parametrize('before_state,purview,alpha', [
    # If C = 1, then AB over AC should be reducible.
    ((1, 1, 1), (0, 2), 0.0),
    # If C = 0, then AB over AC should be irreducible.
    ((1, 1, 0), (0, 2), 1.0)])
def test_background_3_node(before_state, purview, alpha, background_3_node):
    """Looking at transition (AB = 11) -> (AC = 11)"""
    after_state = (1, 1, 1)
    transition = actual.Transition(background_3_node, before_state,
                                   after_state, (0, 1), (0, 2))
    causal_link = transition.find_causal_link(Direction.EFFECT, (0, 1))
    assert causal_link.purview == purview
    assert causal_link.alpha == alpha


def test_potential_purviews(background_3_node):
    """Purviews must be a subset of the corresponding cause/effect system."""
    transition = actual.Transition(background_3_node, (1, 1, 1), (1, 1, 1),
                                   (0, 1), (0, 2))
    assert transition.potential_purviews(Direction.CAUSE, (0, 2)) == [
        (0,), (1,), (0, 1)]
    assert transition.potential_purviews(Direction.EFFECT, (0, 1)) == [
        (0,), (2,), (0, 2)]


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


def test_transition_apply_cut(transition):
    cut = ac_cut(Direction.CAUSE, Part((1,), (2,)), Part((), (0,)))
    cut_transition = transition.apply_cut(cut)
    assert cut_transition.before_state == transition.before_state
    assert cut_transition.after_state == transition.after_state
    assert cut_transition.cause_indices == transition.cause_indices
    assert cut_transition.effect_indices == transition.effect_indices
    assert cut_transition.cut == cut
    assert cut_transition != transition


def test_to_json(transition):
    transition.to_json()


# Test AC models
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def acria(**kwargs):
    defaults = {
        'alpha': 0.0,
        'state': None,
        'direction': None,
        'mechanism': (),
        'purview': (),
        'partition': None,
        'probability': 0.0,
        'partitioned_probability': 0.0,
    }
    defaults.update(kwargs)
    return models.AcRepertoireIrreducibilityAnalysis(**defaults)


def causal_link(**kwargs):
    return models.CausalLink(acria(**kwargs))


def account(links=()):
    return models.Account(links)


def ac_sia(**kwargs):
    defaults = {
        'alpha': 0.0,
        'direction': Direction.BIDIRECTIONAL,
        'account': account(),
        'partitioned_account': account(),
        'transition': None,
        'cut': None
    }
    defaults.update(kwargs)
    return models.AcSystemIrreducibilityAnalysis(**defaults)


def test_acria_ordering():
    assert acria() == acria()
    assert acria(alpha=0.0) < acria(alpha=1.0)
    assert (acria(alpha=0.0, mechanism=(1, 2)) <=
            acria(alpha=1.0, mechanism=(1,)))
    assert (acria(alpha=0.0, mechanism=(1, 2)) >
            acria(alpha=0.0, mechanism=(1,)))

    assert bool(acria(alpha=1.0)) is True
    assert bool(acria(alpha=0.0)) is False
    assert bool(acria(alpha=-1)) is False

    with pytest.raises(TypeError):
        acria(direction=Direction.CAUSE) < acria(direction=Direction.EFFECT)

    with config.override(PICK_SMALLEST_PURVIEW=True):
        assert acria(purview=(1,)) > acria(purview=(0, 2))


def test_acria_hash():
    hash(acria())


def test_acria_phi_alias():
    assert acria(alpha=3.3).phi == 3.3


def test_causal_link_ordering():
    assert causal_link() == causal_link()

    assert causal_link(alpha=0.0) < causal_link(alpha=1.0)
    assert (causal_link(alpha=0.0, mechanism=(1, 2)) <=
            causal_link(alpha=1.0, mechanism=(1,)))
    assert (causal_link(alpha=0.0, mechanism=(1, 2)) >
            causal_link(alpha=0.0, mechanism=(1,)))

    with pytest.raises(TypeError):
        (causal_link(direction=Direction.CAUSE) <
         causal_link(direction=Direction.EFFECT))

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
        a1 + [causal_link()]


def test_ac_sia_repr_and_str(transition):
    bm = ac_sia(transition=transition)
    str(bm)
    repr(bm)


def test_ac_sia_ordering(transition, empty_transition):
    assert ac_sia() == ac_sia()
    assert hash(ac_sia()) == hash(ac_sia())

    assert (ac_sia(alpha=1.0, transition=transition) >
            ac_sia(alpha=0.5, transition=transition))
    assert (ac_sia(alpha=1.0, transition=empty_transition) <=
            ac_sia(alpha=1.0, transition=transition))


@pytest.mark.parametrize('direction,mechanism,purview,repertoire', [
    (Direction.CAUSE, (0,), (1,), [[[0.3333333], [0.66666667]]]),
    (Direction.CAUSE, (0,), (2,), [[[0.3333333, 0.66666667]]]),
    (Direction.CAUSE, (0,), (1, 2), [[[0, 0.3333333],
                                      [0.3333333, 0.3333333]]]),
    (Direction.EFFECT, (1,), (0,), [[[0]], [[1]]]),
    (Direction.EFFECT, (2,), (0,), [[[0]], [[1]]]),
    (Direction.EFFECT, (1, 2), (0,), [[[0]], [[1]]]),
])
def test_repertoires(direction, mechanism, purview, repertoire, transition):
    np.testing.assert_array_almost_equal(
        transition.repertoire(direction, mechanism, purview), repertoire)


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
        transition.unconstrained_cause_repertoire((2,)), [[[0.5, 0.5]]])
    np.testing.assert_array_equal(
        transition.unconstrained_effect_repertoire((0,)), [[[0.25]], [[0.75]]])


@pytest.mark.parametrize('direction,mechanism,purview,probability', [
    (Direction.CAUSE, (0,), (1,), 0.66666667),
    (Direction.CAUSE, (0,), (2,), 0.66666667),
    (Direction.CAUSE, (0,), (1, 2), 0.3333333),
    (Direction.EFFECT, (1,), (0,), 1),
    (Direction.EFFECT, (2,), (0,), 1),
    (Direction.EFFECT, (1, 2), (0,), 1),
])
def test_probability(direction, mechanism, purview, probability, transition):
    assert np.isclose(transition.probability(direction, mechanism, purview),
                      probability)


def test_unconstrained_probability(transition):
    assert transition.unconstrained_probability(Direction.CAUSE, (1,)) == 0.5
    assert transition.unconstrained_probability(Direction.EFFECT, (0,)) == 0.75


@pytest.mark.parametrize('mechanism,purview,ratio', [
    ((0,), (1,), 0.41504),
    ((0,), (2,), 0.41504),
    ((0,), (1, 2), 0.41504),
])
def test_cause_ratio(mechanism, purview, ratio, transition):
    assert np.isclose(transition.cause_ratio(mechanism, purview), ratio)


@pytest.mark.parametrize('mechanism,purview,ratio', [
    ((1,), (0,), 0.41504),
    ((2,), (0,), 0.41504),
    ((1, 2), (0,), 0.41504),
])
def test_effect_ratio(mechanism, purview, ratio, transition):
    assert np.isclose(transition.effect_ratio(mechanism, purview), ratio)


def test_ac_ex1_transition(transition):
    """Basic regression test for ac_ex1 example."""

    cause_account = actual.account(transition, Direction.CAUSE)
    assert len(cause_account) == 1
    cria = cause_account[0].ria

    assert cria.mechanism == (0,)
    assert cria.purview == (1,)
    assert cria.direction == Direction.CAUSE
    assert cria.state == (1, 0, 0)
    assert cria.alpha == 0.415037
    assert cria.probability == 0.66666666666666663
    assert cria.partitioned_probability == 0.5
    assert cria.partition == models.Bipartition(models.Part((), (1,)),
                                                models.Part((0,), ()))

    effect_account = actual.account(transition, Direction.EFFECT)
    assert len(effect_account) == 2
    eria0 = effect_account[0].ria
    eria1 = effect_account[1].ria

    assert eria0.mechanism == (1,)
    assert eria0.purview == (0,)
    assert eria0.direction == Direction.EFFECT
    assert eria0.state == (0, 1, 1)
    assert eria0.alpha == 0.415037
    assert eria0.probability == 1.0
    assert eria0.partitioned_probability == 0.75
    assert eria0.partition == models.Bipartition(models.Part((), (0,)),
                                                 models.Part((1,), ()))

    assert eria1.mechanism == (2,)
    assert eria1.purview == (0,)
    assert eria1.direction == Direction.EFFECT
    assert eria1.state == (0, 1, 1)
    assert eria1.alpha == 0.415037
    assert eria1.probability == 1.0
    assert eria1.partitioned_probability == 0.75
    assert eria1.partition == models.Bipartition(models.Part((), (0,)),
                                                 models.Part((2,), ()))


def test_actual_cut_indices():
    cut = ac_cut(Direction.CAUSE, Part((0,), (2,)), Part((4,), (5,)))
    assert cut.indices == (0, 2, 4, 5)
    cut = ac_cut(Direction.EFFECT, Part((0, 2), (0, 2)), Part((), ()))
    assert cut.indices == (0, 2)


def test_actual_apply_cut():
    cut = ac_cut(Direction.CAUSE, Part((0,), (0, 2)), Part((2,), ()))
    cm = np.ones((3, 3))
    assert np.array_equal(cut.apply_cut(cm), np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 0]]))


def test_actual_cut_matrix():
    cut = ac_cut(Direction.CAUSE, Part((0,), (0, 2)), Part((2,), ()))
    assert np.array_equal(cut.cut_matrix(3), np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 1]]))


def ac_cut(direction, *parts):
    return models.ActualCut(direction, KPartition(*parts))


@config.override(PARTITION_TYPE='TRI')
@pytest.mark.parametrize('direction,answer', [
    (Direction.BIDIRECTIONAL, [
        ac_cut(Direction.CAUSE,
               Part((), ()), Part((), (1, 2)), Part((0,), ())),
        ac_cut(Direction.EFFECT,
               Part((), ()), Part((1,), (0,)), Part((2,), ())),
        ac_cut(Direction.EFFECT,
               Part((), ()), Part((1,), ()), Part((2,), (0,)))]),
    (Direction.CAUSE, [
        ac_cut(Direction.CAUSE,
               Part((), ()), Part((), (1, 2)), Part((0,), ()))]),
    (Direction.EFFECT, [
        ac_cut(Direction.EFFECT,
               Part((), ()), Part((), (0,)), Part((1, 2), ())),
        ac_cut(Direction.EFFECT,
               Part((), ()), Part((1,), (0,)), Part((2,), ())),
        ac_cut(Direction.EFFECT,
               Part((), ()), Part((1,), ()), Part((2,), (0,)))])])
def test_get_actual_cuts(direction, answer, transition):
    cuts = list(actual._get_cuts(transition, direction))
    print(cuts, answer)
    np.testing.assert_array_equal(cuts, answer)


def test_sia(transition):
    sia = actual.sia(transition)
    assert sia.alpha == 0.415037
    assert sia.cut == ac_cut(Direction.CAUSE, Part((), (1,)), Part((0,), (2,)))
    assert len(sia.account) == 3
    assert len(sia.partitioned_account) == 2


def test_null_ac_sia(transition):
    sia = actual._null_ac_sia(transition, Direction.CAUSE)
    assert sia.transition == transition
    assert sia.direction == Direction.CAUSE
    assert sia.account == ()
    assert sia.partitioned_account == ()
    assert sia.alpha == 0.0

    sia = actual._null_ac_sia(transition, Direction.CAUSE, alpha=float('inf'))
    assert sia.alpha == float('inf')


@config.override(PARTITION_TYPE='TRI')
def test_prevention(prevention):
    assert actual.sia(prevention, Direction.CAUSE).alpha == 0.415037
    assert actual.sia(prevention, Direction.EFFECT).alpha == 0.0
    assert actual.sia(prevention, Direction.BIDIRECTIONAL).alpha == 0.0


def test_causal_nexus(standard):
    nexus = actual.causal_nexus(standard, (0, 0, 1), (1, 1, 0))
    assert nexus.alpha == 2.0
    assert nexus.direction == Direction.BIDIRECTIONAL
    assert nexus.transition.cause_indices == (0, 1)
    assert nexus.transition.effect_indices == (2,)


def test_true_events(standard):
    states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # Previous, current, next
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


def test_true_ces(standard):
    previous_state = (1, 0, 0)
    current_state = (0, 0, 1)
    next_state = (1, 1, 0)
    subsystem = Subsystem(standard, current_state, standard.node_indices)

    ces = actual.true_ces(subsystem, previous_state, next_state)

    assert len(ces) == 2
    actual_cause, actual_effect = ces

    assert actual_cause.purview == (0, 1)
    assert actual_cause.mechanism == (2,)

    assert actual_effect.purview == (1,)
    assert actual_effect.mechanism == (2,)


def test_extrinsic_events(standard):
    states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # Previous, current, next

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
