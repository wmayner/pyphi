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
    '''An OR gate with two inputs. The OR gate is ON, others are OFF.'''
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
#

@pytest.fixture
def background_all_on():
    '''Two OR gates, both ON.

    If we look at the transition A -> B, then B should be frozen at t-1, and
    A should have no effect on B.
    '''
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
    '''Two OR gates, both OFF.'''
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
    (background_all_off, Direction.FUTURE, (0,), (1,), 1),
    (background_all_off, Direction.PAST, (1,), (0,), 1),
    (background_all_on, Direction.FUTURE, (0,), (1,), 0),
    (background_all_on, Direction.PAST, (1,), (0,), 0)])
def test_background_conditions(transition, direction, mechanism, purview, ratio):
    assert transition()._ratio(direction, mechanism, purview) == ratio


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

    assert transition._ratio(Direction.FUTURE, (0,), (1,)) == 0.415037
    assert transition._ratio(Direction.PAST, (1,), (0,)) == 0.415037

    # Elements outside the transition are also frozen
    transition = actual.Transition(network, state, state, (0,), (0,),
                                   noise_background=True)
    assert np.array_equal(transition.cause_system.tpm, network.tpm)
    assert np.array_equal(transition.effect_system.tpm, network.tpm)


@pytest.fixture
def background_3_node():
    '''A is MAJ(ABC). B is OR(A, C). C is COPY(A).'''
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
    '''Looking at transition (AB = 11) -> (AC = 11)'''
    after_state = (1, 1, 1)
    transition = actual.Transition(background_3_node, before_state, after_state,
                                   (0, 1), (0, 2))
    causal_link = transition.find_causal_link(Direction.FUTURE, (0, 1))
    assert causal_link.purview == purview
    assert causal_link.alpha == alpha


def test_potential_purviews(background_3_node):
    '''Purviews must be a subset of the corresponding cause/effect system.'''
    transition = actual.Transition(background_3_node, (1, 1, 1), (1, 1, 1),
                                   (0, 1), (0, 2))
    assert transition.potential_purviews(Direction.PAST, (0, 2)) == [
        (0,), (1,), (0, 1)]
    assert transition.potential_purviews(Direction.FUTURE, (0, 1)) == [
        (0,), (2,), (0, 2)]


# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_transition_initialization(transition):
    assert transition.effect_system.state == (0, 1, 1)
    assert transition.cause_system.state == (1, 0, 0)
    assert tuple(n.state for n in transition.cause_system.nodes) == (1, 0, 0)


def test_purview_state(transition):
    assert transition.purview_state(Direction.PAST) == (0, 1, 1)
    assert transition.purview_state(Direction.FUTURE) == (1, 0, 0)


def test_mechanism_state(transition):
    assert transition.mechanism_state(Direction.PAST) == (1, 0, 0)
    assert transition.mechanism_state(Direction.FUTURE) == (0, 1, 1)


def test_mechanism_indices(transition):
    assert transition.mechanism_indices(Direction.PAST) == (0,)
    assert transition.mechanism_indices(Direction.FUTURE) == (1, 2)


def test_purview_indices(transition):
    assert transition.purview_indices(Direction.PAST) == (1, 2)
    assert transition.purview_indices(Direction.FUTURE) == (0,)


def test_system_dict(transition):
    assert transition.system[Direction.PAST] == transition.cause_system
    assert transition.system[Direction.FUTURE] == transition.effect_system


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
    cut = ac_cut(Direction.PAST, Part((1,), (2,)), Part((), (0,)))
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

def acmip(**kwargs):
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
    return models.AcMip(**defaults)


def causal_link(**kwargs):
    return models.CausalLink(acmip(**kwargs))


def account(links=()):
    return models.Account(links)


def ac_bigmip(**kwargs):
    defaults = {
        'alpha': 0.0,
        'direction': Direction.BIDIRECTIONAL,
        'unpartitioned_account': account(),
        'partitioned_account': account(),
        'transition': None,
        'cut': None
    }
    defaults.update(kwargs)
    return models.AcBigMip(**defaults)


def test_acmip_ordering():
    assert acmip() == acmip()
    assert acmip(alpha=0.0) < acmip(alpha=1.0)
    assert acmip(alpha=0.0, mechanism=(1, 2)) <= acmip(alpha=1.0, mechanism=(1,))
    assert acmip(alpha=0.0, mechanism=(1, 2)) > acmip(alpha=0.0, mechanism=(1,))

    assert bool(acmip(alpha=1.0)) is True
    assert bool(acmip(alpha=0.0)) is False
    assert bool(acmip(alpha=-1)) is False

    with pytest.raises(TypeError):
        acmip(direction=Direction.PAST) < acmip(direction=Direction.FUTURE)

    with config.override(PICK_SMALLEST_PURVIEW=True):
        assert acmip(purview=(1,)) > acmip(purview=(0, 2))


def test_acmip_hash():
    hash(acmip())


def test_acmip_phi_alias():
    assert acmip(alpha=3.3).phi == 3.3


def test_causal_link_ordering():
    assert causal_link() == causal_link()

    assert causal_link(alpha=0.0) < causal_link(alpha=1.0)
    assert causal_link(alpha=0.0, mechanism=(1, 2)) <= causal_link(alpha=1.0, mechanism=(1,))
    assert causal_link(alpha=0.0, mechanism=(1, 2)) > causal_link(alpha=0.0, mechanism=(1,))

    with pytest.raises(TypeError):
        causal_link(direction=Direction.PAST) < causal_link(direction=Direction.FUTURE)

    assert bool(causal_link(alpha=1.0)) is True
    assert bool(causal_link(alpha=0.0)) is False
    assert bool(causal_link(alpha=-1)) is False


def test_account_irreducible_causes_and_effects():
    cause = causal_link(direction=Direction.PAST)
    effect = causal_link(direction=Direction.FUTURE)
    account = models.Account((cause, effect))

    assert account.irreducible_causes == (cause,)
    assert account.irreducible_effects == (effect,)


def test_account_repr_and_str():
    str(models.Account())
    repr(models.Account())


def test_ac_big_mip_repr_and_str(transition):
    bm = ac_bigmip(transition=transition)
    str(bm)
    repr(bm)


def test_ac_big_mip_ordering(transition, empty_transition):
    assert ac_bigmip() == ac_bigmip()
    assert hash(ac_bigmip()) == hash(ac_bigmip())

    assert (ac_bigmip(alpha=1.0, transition=transition) >
            ac_bigmip(alpha=0.5, transition=transition))
    assert (ac_bigmip(alpha=1.0, transition=empty_transition) <=
            ac_bigmip(alpha=1.0, transition=transition))


@pytest.mark.parametrize('direction,mechanism,purview,repertoire', [
    (Direction.PAST, (0,), (1,), [[[0.3333333], [0.66666667]]]),
    (Direction.PAST, (0,), (2,), [[[0.3333333, 0.66666667]]]),
    (Direction.PAST, (0,), (1, 2), [[[0, 0.3333333], [0.3333333, 0.3333333]]]),
    (Direction.FUTURE, (1,), (0,), [[[0]], [[1]]]),
    (Direction.FUTURE, (2,), (0,), [[[0]], [[1]]]),
    (Direction.FUTURE, (1, 2), (0,), [[[0]], [[1]]]),
])
def test_repertoires(direction, mechanism, purview, repertoire, transition):
    np.testing.assert_array_almost_equal(
        transition.repertoire(direction, mechanism, purview), repertoire)


def test_invalid_repertoires(transition):
    '''Check that elements outside the transition cannot be passed in
    the mechanism or purview.'''
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
    (Direction.PAST, (0,), (1,), 0.66666667),
    (Direction.PAST, (0,), (2,), 0.66666667),
    (Direction.PAST, (0,), (1, 2), 0.3333333),
    (Direction.FUTURE, (1,), (0,), 1),
    (Direction.FUTURE, (2,), (0,), 1),
    (Direction.FUTURE, (1, 2), (0,), 1),
])
def test_probability(direction, mechanism, purview, probability, transition):
    assert np.isclose(transition.probability(direction, mechanism, purview),
                      probability)


def test_unconstrained_probability(transition):
    assert transition.unconstrained_probability(Direction.PAST, (1,)) == 0.5
    assert transition.unconstrained_probability(Direction.FUTURE, (0,)) == 0.75


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
    '''Basic regression test for ac_ex1 example.'''

    cause_account = actual.account(transition, Direction.PAST)
    assert len(cause_account) == 1
    cmip = cause_account[0].mip

    assert cmip.mechanism == (0,)
    assert cmip.purview == (1,)
    assert cmip.direction == Direction.PAST
    assert cmip.state == (1, 0, 0)
    assert cmip.alpha == 0.415037
    assert cmip.probability == 0.66666666666666663
    assert cmip.partitioned_probability == 0.5
    assert cmip.partition == (((), (1,)), ((0,), ()))

    effect_account = actual.account(transition, Direction.FUTURE)
    assert len(effect_account) == 2
    emip0 = effect_account[0].mip
    emip1 = effect_account[1].mip

    assert emip0.mechanism == (1,)
    assert emip0.purview == (0,)
    assert emip0.direction == Direction.FUTURE
    assert emip0.state == (0, 1, 1)
    assert emip0.alpha == 0.415037
    assert emip0.probability == 1.0
    assert emip0.partitioned_probability == 0.75
    assert emip0.partition == (((), (0,)), ((1,), ()))

    assert emip1.mechanism == (2,)
    assert emip1.purview == (0,)
    assert emip1.direction == Direction.FUTURE
    assert emip1.state == (0, 1, 1)
    assert emip1.alpha == 0.415037
    assert emip1.probability == 1.0
    assert emip1.partitioned_probability == 0.75
    assert emip1.partition == (((), (0,)), ((2,), ()))


def test_actual_cut_indices():
    cut = ac_cut(Direction.PAST, Part((0,), (2,)), Part((4,), (5,)))
    assert cut.indices == (0, 2, 4, 5)
    cut = ac_cut(Direction.FUTURE, Part((0, 2), (0, 2)), Part((), ()))
    assert cut.indices == (0, 2)


def test_actual_apply_cut():
    cut = ac_cut(Direction.PAST, Part((0,), (0, 2)), Part((2,), ()))
    cm = np.ones((3, 3))
    assert np.array_equal(cut.apply_cut(cm), np.array([
        [1, 1, 0],
        [1, 1, 1],
        [1, 1, 0]]))


def test_actual_cut_matrix():
    cut = ac_cut(Direction.PAST, Part((0,), (0, 2)), Part((2,), ()))
    assert np.array_equal(cut.cut_matrix(3), np.array([
        [0, 0, 1],
        [0, 0, 0],
        [0, 0, 1]]))


def ac_cut(direction, *parts):
    return models.ActualCut(direction, KPartition(*parts))


@config.override(PARTITION_TYPE='TRI')
@pytest.mark.parametrize('direction,answer', [
    (Direction.BIDIRECTIONAL, [
        ac_cut(Direction.PAST, Part((), ()), Part((), (1, 2)), Part((0,), ())),
        ac_cut(Direction.FUTURE, Part((), ()), Part((1,), (0,)), Part((2,), ())),
        ac_cut(Direction.FUTURE, Part((), ()), Part((1,), ()), Part((2,), (0,)))]),
    (Direction.PAST, [
        ac_cut(Direction.PAST, Part((), ()), Part((), (1, 2)), Part((0,), ()))]),
    (Direction.FUTURE, [
        ac_cut(Direction.FUTURE, Part((), ()), Part((), (0,)), Part((1, 2), ())),
        ac_cut(Direction.FUTURE, Part((), ()), Part((1,), (0,)), Part((2,), ())),
        ac_cut(Direction.FUTURE, Part((), ()), Part((1,), ()), Part((2,), (0,)))])])
def test_get_actual_cuts(direction, answer, transition):
    cuts = list(actual._get_cuts(transition, direction))
    print(cuts, answer)
    np.testing.assert_array_equal(cuts, answer)


def test_big_acmip(transition):
    bigmip = actual.big_acmip(transition)
    assert bigmip.alpha == 0.415037
    assert bigmip.cut == ac_cut(Direction.PAST, Part((), (1,)), Part((0,), (2,)))
    assert len(bigmip.unpartitioned_account) == 3
    assert len(bigmip.partitioned_account) == 2


def test_null_ac_bigmip(transition):
    bigmip = actual._null_ac_bigmip(transition, Direction.PAST)
    assert bigmip.transition == transition
    assert bigmip.direction == Direction.PAST
    assert bigmip.unpartitioned_account == ()
    assert bigmip.partitioned_account == ()
    assert bigmip.alpha == 0.0

    bigmip = actual._null_ac_bigmip(transition, Direction.PAST, alpha=float('inf'))
    assert bigmip.alpha == float('inf')


@config.override(PARTITION_TYPE='TRI')
def test_prevention(prevention):
    assert actual.big_acmip(prevention, Direction.PAST).alpha == 0.415037
    assert actual.big_acmip(prevention, Direction.FUTURE).alpha == 0.0
    assert actual.big_acmip(prevention, Direction.BIDIRECTIONAL).alpha == 0.0


def test_causal_nexus(standard):
    nexus = actual.causal_nexus(standard, (0, 0, 1), (1, 1, 0))
    assert nexus.alpha == 2.0
    assert nexus.direction == Direction.BIDIRECTIONAL
    assert nexus.transition.cause_indices == (0, 1)
    assert nexus.transition.effect_indices == (2,)


def test_true_events(standard):
    states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # Past, current, future
    events = actual.true_events(standard, *states)

    assert len(events) == 2

    true_cause1, true_effect1 = events[0]
    assert events[0].mechanism == (1,)

    assert true_cause1.alpha == 1.0
    assert true_cause1.mechanism == (1,)
    assert true_cause1.purview == (2,)
    assert true_cause1.direction == Direction.PAST

    assert true_effect1.alpha == 1.0
    assert true_effect1.mechanism == (1,)
    assert true_effect1.purview == (2,)
    assert true_effect1.direction == Direction.FUTURE

    true_cause2, true_effect2 = events[1]
    assert events[1].mechanism == (2,)

    assert true_cause2.alpha == 1.0
    assert true_cause2.mechanism == (2,)
    assert true_cause2.purview == (1,)
    assert true_cause2.direction == Direction.PAST

    assert true_effect2.alpha == 1.0
    assert true_effect2.mechanism == (2,)
    assert true_effect2.purview == (1,)
    assert true_effect2.direction == Direction.FUTURE


def test_true_constellation(standard):
    past_state = (1, 0, 0)
    current_state = (0, 0, 1)
    future_state = (1, 1, 0)
    subsystem = Subsystem(standard, current_state, standard.node_indices)

    constellation = actual.true_constellation(subsystem, past_state, future_state)

    assert len(constellation) == 2
    actual_cause, actual_effect = constellation

    assert actual_cause.purview == (0, 1)
    assert actual_cause.mechanism == (2,)

    assert actual_effect.purview == (1,)
    assert actual_effect.mechanism == (2,)


def test_extrinsic_events(standard):
    states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # Past, current, future

    events = actual.extrinsic_events(standard, *states)

    assert len(events) == 1

    true_cause, true_effect = events[0]
    assert events[0].mechanism == (2,)

    assert true_cause.alpha == 1.0
    assert true_cause.mechanism == (2,)
    assert true_cause.purview == (0, 1)
    assert true_cause.direction == Direction.PAST

    assert true_effect.alpha == 1.0
    assert true_effect.mechanism == (2,)
    assert true_effect.purview == (1,)
    assert true_effect.direction == Direction.FUTURE
