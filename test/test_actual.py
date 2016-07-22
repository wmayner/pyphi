import pytest

from pyphi import actual, examples, models

# TODO
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   * test context equality/hash
#   * state_probability


# Fixtures
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture
def context():
    return examples.ac_ex1_context()


@pytest.fixture
def empty_context(context):
    return actual.Context(context.network, context.before_state,
                          context.after_state, (), ())


# Tests
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def test_context_initialization(context):
    assert context.effect_system.state == (0, 1, 1)
    assert context.cause_system.state == (1, 0, 0)
    assert tuple(n.state for n in context.cause_system.nodes) == (1, 0, 0)


def test_purview_state(context):
    assert context.purview_state('past') == (0, 1, 1)
    assert context.purview_state('future') == (1, 0, 0)


def test_mechanism_state(context):
    assert context.mechanism_state('past') == (1, 0, 0)
    assert context.mechanism_state('future') == (0, 1, 1)


def test_system_dict(context):
    assert context.system['past'] == context.cause_system
    assert context.system['future'] == context.effect_system


def test_context_len(context, empty_context):
    assert len(context) == 3
    assert len(empty_context) == 0


def test_context_bool(context, empty_context):
    assert bool(context)
    assert not bool(empty_context)


def test_context_apply_cut(context):
    cut = models.ActualCut((1,), (2,), (), (0,))
    cut_context = context.apply_cut(cut)
    assert cut_context.before_state == context.before_state
    assert cut_context.after_state == context.after_state
    assert cut_context.cause_indices == context.cause_indices
    assert cut_context.effect_indices == context.effect_indices
    assert cut_context.cut == cut
    assert cut_context != context


def test_to_json(context):
    context.to_json()

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
        'unconstrained_probability': 0.0
    }
    defaults.update(kwargs)
    return models.AcMip(**defaults)


def acmice(**kwargs):
    return models.AcMice(acmip(**kwargs))


def test_acmip_ordering():
    assert acmip() == acmip()
    assert acmip(alpha=0.0) < acmip(alpha=1.0)
    assert acmip(alpha=0.0, mechanism=(1, 2)) <= acmip(alpha=1.0, mechanism=(1,))
    assert acmip(alpha=0.0, mechanism=(1, 2)) > acmip(alpha=0.0, mechanism=(1,))

    with pytest.raises(TypeError):
        acmip(direction='past') < acmip(direction='future')


def test_acmip_hash():
    hash(acmip())


def test_acmice_ordering():
    assert acmice() == acmice()

    assert acmice(alpha=0.0) < acmice(alpha=1.0)
    assert acmice(alpha=0.0, mechanism=(1, 2)) <= acmice(alpha=1.0, mechanism=(1,))
    assert acmice(alpha=0.0, mechanism=(1, 2)) > acmice(alpha=0.0, mechanism=(1,))

    with pytest.raises(TypeError):
        acmice(direction='past') < acmice(direction='future')


def test_coefficients(context):
    A, B, C = (0, 1, 2)

    assert context.cause_coefficient((A,), (B, C), norm=False) == 1/3
    assert context.cause_coefficient((A,), (B,), norm=False) == 2/3
    assert context.cause_coefficient((A,), (C,), norm=False) == 2/3
    assert context.effect_coefficient((B, C), (A,), norm=False) == 1
    assert context.effect_coefficient((A,), (B, C), norm=False) == 1/4
    # ...

    assert context.cause_coefficient((A,), (B, C)) == 4/3
    assert context.cause_coefficient((A,), (B,)) == 4/3
    assert context.cause_coefficient((A,), (C,)) == 4/3
    assert context.effect_coefficient((B, C), (A,)) == 4/3
    assert context.effect_coefficient((A,), (B, C)) == 1
    # ...


def test_ac_ex1_context(context):
    """Basic regression test for ac_ex1 example."""

    cause_account = actual.directed_account(context, 'past')
    assert len(cause_account) == 1
    cmip = cause_account[0].mip

    assert cmip.mechanism == (0,)
    assert cmip.purview == (1,)
    assert cmip.direction == 'past'
    assert cmip.state == (1, 0, 0)
    assert cmip.alpha == 0.33333333333333326
    assert cmip.probability == 0.66666666666666663
    assert cmip.partitioned_probability == 0.5
    assert cmip.unconstrained_probability == 0.5
    assert cmip.partition == (((), (1,)), ((0,), ()))

    effect_account = actual.directed_account(context, 'future')
    assert len(effect_account) == 2
    emip0 = effect_account[0].mip
    emip1 = effect_account[1].mip

    assert emip0.mechanism == (1,)
    assert emip0.purview == (0,)
    assert emip0.direction == 'future'
    assert emip0.state == (0, 1, 1)
    assert emip0.alpha == 0.33333333333333331
    assert emip0.probability == 1.0
    assert emip0.partitioned_probability == 0.75
    assert emip0.unconstrained_probability == 0.75
    assert emip0.partition == (((), (0,)), ((1,), ()))

    assert emip1.mechanism == (2,)
    assert emip1.purview == (0,)
    assert emip1.direction == 'future'
    assert emip1.state == (0, 1, 1)
    assert emip1.alpha == 0.33333333333333331
    assert emip1.probability == 1.0
    assert emip1.partitioned_probability == 0.75
    assert emip1.unconstrained_probability == 0.75
    assert emip1.partition == (((), (0,)), ((2,), ()))


# TODO: fix unreachable state issue
@pytest.mark.xfail
def test_ac_ex3_context():
    """Regression test for ac_ex3 example"""
    context = examples.ac_ex3_context()

    cause_account = actual.directed_account(context, 'past')
    assert len(cause_account) == 1
    cmip = cause_account[0].mip

    assert cmip.mechanism == (0,)
    assert cmip.purview == (2,)
    assert cmip.direction == 'past'
    assert cmip.state == (0, 0, 0)
    assert cmip.alpha == 0.33333333333333326
    assert cmip.probability == 0.66666666666666663
    assert cmip.partitioned_probability == 0.5
    assert cmip.unconstrained_probability == 0.5
    assert cmip.partition == (((), (2,)), ((0,), ()))

    effect_account = actual.directed_account(context, 'future')
    assert len(effect_account) == 2
    emip0 = effect_account[0].mip
    emip1 = effect_account[1].mip

    assert emip0.mechanism == (1,)
    assert emip0.purview == (0,)
    assert emip0.direction == 'future'
    assert emip0.state == (0, 0, 1)
    assert emip0.alpha == 0.33333333333333331
    assert emip0.probability == 1.0
    assert emip0.partitioned_probability == 0.75
    assert emip0.unconstrained_probability == 0.75
    assert emip0.partition == (((), (0,)), ((1,), ()))

    assert emip1.mechanism == (2,)
    assert emip1.purview == (0,)
    assert emip1.direction == 'future'
    assert emip1.state == (0, 0, 1)
    assert emip1.alpha == 0.33333333333333331
    assert emip1.probability == 1.0
    assert emip1.partitioned_probability == 0.75
    assert emip1.unconstrained_probability == 0.75
    assert emip1.partition == (((), (0,)), ((2,), ()))


def test_actual_cut_indices():
    cut = models.ActualCut((0,), (4,), (2,), (5,))
    assert cut.indices == (0, 2, 4, 5)
    cut = models.ActualCut((0, 2), (), (0, 2), ())
    assert cut.indices == (0, 2)


def test_big_acmip(context):
    bigmip = actual.big_acmip(context)
    assert bigmip.alpha == 0.33333333333333326
    assert bigmip.cut == models.ActualCut((1,), (2,), (), (0,))
    assert len(bigmip.unpartitioned_account) == 3
    assert len(bigmip.partitioned_account) == 2


def test_causal_nexus(standard):
    nexus = actual.causal_nexus(standard, (0, 0, 1), (1, 1, 0))
    assert nexus.alpha == 2.0
    assert nexus.direction == 'bidirectional'
    assert nexus.context.cause_indices == (0, 1)
    assert nexus.context.effect_indices == (2,)


def test_true_events(standard):
    states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # Past, current, future
    events = actual.true_events(standard, *states)

    assert len(events) == 2

    true_cause1, true_effect1 = events[0]
    assert events[0].mechanism == (0,)

    assert true_cause1.alpha == 1.0
    assert true_cause1.mechanism == (0,)
    assert true_cause1.purview == (2,)
    assert true_cause1.direction == 'past'

    assert true_effect1.alpha == 1.0
    assert true_effect1.mechanism == (0,)
    assert true_effect1.purview == (2,)
    assert true_effect1.direction == 'future'

    true_cause2, true_effect2 = events[1]
    assert events[1].mechanism == (2,)

    assert true_cause2.alpha == 1.0
    assert true_cause2.mechanism == (2,)
    assert true_cause2.purview == (0,)
    assert true_cause2.direction == 'past'

    assert true_effect2.alpha == 1.0
    assert true_effect2.mechanism == (2,)
    assert true_effect2.purview == (0,)
    assert true_effect2.direction == 'future'


def test_extrinsic_events(standard):
    states = ((1, 0, 0), (0, 0, 1), (1, 1, 0))  # Past, current, future

    events = actual.extrinsic_events(standard, *states)

    assert len(events) == 1

    true_cause, true_effect = events[0]
    assert events[0].mechanism == (2,)

    assert true_cause.alpha == 1.0
    assert true_cause.mechanism == (2,)
    assert true_cause.purview == (0, 1)
    assert true_cause.direction == 'past'

    assert true_effect.alpha == 1.0
    assert true_effect.mechanism == (2,)
    assert true_effect.purview == (1,)
    assert true_effect.direction == 'future'
