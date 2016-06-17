import pytest

from pyphi import actual, examples

# TODO:
# ~~~~~
#   * make pytest fixtures out of AC examples
#   * test context equality/hash
#   * state_probability


def test_context_initialization():
    context = examples.ac_ex1_context()
    assert context.effect_system.state == (0, 1, 1)
    assert context.cause_system.state == (1, 0, 0)
    assert tuple(n.state for n in context.cause_system.nodes) == (1, 0, 0)


def test_purview_state():
    context = examples.ac_ex1_context()
    assert context.purview_state('past') == (0, 1, 1)
    assert context.purview_state('future') == (1, 0, 0)


def test_mechanism_state():
    context = examples.ac_ex1_context()
    assert context.mechanism_state('past') == (1, 0, 0)
    assert context.mechanism_state('future') == (0, 1, 1)


def test_system_dict():
    context = examples.ac_ex1_context()
    assert context.system['past'] == context.cause_system
    assert context.system['future'] == context.effect_system


def test_coefficients():
    context = examples.ac_ex1_context()
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


def test_ac_ex1_context():
    """Basic regression test for ac_ex1 example."""
    context = examples.ac_ex1_context()

    cause_account = actual.directed_account(context, 'past')
    assert len(cause_account) == 1
    cmip = cause_account[0].mip

    assert cmip.mechanism == (0,)
    assert cmip.purview == (2,)
    assert cmip.direction == 'past'
    assert cmip.state == (1, 0, 0)
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
