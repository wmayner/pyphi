"""Tests for predicate selection on distinction bags."""

import pytest

from pyphi import examples
from pyphi.models.distinctions import ResolvedDistinctions
from pyphi.models.distinctions import UnresolvedDistinctions


@pytest.fixture(scope="module")
def xor_ces():
    return examples.xor_system().ces()


def test_filter_selects_by_predicate(xor_ces):
    result = xor_ces.distinctions.filter(lambda d: len(d.mechanism) == 2)
    assert all(len(d.mechanism) == 2 for d in result)
    assert len(result) == sum(1 for d in xor_ces.distinctions if len(d.mechanism) == 2)


def test_filter_preserves_subtype(xor_ces):
    assert isinstance(xor_ces.distinctions, ResolvedDistinctions)
    result = xor_ces.distinctions.filter(lambda _d: True)
    assert type(result) is type(xor_ces.distinctions)


def test_filter_on_unresolved_preserves_subtype(xor_ces):
    unresolved = UnresolvedDistinctions(xor_ces.distinctions)
    result = unresolved.filter(lambda _d: True)
    assert type(result) is UnresolvedDistinctions


def test_filter_empty_result(xor_ces):
    result = xor_ces.distinctions.filter(lambda _d: False)
    assert len(result) == 0
    assert type(result) is type(xor_ces.distinctions)


def test_distinction_identity_includes_specified_states():
    """Two readings of the same purview specifying different states are
    different distinctions: they support different relations and different
    structure Phi, so equality and hash must separate them (matching the RIA
    layer, which compares and hashes its specified state).
    """
    import pyphi
    from pyphi.conf import config
    from pyphi.conf import presets
    from pyphi.models.distinction import Distinction

    with config.override(**presets.iit4_2023):
        result = pyphi.analyze(examples.basic_substrate(), (1, 0, 0))
        distinctions = list(result.ces.distinctions)
        target = next(d for d in distinctions if len(d.cause.state_ties) > 1)
        alt_cause = next(t for t in target.cause.state_ties if t != target.cause)
        alt = Distinction(target.mechanism, alt_cause, target.effect)
        # The alternative reading specifies a different cause state.
        assert tuple(target.cause.specified_state.state) != tuple(
            alt.cause.specified_state.state
        )
        assert target != alt
        assert hash(target) != hash(alt)
        assert len({target, alt}) == 2


def test_macro_system_never_equals_plain_system():
    """A MacroSystem and a plain System over its macro substrate are
    different analyses (the macro construction overrides the cause TPM), so
    they must not compare equal in either direction."""
    import numpy as np

    from pyphi.conf import config
    from pyphi.conf import presets
    from pyphi.macro.system import MacroSystem
    from pyphi.macro.units import MacroUnit
    from pyphi.substrate import Substrate
    from pyphi.system import System

    with config.override(**presets.iit4_2023):
        substrate = Substrate(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
        macro = MacroSystem.from_micro(
            substrate, (MacroUnit((0, 1), 1, (0, 0, 0, 1)),), ((0, 0),)
        )
        plain = System(
            macro.substrate,
            macro.state,
            node_indices=macro.node_indices,
            external_indices=macro.external_indices,
        )
        assert macro != plain
        assert plain != macro
        assert len({macro, plain}) == 2
        assert plain not in [macro]
