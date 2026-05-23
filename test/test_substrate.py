import numpy as np
import pytest

from pyphi import Direction
from pyphi import config
from pyphi import exceptions
from pyphi import substrate as substrate_mod
from pyphi.substrate import Substrate


@pytest.fixture()
def substrate():
    size = 3
    tpm = np.ones([2] * size + [size]).astype(float) / 2
    return Substrate(tpm)


def test_substrate_init_validation(substrate):
    with pytest.raises(ValueError):
        # Totally wrong shape
        tpm = np.arange(3).astype(float)
        Substrate(tpm)
    with pytest.raises(ValueError):
        # Non-binary nodes (4 states)
        tpm = np.ones((4, 4, 4, 3)).astype(float)
        Substrate(tpm)

    # Conditionally dependent
    # fmt: off
    tpm = np.array([
            [1, 0.0, 0.0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.0, 0.0, 1],
    ])
    # fmt: on
    with config.override(validate_conditional_independence=False):
        Substrate(tpm)
    with (
        config.override(validate_conditional_independence=True),
        pytest.raises(exceptions.ConditionallyDependentError),
    ):
        Substrate(tpm)


def test_substrate_creates_fully_connected_cm_by_default():
    tpm = np.zeros((2 * 2 * 2, 3))
    substrate = Substrate(tpm, cm=None)
    target_cm = np.ones((3, 3))
    assert np.array_equal(substrate.cm, target_cm)


def test_potential_purviews(s):
    mechanism = (0,)
    assert s.substrate.potential_purviews(Direction.CAUSE, mechanism) == [
        (1,),
        (2,),
        (1, 2),
    ]
    assert s.substrate.potential_purviews(Direction.EFFECT, mechanism) == [(2,)]


def test_node_labels(standard):
    labels = ("A", "B", "C")
    substrate = Substrate(standard.joint_tpm(), node_labels=labels)
    assert substrate.node_labels.labels == labels

    labels = ("A", "B")  # Too few labels
    with pytest.raises(ValueError):
        Substrate(standard.joint_tpm(), node_labels=labels)

    # Auto-generated labels
    substrate = Substrate(standard.joint_tpm(), node_labels=None)
    assert substrate.node_labels.labels == ("n0", "n1", "n2")


def test_num_states(standard):
    assert standard.num_states == 8


def test_repr(standard):
    print(repr(standard))


def test_str(standard):
    print(str(standard))


def test_len(standard):
    assert len(standard) == 3


def test_size(standard):
    assert standard.size == 3


class _StubSia:
    """Minimal SIA stub for _iit3_exclusion_cascade tests.

    Supports ``phi`` (for tier grouping) and ``node_indices`` (for
    overlap detection and the cascade key). ``__lt__`` keys on
    ``(phi, node_indices)`` so sorted(...) produces a deterministic
    ordering.
    """

    def __init__(self, phi, node_indices):
        self.phi = phi
        self.node_indices = tuple(node_indices)

    def __lt__(self, other):
        return (self.phi, self.node_indices) < (other.phi, other.node_indices)

    def __repr__(self):
        return f"_StubSia(phi={self.phi}, node_indices={self.node_indices})"


def test_iit3_cascade_single_tier_non_overlapping_accepts_all():
    """Tied phi, non-overlapping units: both accepted as complexes."""
    a = _StubSia(phi=1.0, node_indices=(0, 1))
    b = _StubSia(phi=1.0, node_indices=(2, 3))
    result = substrate_mod._iit3_exclusion_cascade(  # pyright: ignore[reportPrivateUsage]
        sorted([a, b], reverse=True), substrate=None, state=None
    )
    assert set(result) == {a, b}


def test_iit3_cascade_single_tier_overlapping_yields_no_complex():
    """Tied phi, overlapping units: indeterminate clique, no complex."""
    a = _StubSia(phi=1.0, node_indices=(0, 2))
    b = _StubSia(phi=1.0, node_indices=(1, 2))
    result = substrate_mod._iit3_exclusion_cascade(  # pyright: ignore[reportPrivateUsage]
        sorted([a, b], reverse=True), substrate=None, state=None
    )
    assert result == []


def test_iit3_cascade_multi_tier_lower_phi_accepted_when_higher_is_indeterminate():
    """If the top tier is indeterminate, the next tier still gets evaluated."""
    a = _StubSia(phi=2.0, node_indices=(0, 2))
    b = _StubSia(
        phi=2.0, node_indices=(1, 2)
    )  # ties with a; overlaps; clique indeterminate
    c = _StubSia(
        phi=1.0, node_indices=(0, 1)
    )  # different tier; the indeterminate clique
    # didn't cover any units (no winner accepted),
    # so c is evaluated and accepted.
    result = substrate_mod._iit3_exclusion_cascade(  # pyright: ignore[reportPrivateUsage]
        sorted([a, b, c], reverse=True), substrate=None, state=None
    )
    assert result == [c]


def test_iit3_cascade_solo_candidate_accepted():
    """A solo high-phi SIA is accepted without ambiguity."""
    a = _StubSia(phi=3.0, node_indices=(0, 1, 2))
    result = substrate_mod._iit3_exclusion_cascade(  # pyright: ignore[reportPrivateUsage]
        [a], substrate=None, state=None
    )
    assert result == [a]


def test_iit3_cascade_empty_input_returns_empty():
    """No candidates → no complexes."""
    result = substrate_mod._iit3_exclusion_cascade(  # pyright: ignore[reportPrivateUsage]
        [], substrate=None, state=None
    )
    assert result == []


def test_complexes_dispatches_to_iit3_cascade_under_iit3_version():
    """Integration: complexes() routes IIT_3_0 through _iit3_exclusion_cascade.

    Worked example: ``basic_substrate`` at ``current_state=(0,0,1)`` has
    two SIAs over ``(1,2)`` and ``(0,2)`` tied at ``phi=1.0`` that
    overlap on node 2 — these form one indeterminate clique, so the
    top tier produces no complex. The cascade then continues to the
    next tier, where the full subsystem ``(0,1,2)`` at ``phi=0.1875``
    has no overlap with the (empty) covered set and is accepted as
    the sole complex. This mirrors IIT 4.0's exclusion-cascade
    behavior: indeterminate cliques skip without poisoning their
    overlap region.
    """
    from pyphi import examples
    from pyphi.conf import presets

    substrate = examples.basic_substrate()
    state = (0, 0, 1)
    with config.override(**presets.iit3):
        result = substrate_mod.complexes(substrate, state)
    assert len(result) == 1
    assert tuple(result[0].node_indices) == (0, 1, 2)
    assert abs(float(result[0].phi) - 0.1875) < 1e-4


def test_greedy_condensation_no_longer_exists():
    """_greedy_condensation was unreachable after the dispatch change; removed."""
    assert not hasattr(substrate_mod, "_greedy_condensation")
