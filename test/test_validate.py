import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi import System
from pyphi import exceptions
from pyphi import validate


def test_validate_direction():
    validate.direction(Direction.CAUSE)
    validate.direction(Direction.EFFECT)

    with pytest.raises(ValueError):
        validate.direction("dogeeeee")

    validate.direction(Direction.BIDIRECTIONAL, allow_bi=True)
    with pytest.raises(ValueError):
        validate.direction(Direction.BIDIRECTIONAL)


def test_validate_connectivity_matrix_valid(s):
    assert validate.connectivity_matrix(s.substrate.cm)


def test_validate_connectivity_matrix_not_square():
    cm = np.random.binomial(1, 0.5, (4, 5))
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_connectivity_matrix_not_2D():
    cm = np.arange(8).reshape(2, 2, 2)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_connectivity_matrix_not_binary():
    cm = np.arange(16).reshape(4, 4)
    with pytest.raises(ValueError):
        assert validate.connectivity_matrix(cm)


def test_validate_substrate_wrong_cm_size(s):
    with pytest.raises(ValueError):
        Substrate(s.substrate.joint_tpm(), np.ones(16).reshape(4, 4))


def test_validate_is_substrate(s):
    with pytest.raises(ValueError):
        validate.is_substrate(s)
    validate.is_substrate(s.substrate)


def test_validate_state_no_error_1(s):
    validate.state_reachable(s)


def test_validate_state_error(s):
    with pytest.raises(exceptions.StateUnreachableError):
        state = (0, 1, 0)
        System(s.substrate, state, s.node_indices)


FIXED = "CONDITION_CURRENT_STATE"
MARGINALIZED = "CAUSAL_MARGINALIZATION"


def test_validate_state_subsystem_unreachable(s):
    """Subsystem-level reachability under CONDITION_CURRENT_STATE: the
    state component must be in the image of the subsystem dynamics with
    the background's past held at its current state.

    For the standard substrate ``s`` in state ``(1, 0, 0)``, the singleton
    subsystem ``{A}`` cannot have A=1 because the conditioned dynamics
    (B=0, C=0) deterministically produce A_next = OR(B,C) = 0. Likewise
    ``{C}`` cannot have C=0 because conditioned C_next = XOR(A=1,B=0) = 1.
    ``{B}`` passes because conditioned B_next = COPY(C=0) = 0 matches B=0.
    """
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(s.substrate, s.state, (0,), background_conditioning=FIXED)
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(s.substrate, s.state, (2,), background_conditioning=FIXED)
    # No raise for B:
    System(s.substrate, s.state, (1,), background_conditioning=FIXED)


def test_validate_state_subsystem_reachable_under_causal_marginalization(s):
    """Under IIT 4.0 the substrate-level check is sufficient.

    The background's past is weighted by its probability given the current
    state (Albantakis et al. 2023, Eq. 4), not held at its current state,
    so ``{A}`` and ``{C}`` in ``(1, 0, 0)`` have causes.
    """
    for indices in [(0,), (1,), (2,)]:
        System(s.substrate, s.state, indices, background_conditioning=MARGINALIZED)


def test_validate_state_background_past_differs_from_present():
    """Regression: a state reachable only through a background past that
    differs from the background's present.

    A copies B and B flips (A' = B, B' = NOT B). In state (A=1, B=0) the
    only possible past has B=1. Holding B's past at its present value 0
    makes A=1 unreachable, which is the pathology IIT 4.0 removed (S2 Text,
    "Background Conditions"). Under Eq. 4 the cause of A=1 is certain.
    """
    tpm = np.array([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=float)
    sub = Substrate(tpm)
    state = (1, 0)
    system = System(sub, state, (0,), background_conditioning=MARGINALIZED)
    np.testing.assert_array_equal(
        system.cause_marginal.factor(0).squeeze(), [[0.0, 1.0], [0.0, 1.0]]
    )
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(sub, state, (0,), background_conditioning=FIXED)


def _k3_copy_substrate() -> Substrate:
    """k=3 two-node substrate. Node 0 copies node 1's input; node 1 is
    constant 0. So node 0's conditioned dynamics (node 1 fixed at the
    external state) can only output that fixed value."""
    f0 = np.zeros((3, 3, 3))
    f1 = np.zeros((3, 3, 3))
    for a in range(3):
        for b in range(3):
            f0[a, b, b] = 1.0  # node 0 next = node 1's input value
            f1[a, b, 0] = 1.0  # node 1 next = 0 (constant)
    return Substrate(marginals=[f0, f1])


def test_validate_state_subsystem_unreachable_kary():
    """Subsystem-level reachability for a k>2 substrate.

    Full state ``(1, 0)`` is substrate-reachable (a past with node 1 = 1
    produces node-0-next = 1, and node 1 is always 0). But subsystem
    ``{0}`` with node 1 fixed at its observed value 0 can only produce
    node-0-next = 0, so ``proper_state = (1,)`` is unreachable under the
    conditioned dynamics. Under causal marginalization the past with
    node 1 = 1 is the cause, so ``{0}`` is reachable.
    """
    sub = _k3_copy_substrate()
    # Full substrate: reachable, no raise.
    System(sub, (1, 0), (0, 1))
    # Subsystem {0}: node-0 conditioned dynamics cannot produce 1.
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(sub, (1, 0), (0,), background_conditioning=FIXED)
    System(sub, (1, 0), (0,), background_conditioning=MARGINALIZED)
    # Subsystem {1}: node 1 is constant 0, so {1}=0 is reachable.
    System(sub, (1, 0), (1,), background_conditioning=FIXED)


@pytest.mark.skip(
    reason="StateUnreachableBackwardsError not raised by current state_reachable; "
    "backward-reachability check pending implementation"
)
def test_validate_state_no_error_2():
    tpm = np.ones([16, 4])
    net = Substrate(tpm)
    # Globally impossible state.
    state = (1, 1, 0, 0)
    # But locally possible for first two nodes.
    # The forward reachability check should pass, but backward TPM computation
    # fails due to zero normalization. We expect StateUnreachableBackwardsError,
    # NOT StateUnreachableForwardsError.
    with pytest.raises(exceptions.StateUnreachableBackwardsError):
        System(net, state, (0, 1))


def test_validate_node_labels():
    validate.node_labels(["A", "B"], (0, 1))

    with pytest.raises(ValueError):
        validate.node_labels(["A"], (0, 1))
    with pytest.raises(ValueError):
        validate.node_labels(["A", "B"], (0,))
    with pytest.raises(ValueError):
        validate.node_labels(["A", "A"], (0, 1))


def test_validate_relata_empty():
    with pytest.raises(ValueError):
        validate.relata([])


def test_validate_relata_nonempty():
    validate.relata([object()])
