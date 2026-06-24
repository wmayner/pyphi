import numpy as np
import pytest

from pyphi import Direction
from pyphi import Substrate
from pyphi import System
from pyphi import exceptions
from pyphi import validate
from pyphi.core.tpm.joint_distribution import JointTPM


def test_validate_direction():
    validate.direction(Direction.CAUSE)
    validate.direction(Direction.EFFECT)

    with pytest.raises(ValueError):
        validate.direction("dogeeeee")

    validate.direction(Direction.BIDIRECTIONAL, allow_bi=True)
    with pytest.raises(ValueError):
        validate.direction(Direction.BIDIRECTIONAL)


def test_validate_tpm_wrong_shape():
    with pytest.raises(ValueError):
        tpm = JointTPM(np.arange(3**3).reshape(3, 3, 3))
        assert tpm.validate()


def test_validate_tpm_nonbinary_nodes():
    with pytest.raises(ValueError):
        tpm = JointTPM(np.arange(3 * 3 * 2).reshape(3, 3, 2))
        assert tpm.validate()


def test_validate_tpm_conditional_independence():
    # fmt: off
    tpm = JointTPM(
        np.array([
            [1, 0.0, 0.0, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.5, 0.5, 0],
            [0, 0.0, 0.0, 1],
        ])
    )
    # fmt: on
    with pytest.raises(exceptions.ConditionallyDependentError):
        tpm.conditionally_independent()
    with pytest.raises(exceptions.ConditionallyDependentError):
        tpm.validate()
    tpm.validate(check_independence=False)


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


def test_validate_state_subsystem_unreachable(s):
    """Subsystem-level reachability: state component must be in image of
    background-conditioned subsystem dynamics.

    For the standard substrate ``s`` in state ``(1, 0, 0)``, the singleton
    subsystem ``{A}`` cannot have A=1 because the conditioned dynamics
    (B=0, C=0) deterministically produce A_next = OR(B,C) = 0. Likewise
    ``{C}`` cannot have C=0 because conditioned C_next = XOR(A=1,B=0) = 1.
    ``{B}`` passes because conditioned B_next = COPY(C=0) = 0 matches B=0.
    """
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(s.substrate, s.state, (0,))
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(s.substrate, s.state, (2,))
    # No raise for B:
    System(s.substrate, s.state, (1,))


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
    conditioned dynamics.
    """
    sub = _k3_copy_substrate()
    # Full substrate: reachable, no raise.
    System(sub, (1, 0), (0, 1))
    # Subsystem {0}: node-0 conditioned dynamics cannot produce 1.
    with pytest.raises(exceptions.StateUnreachableForwardsError):
        System(sub, (1, 0), (0,))
    # Subsystem {1}: node 1 is constant 0, so {1}=0 is reachable.
    System(sub, (1, 0), (1,))


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
