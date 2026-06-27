import numpy as np

from pyphi import Substrate
from pyphi import examples
from pyphi.conf import config

# Raw-numpy substrates below exercise the fingerprint digest only, so
# connectivity validation (which would reject a cm that under-specifies the
# TPM's implied edges) is irrelevant and disabled.
_NO_CM_VALIDATION = {"validate_connectivity": False, "welcome_off": True}


def test_cm_fingerprint_ignores_tpm_weights():
    # Same connectivity, different TPM weights -> equal cm fingerprint.
    cm = np.array([[1, 1], [1, 1]])
    tpm_a = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    tpm_b = np.array([[0.9, 0.8], [0.7, 0.6], [0.5, 0.4], [0.3, 0.2]])
    with config.override(**_NO_CM_VALIDATION):
        a = Substrate(tpm_a, cm)
        b = Substrate(tpm_b, cm)
    assert a._cm_fingerprint == b._cm_fingerprint
    assert a._math_fingerprint != b._math_fingerprint  # TPM differs


def test_cm_fingerprint_separates_topologies():
    tpm = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    with config.override(**_NO_CM_VALIDATION):
        a = Substrate(tpm, np.array([[1, 1], [0, 1]]))
        b = Substrate(tpm, np.array([[1, 1], [1, 1]]))
    assert a._cm_fingerprint != b._cm_fingerprint


def test_substrate_fingerprint_ignores_labels():
    s = examples.basic_substrate()
    relabeled = Substrate.from_factored(
        s.factored_tpm, cm=s.cm, node_labels=("X", "Y", "Z")
    )
    assert relabeled == s  # same math
    assert s._math_fingerprint == relabeled._math_fingerprint
    assert s._cm_fingerprint == relabeled._cm_fingerprint


def test_fingerprint_is_32_bytes_and_deterministic():
    s = examples.basic_substrate()
    assert len(s._math_fingerprint) == 32
    assert len(s._cm_fingerprint) == 32
    assert s._math_fingerprint == examples.basic_substrate()._math_fingerprint
