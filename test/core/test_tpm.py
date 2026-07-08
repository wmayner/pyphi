"""FactoredTPM connectivity inference."""

import numpy as np


def test_infer_cm(rule152):
    assert np.array_equal(rule152.tpm.infer_cm(), rule152.cm)
