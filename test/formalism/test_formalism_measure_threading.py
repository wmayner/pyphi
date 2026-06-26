"""Verify explicit metric kwargs override formalism defaults.

Pin the architectural invariant: passing an explicit metric to a
formalism method actually uses that metric, with no implicit override
from ``default_*_metric`` ClassVars. This is the structural guarantee
that makes the cap-regression class of bug impossible by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import System
from pyphi.formalism.iit4.formalism import IIT4_2023Formalism
from pyphi.formalism.iit4.formalism import IIT4_2026Formalism
from pyphi.measures.distribution import composite_measures


@pytest.fixture
def noisy_copy_system():
    """2-node noisy COPY system, state (1, 1).

    Same fixture as ``TestEq23IntrinsicInformationCap``. At p=0.8:
    GID(MIP) ~ 0.868, i_diff ~ 0.644, cap (if applied) ~ 0.644.
    """
    p = 0.8
    tpm = np.array(
        [
            [1 - p, 1 - p],  # (0, 0)
            [1 - p, p],  # (1, 0)
            [p, 1 - p],  # (0, 1)
            [p, p],  # (1, 1)
        ]
    )
    cm = np.array([[0, 1], [1, 0]])
    substrate = Substrate(tpm, cm=cm, node_labels=["A", "B"])
    return System(substrate, (1, 1))


def test_2026_explicit_gid_skips_cap(noisy_copy_system):
    """Passing GID explicitly to 2026 ``evaluate_system`` -> no cap applied.

    The 2026 formalism's ``ii(s)`` cap (Eq. 23) is keyed off the resolved
    ``system_measure.name``. When the caller passes
    ``GENERALIZED_INTRINSIC_DIFFERENCE`` explicitly, the cap branch
    doesn't fire, so ``phi = GID(MIP) ~ 0.868``.
    """
    formalism = IIT4_2026Formalism()
    gid_metric = composite_measures["GENERALIZED_INTRINSIC_DIFFERENCE"]
    spec_metric = composite_measures["INTRINSIC_SPECIFICATION"]

    result = formalism.evaluate_system(
        noisy_copy_system,
        system_measure=gid_metric,
        specification_measure=spec_metric,
    )

    # Without cap: phi = GID(MIP) ~ 0.868
    assert float(result.phi) == pytest.approx(0.868, abs=0.001)


def test_2026_omitted_metric_uses_config(noisy_copy_system):
    """Omitting ``system_measure`` -> falls back to config -> cap fires.

    With ``config.formalism.iit.system_phi_measure="INTRINSIC_INFORMATION"``
    the cap binds: phi = min(GID(MIP), ii_s) = ii_s ~ 0.644.
    """
    from pyphi import config

    formalism = IIT4_2026Formalism()
    spec_metric = composite_measures["INTRINSIC_SPECIFICATION"]

    with config.override(
        system_phi_measure="INTRINSIC_INFORMATION",
        specification_measure="INTRINSIC_SPECIFICATION",
    ):
        result = formalism.evaluate_system(
            noisy_copy_system,
            specification_measure=spec_metric,
        )

    # With cap: phi = min(GID(MIP), ii_s) = ii_s ~ 0.644
    assert float(result.phi) == pytest.approx(0.644, abs=0.001)


def test_2023_omitted_metric_uses_default(noisy_copy_system):
    """IIT 4.0 (2023) formalism: default config -> default GID -> no cap.

    With default config ``system_phi_measure=GENERALIZED_INTRINSIC_DIFFERENCE``
    the 2023 formalism applies no cap: phi = GID(MIP) ~ 0.868.
    """
    formalism = IIT4_2023Formalism()
    spec_metric = composite_measures["INTRINSIC_SPECIFICATION"]

    result = formalism.evaluate_system(
        noisy_copy_system,
        specification_measure=spec_metric,
    )

    assert float(result.phi) == pytest.approx(0.868, abs=0.001)
