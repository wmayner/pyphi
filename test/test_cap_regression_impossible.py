"""The cap-regression class of bug is impossible by construction.

A prior bug was discovered where setting ``mechanism_phi_measure`` to
``INTRINSIC_INFORMATION`` silently failed to activate the Eq. 23 ``ii(s)``
cap (the cap is gated on ``system_phi_measure``, not on
``mechanism_phi_measure``). The original fix was test-side; the
subsequent refactor makes the underlying architecture immune to the bug
class by requiring callers to pass metric callables explicitly.

This test pins the new invariant: raw module-level ``sia()`` requires
explicit metric kwargs. Setting ``mechanism_phi_measure`` and calling
raw ``sia()`` without explicit metrics produces a ``TypeError`` (missing
kwarg), not a silent wrong-scope read.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import System
from pyphi import config
from pyphi.formalism.iit4 import sia
from pyphi.metrics.distribution import composite_metrics


@pytest.fixture
def noisy_copy_system():
    """2-node noisy COPY, p=0.8, state (1, 1).

    Matches the canonical fixture used by
    :class:`TestEq23IntrinsicInformationCap` so ground-truth values are
    directly comparable.
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


def test_raw_sia_requires_explicit_system_metric(noisy_copy_system):
    """The cap-regression bug shape now raises TypeError.

    Setting ``mechanism_phi_measure`` to ``INTRINSIC_INFORMATION`` and
    calling raw ``sia()`` used to silently miss the Eq. 23 cap. After
    the refactor, ``sia()`` requires ``system_metric`` as an explicit
    keyword argument, so the same call now fails loudly.
    """
    with (
        config.override(mechanism_phi_measure="INTRINSIC_INFORMATION"),
        pytest.raises(TypeError, match="system_metric"),
    ):
        sia(noisy_copy_system)


def test_raw_sia_with_explicit_metrics_works(noisy_copy_system):
    """Sanity: with explicit metrics, raw ``sia()`` works.

    Callers must declare which metric they want at the call site; no
    implicit fallback to config. Under ``INTRINSIC_INFORMATION`` the
    Eq. 23 ``ii(s)`` cap fires and phi is ~0.644 for the canonical
    noisy-COPY fixture.
    """
    result = sia(
        noisy_copy_system,
        system_metric=composite_metrics["INTRINSIC_INFORMATION"],
        specification_metric=composite_metrics["INTRINSIC_SPECIFICATION"],
    )
    assert float(result.phi) == pytest.approx(0.644, abs=0.001)
