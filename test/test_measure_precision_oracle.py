"""B9 — high-precision oracle for the φ measure primitives.

The IIT 4.0 small-φ / ii path is built on ``information_density``
(``p·log₂(p/q)``) and the pointwise mutual information (``log₂(p/q)``), the
``log₂`` factors of which were a catastrophic-cancellation suspect when
``p ≈ q`` (the result is tiny) or near the 0/1 boundary.

This module answers that with a test rather than a caveat: it recomputes the
primitives in 60-digit ``decimal`` arithmetic and asserts the production float64
values agree, across exactly those adversarial regimes. Empirically the
agreement is ≤ ~1 ULP relative — because the primitives compute the ratio
``p/q`` *first* and take a single ``log₂`` (via ``scipy.special.rel_entr`` for
the density), so there is no subtraction of nearly-equal logs to cancel. The
guard's tolerance (``1e-10`` relative) is far tighter than any real value yet
loose enough never to flake, so a future refactor that reintroduced a
``log2(p) - log2(q)`` formulation (relative error ~1e-6 for ``p≈q``) would fail
here.

Because the primitives are well-conditioned, no runtime condition-number guard
or ``precision``-trust warning is warranted; this test is the standing evidence
for that decision.
"""

from __future__ import annotations

from decimal import Decimal
from decimal import getcontext

import numpy as np
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st

from pyphi.measures import distribution as measures

getcontext().prec = 60
_LN2 = Decimal(2).ln()
_REL_TOL = 1e-10
_ABS_TOL = 1e-12


def _dec_log2_ratio(p: float, q: float) -> Decimal:
    return (Decimal(p) / Decimal(q)).ln() / _LN2


def _dec_information_density(p: np.ndarray, q: np.ndarray) -> list[Decimal | None]:
    """p_i·log₂(p_i/q_i), with the 0/0-boundary convention (0 when p_i=0);
    None marks q_i=0<p_i (production gives +inf, excluded from the check)."""
    out: list[Decimal | None] = []
    for pi, qi in zip(p.tolist(), q.tolist(), strict=True):
        if pi <= 0.0:
            out.append(Decimal(0))
        elif qi <= 0.0:
            out.append(None)
        else:
            out.append(Decimal(pi) * _dec_log2_ratio(pi, qi))
    return out


def _agrees(actual: float, expected: Decimal) -> bool:
    a, e = float(actual), float(expected)
    return abs(a - e) <= _ABS_TOL + _REL_TOL * abs(e)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.abs(x) + 1e-18
    return x / x.sum()


@st.composite
def _adversarial_pq(draw):
    """A (p, q) distribution pair biased toward the dangerous regimes:
    q ≈ p (near-equal at a drawn scale) and 0/1-boundary entries."""
    n = draw(st.integers(2, 5))
    p = _normalize(np.array(draw(st.lists(st.floats(1e-6, 1.0), min_size=n, max_size=n))))
    mode = draw(st.sampled_from(["near_equal", "boundary", "random"]))
    if mode == "near_equal":
        scale = 10.0 ** draw(st.integers(-13, -3))
        delta = np.array(
            draw(st.lists(st.floats(-1, 1), min_size=n, max_size=n)), dtype=float
        )
        q = _normalize(p + delta * scale)
    elif mode == "boundary":
        q = p.copy()
        q[0] = 10.0 ** draw(st.integers(-15, -6))
        q = _normalize(q)
    else:
        q = _normalize(np.array(draw(st.lists(st.floats(1e-6, 1.0), min_size=n, max_size=n))))
    return p, q


class TestInformationDensityPrecision:
    @settings(max_examples=300, deadline=None)
    @given(pq=_adversarial_pq())
    def test_float64_matches_decimal_oracle(self, pq) -> None:
        p, q = pq
        f64 = measures.information_density(p, q)
        oracle = _dec_information_density(p, q)
        for a, e in zip(np.asarray(f64).tolist(), oracle, strict=True):
            if e is None or not np.isfinite(a):
                continue
            assert _agrees(a, e), f"info_density drift: f64={a!r} oracle={float(e)!r}"

    @pytest.mark.parametrize("scale", [1e-3, 1e-6, 1e-9, 1e-12])
    def test_near_equal_is_well_conditioned(self, scale: float) -> None:
        """The tiny ``p≈q`` result keeps full relative precision (no
        log-difference cancellation)."""
        rng = np.random.default_rng(0)
        p = _normalize(rng.random(4))
        q = _normalize(p + scale * rng.standard_normal(4))
        f64 = np.asarray(measures.information_density(p, q)).tolist()
        for a, e in zip(f64, _dec_information_density(p, q), strict=True):
            if e is None or not np.isfinite(a):
                continue
            assert _agrees(a, e), f"near-equal drift at scale {scale}: {a!r} vs {float(e)!r}"


class TestPointwiseMutualInformationPrecision:
    @settings(max_examples=200, deadline=None)
    @given(pq=_adversarial_pq())
    def test_pmi_float64_matches_decimal_oracle(self, pq) -> None:
        p, q = pq
        assume(np.all(p > 0) and np.all(q > 0))
        f64 = np.asarray(measures.pointwise_mutual_information_vector(p, q)).tolist()
        for a, pi, qi in zip(f64, p.tolist(), q.tolist(), strict=True):
            e = _dec_log2_ratio(pi, qi)
            assert _agrees(a, e), f"PMI drift: f64={a!r} oracle={float(e)!r}"
