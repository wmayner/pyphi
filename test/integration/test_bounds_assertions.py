"""Runtime bound-certificate assertions (B1).

Tests the ``validate_phi_bounds`` machinery: ``check_phi_bound`` raises
``BoundViolationError`` only for an in-domain overshoot, and is a silent no-op
outside the certified domain (flag off, non-binary, out-of-domain config,
uncertified bound). End-to-end, a real IIT 4.0 pipeline stays within bounds,
and a deliberately-shrunk ceiling is caught at the SIA construction site.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyphi import Substrate
from pyphi import System
from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets
from pyphi.formalism.iit4 import bounds
from pyphi.formalism.iit4.bounds import BoundViolationError
from pyphi.formalism.iit4.bounds import UpperBound
from pyphi.formalism.iit4.bounds import check_phi_bound


def _binary_system() -> System:
    return System(examples.basic_substrate(), (1, 0, 0))


def _kary_system() -> System:
    """2-node fully-connected substrate with alphabet size 3 (non-binary)."""
    rng = np.random.default_rng(2026)
    factors = []
    for _ in range(2):
        f = rng.uniform(size=(3, 3, 3))
        f /= f.sum(axis=-1, keepdims=True)
        factors.append(f)
    return System(Substrate(marginals=factors, state_space=(0, 1, 2)), (0, 0))


def _ceiling(value: float, *, certified: bool = True):
    """A thunk returning a fixed certified (or not) UpperBound."""
    return lambda: UpperBound(
        value=value, certified=certified, assumptions=("test",), citation="test"
    )


# ---- check_phi_bound gating ------------------------------------------------


def test_raises_on_in_domain_overshoot() -> None:
    with (
        config.override(**presets.iit4_2023, validate_phi_bounds=True),
        pytest.raises(BoundViolationError),
    ):
        check_phi_bound(1.0, _ceiling(0.0), system=_binary_system(), label="x")


def test_passes_within_bound() -> None:
    with config.override(**presets.iit4_2023, validate_phi_bounds=True):
        check_phi_bound(0.5, _ceiling(0.5), system=_binary_system(), label="x")
        # within float tolerance of the ceiling
        check_phi_bound(0.5 + 1e-15, _ceiling(0.5), system=_binary_system(), label="x")


def test_noop_when_flag_off() -> None:
    with config.override(**presets.iit4_2023, validate_phi_bounds=False):
        check_phi_bound(1.0, _ceiling(0.0), system=_binary_system(), label="x")


def test_skips_non_binary_system() -> None:
    # k-ary phi can legitimately exceed the binary |M||Z| bound, so the check
    # must skip non-binary systems rather than false-flag them.
    with config.override(**presets.iit4_2023, validate_phi_bounds=True):
        check_phi_bound(1.0, _ceiling(0.0), system=_kary_system(), label="x")


def _macro_system():
    """A genuine 2-unit macro coarse-graining (mirrors ``_cg_macro_system`` in
    ``test_macro_system``). ``examples.macro_system()`` is a plain binary
    ``System`` over a pre-built substrate — not a ``MacroSystem`` — so it is
    not the type the macro skip targets."""
    from pyphi.macro import MacroSystem
    from pyphi.macro.units import MacroUnit
    from pyphi.macro.units import coarse_grain
    from test.macro.test_macro_tpm import CG_TPM

    units = (
        MacroUnit((0, 1), 1, coarse_grain(2, on_counts={2})),
        MacroUnit((2, 3), 1, coarse_grain(2, on_counts={2})),
    )
    return MacroSystem.from_micro(Substrate(CG_TPM), units, ((0, 0, 0, 0),))


def test_skips_macro_system() -> None:
    # A macro coarse-graining's phi_s derives from its micro constituents, so
    # the macro-unit n(n-1) ceiling does not apply (a single macro unit can
    # have phi_s > 0 while n(n-1) = 0). The check must skip it; here a value
    # that would otherwise violate the ceiling must not raise.
    macro = _macro_system()
    with config.override(**presets.iit4_2023, validate_phi_bounds=True):
        check_phi_bound(1e9, _ceiling(0.0), system=macro, label="x")


def test_skips_out_of_domain_config() -> None:
    # Under IIT 3.0 the real bound thunk raises ValueError (out of the
    # certified version/measure domain); the check must swallow it.
    with config.override(**presets.iit3, validate_phi_bounds=True):
        check_phi_bound(
            1e9,
            lambda: bounds.distinction_phi_upper_bound((0,), (1,)),
            system=_binary_system(),
            label="x",
        )


def test_skips_uncertified_bound() -> None:
    with config.override(**presets.iit4_2023, validate_phi_bounds=True):
        check_phi_bound(
            1.0, _ceiling(0.0, certified=False), system=_binary_system(), label="x"
        )


# ---- end-to-end ------------------------------------------------------------


def test_iit4_pipeline_within_bounds() -> None:
    """A real IIT 4.0 SIA computes clean with validation on (no overshoot)."""
    with config.override(**presets.iit4_2023, validate_phi_bounds=True):
        sia = _binary_system().sia()
    assert sia.phi is not None


def test_sia_overshoot_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrinking the system ceiling to 0 makes the (positive) SIA phi trip the
    assertion at the SIA construction site — proving the wiring is live."""
    monkeypatch.setattr(
        bounds,
        "system_phi_upper_bound",
        lambda _n: UpperBound(
            value=0.0, certified=True, assumptions=("test",), citation="test"
        ),
    )
    with (
        config.override(**presets.iit4_2023, validate_phi_bounds=True),
        pytest.raises(BoundViolationError),
    ):
        _binary_system().sia()
