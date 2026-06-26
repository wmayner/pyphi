"""Standing regression tests for the IIT 3.0 code-path divergence audit (P11.95e).

The 2.0 refactor routes IIT 3.0 through code paths shared with IIT 4.0. This
module locks in the audit's confirmed result: **none of the shared paths that
carry IIT 4.0-faithful logic alter IIT 3.0 results.** Each test perturbs one
shared path to its 3.0-faithful alternative and asserts the captured fixture
output is unchanged — so a future change that lets 4.0 logic leak onto the 3.0
path fails here rather than silently moving a golden value.

The audited shared paths and their confirmed status:

- ``positive_part`` clamp (``models/ria.py``, Eqs. 19-20 ``|·|+``): inert — the
  3.0 EMD path never produces a negative integration value, so the clamp is a
  no-op (``test_clamp_*``).
- ``_compute_distinctions`` ``UnresolvedDistinctions`` wrapping (a 4.0
  specified-state concept): inert — ``ces_distance`` and the 3.0 SIA do not
  branch on the wrapper type (``test_distinctions_wrapping_is_inert``).
- ``mip``/``sia`` ``PARTITION_LEX`` secondary tie-break: value-inert — it only
  canonicalizes *which* equi-φ partition is reported, never a φ value
  (``test_mip_and_sia_lex_tiebreak_preserve_all_phi_values``).
- The 4.0 cascade resolvers (``resolve_state_tie`` etc.) and the legacy
  ``resolve_ties.states``: not on the 3.0 path at all
  (``test_iit3_path_never_invokes_4_0_cascade_resolvers``).

The audit also found that the *one* value-bearing tie-break on the 3.0 path is
``purview_tie_resolution``'s ``PURVIEW_SIZE`` key — which is the classic IIT 3.0
/ PyPhi 1.x "prefer larger purview" convention, **not** a 4.0 contaminant.
``test_purview_size_tiebreak_is_load_bearing`` pins that it genuinely affects
system φ, so a silent regression to PHI-only is caught.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from pyphi.conf import config
from pyphi.models import ResolvedDistinctions
from pyphi.models.cmp import EQUALITY_TOLERANCE
from test.golden import ALL_FIXTURES
from test.golden import GoldenFixture
from test.golden.compute import compute_all_layers
from test.golden.fixture import deref_array_ref
from test.golden.fixture import is_array_ref

pytestmark = pytest.mark.emd

TOL = EQUALITY_TOLERANCE

IIT3_FIXTURES = [f for f in ALL_FIXTURES if "iit3" in f.name]
# Fixtures whose system φ is sensitive to the purview tie-break (verified by
# the audit's disentangling sweep). On the others, every purview φ-tie is
# either absent or does not propagate to a distinct system φ.
PURVIEW_PHI_SENSITIVE = ("rule110_iit3_emd", "grid3_iit3_emd")

_BASELINE_CACHE: dict[str, tuple[dict[str, Any], dict[str, np.ndarray]]] = {}


def _run(fixture: GoldenFixture) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute the fixture's full captured output under its config context."""
    with fixture.config_context():
        return compute_all_layers(fixture)


def _baseline(fixture: GoldenFixture) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Memoized unperturbed compute. Call before applying any perturbation."""
    if fixture.name not in _BASELINE_CACHE:
        _BASELINE_CACHE[fixture.name] = _run(fixture)
    return _BASELINE_CACHE[fixture.name]


def _diff(  # noqa: PLR0911
    a: Any,
    a_arr: dict[str, np.ndarray],
    b: Any,
    b_arr: dict[str, np.ndarray],
    path: str = "",
) -> list[str]:
    """Return the paths at which two captured fixture outputs differ."""
    if is_array_ref(a) or is_array_ref(b):
        if not (is_array_ref(a) and is_array_ref(b)):
            return [f"{path}: array-ref mismatch {a!r} vs {b!r}"]
        xa, xb = a_arr[deref_array_ref(a)], b_arr[deref_array_ref(b)]
        if xa.shape != xb.shape:
            return [f"{path}: shape {xa.shape} vs {xb.shape}"]
        if not np.allclose(xa, xb, rtol=TOL, atol=TOL):
            return [f"{path}: array max|Δ|={float(np.max(np.abs(xa - xb))):.3e}"]
        return []
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return [f"{path}: dict-type mismatch"]
        out: list[str] = []
        for k in set(a) | set(b):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: present on one side only")
            else:
                out += _diff(a[k], a_arr, b[k], b_arr, f"{path}.{k}")
        return out
    if isinstance(a, list) or isinstance(b, list):
        if not (isinstance(a, list) and isinstance(b, list)) or len(a) != len(b):
            return [f"{path}: list mismatch len {len(a)} vs {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += _diff(x, a_arr, y, b_arr, f"{path}[{i}]")
        return out
    if isinstance(a, float) or isinstance(b, float):
        if not np.isclose(float(a), float(b), rtol=TOL, atol=TOL):
            return [f"{path}: float {a!r} vs {b!r}"]
        return []
    return [] if a == b else [f"{path}: {a!r} vs {b!r}"]


def _value_diffs(diffs: list[str]) -> list[str]:
    """Diffs that touch a φ magnitude (not merely which equi-φ partition,
    repertoire, or state is reported)."""
    out = []
    for d in diffs:
        head = d.split(":")[0]
        leaf = head.rsplit(".", 1)[-1]
        is_value = leaf.endswith("phi") or "phi_sum" in head
        is_representative = any(
            tok in head
            for tok in ("partition", "repertoire", "specified_states", "system_state")
        )
        if is_value and not is_representative:
            out.append(d)
    return out


def _override_iit(fixture: GoldenFixture, **fields: Any):
    """Recompute the fixture with specific ``IITConfig`` fields replaced,
    leaving every other 3.0 setting intact."""
    with fixture.config_context():
        new_iit = dataclasses.replace(config.formalism.iit, **fields)
        with config.override(iit=new_iit):
            return compute_all_layers(fixture)


@pytest.mark.parametrize("fixture", IIT3_FIXTURES, ids=lambda f: f.name)
def test_clamp_sees_no_negative_input(
    fixture: GoldenFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ``|·|+`` clamp never sees a negative value on the 3.0 EMD path.

    If it never sees a negative input, it provably cannot change a result —
    the clamp is a no-op for IIT 3.0 distribution distances.
    """
    from pyphi import utils

    orig = utils.positive_part
    seen_min = float("inf")

    def recording(x: float) -> float:
        nonlocal seen_min
        seen_min = min(seen_min, float(x))
        return orig(x)

    monkeypatch.setattr("pyphi.utils.positive_part", recording)
    _run(fixture)

    assert seen_min >= -TOL, (
        f"{fixture.name}: positive_part saw a negative input ({seen_min}); the "
        "|·|+ clamp is no longer inert on the IIT 3.0 path — a 3.0 result may "
        "now silently depend on the 4.0 clamp."
    )


@pytest.mark.parametrize("fixture", IIT3_FIXTURES, ids=lambda f: f.name)
def test_clamp_removal_is_inert(
    fixture: GoldenFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing the ``|·|+`` clamp (identity) leaves every 3.0 value unchanged."""
    base = _baseline(fixture)
    monkeypatch.setattr("pyphi.utils.positive_part", lambda x: float(x))
    perturbed = _run(fixture)
    diffs = _diff(base[0], base[1], perturbed[0], perturbed[1])
    assert not diffs, f"{fixture.name}: clamp removal changed 3.0 output:\n" + "\n".join(
        diffs[:10]
    )


@pytest.mark.parametrize("fixture", IIT3_FIXTURES, ids=lambda f: f.name)
def test_distinctions_wrapping_is_inert(
    fixture: GoldenFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing the 4.0-motivated ``UnresolvedDistinctions`` wrapper with
    ``ResolvedDistinctions`` leaves every 3.0 value unchanged."""
    base = _baseline(fixture)
    monkeypatch.setattr(
        "pyphi.formalism.iit3.UnresolvedDistinctions", ResolvedDistinctions
    )
    perturbed = _run(fixture)
    diffs = _diff(base[0], base[1], perturbed[0], perturbed[1])
    assert not diffs, (
        f"{fixture.name}: Unresolved->Resolved distinctions changed 3.0 output:\n"
        + "\n".join(diffs[:10])
    )


@pytest.mark.parametrize("fixture", IIT3_FIXTURES, ids=lambda f: f.name)
def test_mip_and_sia_lex_tiebreak_preserve_all_phi_values(
    fixture: GoldenFixture,
) -> None:
    """Dropping the ``PARTITION_LEX`` secondary key from the mechanism-MIP and
    system-SIA tie-breaks may change *which* equi-φ partition is reported, but
    never a φ value. ``PARTITION_LEX`` is a reproducibility canonicalization,
    not a value-bearing decision."""
    base = _baseline(fixture)
    perturbed = _override_iit(
        fixture, mip_tie_resolution=["PHI"], sia_tie_resolution=["PHI"]
    )
    value_diffs = _value_diffs(_diff(base[0], base[1], perturbed[0], perturbed[1]))
    assert not value_diffs, (
        f"{fixture.name}: dropping the MIP/SIA lex tie-break changed a φ value — "
        "the lex canonicalization is no longer value-inert:\n" + "\n".join(value_diffs)
    )


def test_iit3_path_never_invokes_4_0_cascade_resolvers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full IIT 3.0 computation never calls the 4.0 specified-state /
    distinction-congruence cascade resolvers or the legacy
    ``resolve_ties.states``. The 3.0 path resolves ties only through the
    config-driven ``resolve_ties.{partitions,purviews,sias}``."""

    def boom(*_args: Any, **_kwargs: Any):
        raise AssertionError("a 4.0 tie resolver was called on the IIT 3.0 path")

    for name in (
        "states",
        "resolve_state_tie",
        "resolve_distinction_tie",
        "resolve_complex_tie",
    ):
        monkeypatch.setattr(f"pyphi.resolve_ties.{name}", boom)

    fixture = next(f for f in IIT3_FIXTURES if f.name == "basic_iit3_emd")
    # Exercises mechanism MIPs (purview + partition tie resolution) and the
    # full 3.0 SIA; must complete without hitting any patched resolver.
    _run(fixture)


@pytest.mark.parametrize(
    "fixture",
    [f for f in IIT3_FIXTURES if f.name in PURVIEW_PHI_SENSITIVE],
    ids=lambda f: f.name,
)
def test_purview_size_tiebreak_is_load_bearing(fixture: GoldenFixture) -> None:
    """The ``PURVIEW_SIZE`` purview tie-break genuinely determines system φ.

    This is the one value-bearing 3.0 tie-break (the classic "prefer larger
    purview" convention). Pinning that it *changes* system φ guards against a
    silent regression to PHI-only that would move these goldens without
    tripping the golden harness's value comparison being noticed as
    intentional.
    """
    base_phi = _baseline(fixture)[0]["sia"]["phi"]
    phi_only = _override_iit(fixture, purview_tie_resolution=["PHI"])[0]["sia"]["phi"]
    assert abs(base_phi - phi_only) > TOL, (
        f"{fixture.name}: dropping PURVIEW_SIZE no longer changes system φ "
        f"({base_phi} vs {phi_only}). The purview tie-break is expected to be "
        "load-bearing here; if this is intentional, update the audit and the "
        "iit3 preset documentation."
    )
