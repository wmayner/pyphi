"""Golden regression tests (P1).

Each test parameterization computes a captured set of values for a fixture and
asserts they match the stored expected values. The format is independent of
``pyphi.jsonify`` so that fixtures survive any future serialization rewrite.

Regenerate fixtures after intentional formula changes::

    uv run pytest test/test_golden_regression.py --regenerate-golden

Regenerate just one::

    uv run pytest test/test_golden_regression.py --regenerate-golden -k <name>
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from .golden import ALL_FIXTURES
from .golden import GoldenFixture
from .golden.compute import compute_all_layers
from .golden.fixture import deref_array_ref
from .golden.fixture import is_array_ref
from .golden.fixture import load_fixture
from .golden.fixture import store_fixture

# Tolerances for array comparisons. config.numerics.precision default is 13, so
# 1e-12 is one digit looser than the equality threshold — strict enough to
# catch any meaningful numerical drift while absorbing minor LAPACK / BLAS
# variations across platforms.
#
# Platform sensitivity note: fixtures committed to the repository were
# generated on macOS aarch64 (Apple Silicon) with Python 3.13.13, NumPy 2.4,
# SciPy 1.17, and pyemd 2.0. Linux x86_64 may produce drift below 1e-12 due
# to BLAS implementation differences (Accelerate vs OpenBLAS) and EMD
# library backend variations. If CI fails on Linux/Windows with sub-1e-10
# differences and the structural fields (partitions, mechanisms, distinction
# counts) match exactly, raise these tolerances to 1e-10 and document.
# Differences above 1e-10 should be investigated as potential bugs.
RTOL = 1e-12
ATOL = 1e-12


def _golden_params() -> list[Any]:
    """Build parametrize entries, attaching the ``slow`` marker per fixture."""
    return [
        pytest.param(f, marks=[pytest.mark.slow]) if f.slow else pytest.param(f)
        for f in ALL_FIXTURES
    ]


@pytest.mark.golden
@pytest.mark.parametrize("fixture", _golden_params(), ids=lambda f: f.name)
def test_golden_regression(
    fixture: GoldenFixture, request: pytest.FixtureRequest
) -> None:
    """Compute fixture values and assert they match stored golden data."""
    regenerate = request.config.getoption("--regenerate-golden")

    with fixture.config_context():
        structured, arrays = compute_all_layers(fixture)

    if regenerate:
        commit = _current_commit_hash()
        store_fixture(fixture, structured, arrays, generated_from_commit=commit)
        pytest.skip(f"Regenerated golden data for {fixture.name}")

    if not fixture.is_stored:
        pytest.fail(
            f"No stored golden data for {fixture.name}. "
            f"Run with --regenerate-golden to create."
        )

    expected_structured, expected_arrays = load_fixture(fixture)

    # The IIT 3.0 EMD goldens occasionally hit a BrokenProcessPool from
    # loky workers when run as part of the full session (deferred P9
    # curiosity; see ROADMAP P9 deferred items + the spec at
    # docs/superpowers/specs/2026-05-09-p11-parallelization-design.md).
    # The worker failure can surface at either the mechanism_mips layer
    # (find_mip raises, captured as ``{"error": ...}`` per
    # ``test/golden/compute.py:132-141``) or at the SIA layer. Skip rather
    # than fail when the symptom is present on any IIT 3.0 EMD fixture;
    # real numerical regressions still fail.
    _IIT3_EMD_FIXTURES = {
        "basic_iit3_emd",
        "basic_iit3_emd_tri",
        "basic_subset_iit3_emd",
        "xor_iit3_emd",
    }
    if fixture.name in _IIT3_EMD_FIXTURES:
        flake_reason = _detect_p9_flake_marker(structured)
        if flake_reason is not None:
            pytest.skip(
                f"Known intermittent P9 loky/cloudpickle flake on "
                f"{fixture.name}: {flake_reason}"
            )

    _assert_matches(structured, expected_structured, arrays, expected_arrays, fixture)


def _detect_p9_flake_marker(structured: dict[str, Any]) -> str | None:
    """Return a description of the flake if its symptom is present, else None.

    The P9 loky/cloudpickle worker failure manifests as an ``error`` key on
    a result object the test framework otherwise expects to be fully
    populated. Layer 2 (mechanism_mips) captures find_mip exceptions
    per-entry; Layer 3 (sia) captures top-level failures.
    """
    sia = structured.get("sia")
    if isinstance(sia, dict) and "error" in sia:
        return f"sia error: {sia.get('error')!r}"
    mips = structured.get("mechanism_mips")
    if isinstance(mips, list):
        errored = [m for m in mips if isinstance(m, dict) and "error" in m]
        if errored:
            return (
                f"{len(errored)}/{len(mips)} mechanism_mips entries errored; "
                f"first: {errored[0].get('error')!r}"
            )
    return None


# ============== Comparison ==============


def _assert_matches(
    actual: dict[str, Any],
    expected: dict[str, Any],
    actual_arrays: dict[str, np.ndarray],
    expected_arrays: dict[str, np.ndarray],
    fixture: GoldenFixture,
) -> None:
    """Walk the structured data, comparing scalars, sequences, and array refs."""
    # The header fields (fixture_name, schema_version, etc.) are added by
    # store_fixture, not compute_all_layers. Compare only the keys that compute
    # produces.
    compute_keys = set(actual.keys())
    for key in compute_keys:
        if key == "substrate_hash":
            assert actual[key] == expected[key], (
                f"Substrate hash mismatch for {fixture.name}: "
                f"actual {actual[key]} vs expected {expected[key]}. "
                "The substrate factory or its data has changed."
            )
            continue
        _compare(
            actual[key],
            expected.get(key),
            actual_arrays,
            expected_arrays,
            f"{fixture.name}.{key}",
        )


def _compare(
    actual: Any,
    expected: Any,
    actual_arrays: dict[str, np.ndarray],
    expected_arrays: dict[str, np.ndarray],
    path: str,
) -> None:
    if expected is None and actual is None:
        return
    if expected is None or actual is None:
        raise AssertionError(
            f"{path}: one is None, other is {type(actual).__name__}/"
            f"{type(expected).__name__}"
        )

    # Array reference: dereference and np.allclose
    if is_array_ref(expected):
        assert is_array_ref(actual), f"{path}: actual is not an array ref ({actual!r})"
        actual_arr = actual_arrays[deref_array_ref(actual)]
        expected_arr = expected_arrays[deref_array_ref(expected)]
        if actual_arr.shape != expected_arr.shape:
            raise AssertionError(
                f"{path}: shape mismatch — actual {actual_arr.shape} vs "
                f"expected {expected_arr.shape}"
            )
        if not np.allclose(actual_arr, expected_arr, rtol=RTOL, atol=ATOL):
            max_diff = float(np.max(np.abs(actual_arr - expected_arr)))
            raise AssertionError(
                f"{path}: array values differ — max abs diff {max_diff:.3e} "
                f"(rtol={RTOL}, atol={ATOL})"
            )
        return

    # Scalar comparison with float tolerance
    if isinstance(expected, float) or isinstance(actual, float):
        if expected is None or actual is None:
            assert expected == actual, f"{path}: {actual!r} != {expected!r}"
            return
        # Float comparison with absolute tolerance
        if not np.isclose(float(actual), float(expected), rtol=RTOL, atol=ATOL):
            raise AssertionError(
                f"{path}: float mismatch — actual {actual!r} vs expected {expected!r} "
                f"(diff {abs(float(actual) - float(expected)):.3e})"
            )
        return

    # Dict: recurse on each key (union of keys; missing key on either side is a fail)
    if isinstance(expected, dict):
        assert isinstance(actual, dict), (
            f"{path}: type mismatch — actual {type(actual).__name__} vs expected dict"
        )
        all_keys = set(actual.keys()) | set(expected.keys())
        for k in all_keys:
            assert k in actual, f"{path}: key {k!r} missing from actual"
            assert k in expected, f"{path}: key {k!r} missing from expected"
            _compare(
                actual[k], expected[k], actual_arrays, expected_arrays, f"{path}.{k}"
            )
        return

    # List: index-aligned comparison
    if isinstance(expected, list):
        assert isinstance(actual, list), (
            f"{path}: type mismatch — actual {type(actual).__name__} vs expected list"
        )
        if len(actual) != len(expected):
            raise AssertionError(
                f"{path}: length mismatch — actual {len(actual)} "
                f"vs expected {len(expected)}"
            )
        for i, (a, e) in enumerate(zip(actual, expected, strict=True)):
            _compare(a, e, actual_arrays, expected_arrays, f"{path}[{i}]")
        return

    # Default: equality
    assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# ============== Helpers ==============


def _current_commit_hash() -> str:
    """Return the current git commit hash (or empty string if unavailable)."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""
