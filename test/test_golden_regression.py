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

from pyphi.models.cmp import EQUALITY_TOLERANCE

from .golden import ALL_FIXTURES
from .golden import GoldenFixture
from .golden.compute import compute_all_layers
from .golden.fixture import deref_array_ref
from .golden.fixture import is_array_ref
from .golden.fixture import load_fixture
from .golden.fixture import store_fixture

# Tolerances for array comparisons. Shared with ``pyphi.models.cmp``'s
# ``numpy_aware_eq`` (the production-side structural-equality comparator)
# so the test-fixture comparator and the model-level ``__eq__`` ask the
# same question: did this value drift by more than float64 op-order noise?
#
# Platform sensitivity note: fixtures committed to the repository were
# generated on macOS aarch64 (Apple Silicon) with Python 3.13.13, NumPy
# 2.4, SciPy 1.17, and pyemd 2.0. Linux x86_64 may produce drift below
# 1e-13 due to BLAS implementation differences (Accelerate vs OpenBLAS)
# and EMD library backend variations. If CI fails on Linux/Windows with
# sub-1e-10 differences and the structural fields (partitions, mechanisms,
# distinction counts) match exactly, raise the constant in ``cmp.py`` and
# document. Differences above 1e-10 should be investigated as potential
# bugs.
RTOL = EQUALITY_TOLERANCE
ATOL = EQUALITY_TOLERANCE


def _marks_for(fixture: GoldenFixture) -> list[pytest.MarkDecorator]:
    marks: list[pytest.MarkDecorator] = []
    if fixture.slow:
        marks.append(pytest.mark.slow)
    return marks


def _golden_params() -> list[Any]:
    """Build parametrize entries, attaching per-fixture markers."""
    return [pytest.param(f, marks=_marks_for(f)) for f in ALL_FIXTURES]


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

    _assert_matches(structured, expected_structured, arrays, expected_arrays, fixture)


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


# ============== Coverage guardrails ==============


def test_canonical_iit3_preset_is_exercised() -> None:
    """At least one golden fixture must exercise the canonical IIT 3.0 preset.

    Without this guardrail, a future config rename or default change could
    silently remove EMD CES distance coverage from regression testing — which
    is exactly how the EMD path went unmaintained from 2023 through 2026.
    """
    from pyphi.conf import presets

    canonical = presets.iit3["iit"]
    matching = [
        f
        for f in ALL_FIXTURES
        if isinstance(f.config_overrides.get("iit"), type(canonical))
        and f.config_overrides["iit"].version == canonical.version
        and f.config_overrides["iit"].ces_measure == canonical.ces_measure
        and f.config_overrides["iit"].mechanism_phi_measure
        == canonical.mechanism_phi_measure
    ]
    assert matching, (
        "No fixture in ALL_FIXTURES exercises the canonical IIT 3.0 preset "
        f"(version={canonical.version}, ces_measure={canonical.ces_measure}, "
        f"mechanism_phi_measure={canonical.mechanism_phi_measure}). The "
        "EMD CES distance code path will go unmaintained again without at "
        "least one fixture covering it."
    )


# ============== Helpers ==============


def _load_canonical(name: str) -> dict:
    """Load a canonical-reference JSON from test/data/iit3-canonical/."""
    import json
    from pathlib import Path

    return json.loads(Path(f"test/data/iit3-canonical/{name}.json").read_text())


def _load_fixture(name: str) -> dict:
    """Load a stored golden fixture JSON from test/data/golden/v1/."""
    import json
    from pathlib import Path

    return json.loads(Path(f"test/data/golden/v1/{name}.json").read_text())


def test_basic_iit3_emd_sia_phi_matches_canonical_reference() -> None:
    """basic_iit3_emd's sia.phi matches the canonical reference."""
    fixture = _load_fixture("basic_iit3_emd")
    canonical = _load_canonical("basic_sia_phi_canonical")
    fixture_phi = fixture["sia"]["phi"]
    canonical_phi = canonical["canonical_target"]["sia_phi"]
    tolerance = canonical["canonical_target"]["tolerance"]
    assert abs(fixture_phi - canonical_phi) < tolerance, (
        f"basic_iit3_emd sia.phi {fixture_phi} does not match canonical "
        f"{canonical_phi} (tolerance {tolerance})"
    )


def test_basic_iit3_emd_tri_sia_phi_matches_canonical_reference() -> None:
    """basic_iit3_emd_tri's sia.phi matches the WEDGE_TRIPARTITION canonical reference."""
    fixture = _load_fixture("basic_iit3_emd_tri")
    canonical = _load_canonical("basic_tri_sia_phi_canonical")
    fixture_phi = fixture["sia"]["phi"]
    canonical_phi = canonical["canonical_target"]["sia_phi"]
    tolerance = canonical["canonical_target"]["tolerance"]
    assert abs(fixture_phi - canonical_phi) < tolerance, (
        f"basic_iit3_emd_tri sia.phi {fixture_phi} does not match canonical "
        f"{canonical_phi} (tolerance {tolerance})"
    )


@pytest.mark.parametrize(
    ("fixture_name", "canonical_name"),
    [
        ("rule110_iit3_emd", "rule110_sia_phi_canonical"),
        ("grid3_iit3_emd", "grid3_sia_phi_canonical"),
    ],
)
def test_iit3_emd_sia_phi_matches_pyphi_1x_reference(
    fixture_name: str, canonical_name: str
) -> None:
    """rule110/grid3 IIT 3.0 EMD sia.phi match the independent PyPhi 1.2.0 reference.

    Unlike ``basic``, these substrates are not published worked examples, so their
    only independent anchor is a genuine PyPhi 1.x IIT 3.0 SIA. The reference values
    were generated by ``scripts/gen_iit3_emd_oracle.py`` against pyphi==1.2.0 (whose
    correctness is itself controlled by reproducing the anchored ``basic`` value of
    2.3125). The sia.phi here depends on the load-bearing ``PURVIEW_SIZE`` purview
    tie-break; the reference pins the larger-purview resolution that both 2.0 and
    PyPhi 1.x's default produce.
    """
    fixture = _load_fixture(fixture_name)
    canonical = _load_canonical(canonical_name)
    fixture_phi = fixture["sia"]["phi"]
    canonical_phi = canonical["canonical_target"]["sia_phi"]
    tolerance = canonical["canonical_target"]["tolerance"]
    assert abs(fixture_phi - canonical_phi) < tolerance, (
        f"{fixture_name} sia.phi {fixture_phi} does not match the PyPhi 1.2.0 "
        f"reference {canonical_phi} (tolerance {tolerance})"
    )


def test_basic_iit3_emd_mechanism_mip_partitions_match_canonical_reference() -> None:
    """basic_iit3_emd's MIC mechanism-MIP partitions match the canonical reference.

    Detects mechanism-MIP regressions that don't propagate to sia.phi.
    """
    fixture = _load_fixture("basic_iit3_emd")
    canonical = _load_canonical("basic_sia_phi_canonical")
    if "mechanism_mip_partitions" not in canonical["canonical_target"]:
        import pytest

        pytest.skip("canonical reference does not pin mechanism_mip_partitions")
    expected = canonical["canonical_target"]["mechanism_mip_partitions"]
    expected_keys = {k for k in expected if k != "comment"}

    # Build the actual MIC map from the fixture's mechanism_mips list
    by_md: dict = {}
    for m in fixture["mechanism_mips"]:
        key = (tuple(m["mechanism"]), m["direction"])
        if key not in by_md or m["phi"] > by_md[key]["phi"]:
            by_md[key] = m

    actual = {}
    for (mech, direction), mip in by_md.items():
        canonical_key = f"({','.join(str(i) for i in mech)},)|{direction}"
        actual[canonical_key] = {
            "purview": list(mip["purview"]),
            "phi": mip["phi"],
            "partition": mip["partition"],
        }

    missing = expected_keys - actual.keys()
    assert not missing, f"Canonical keys missing from fixture: {missing}"
    for key in sorted(expected_keys):
        assert actual[key]["partition"] == expected[key]["partition"], (
            f"Mechanism-MIP partition mismatch at {key}:\n"
            f"  fixture:   {actual[key]['partition']}\n"
            f"  canonical: {expected[key]['partition']}"
        )


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
