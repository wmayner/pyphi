"""
Test helpers for PyPhi golden test suite.

This module provides diagnostic utilities and comparison functions for
making tests more robust to refactoring. These helpers provide detailed
failure diagnostics and enable serialization-independent comparisons.
"""

from typing import Any

import pytest

from pyphi import config
from pyphi import utils
from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis
from pyphi.models.partitions import NullCut
from pyphi.models.partitions import SystemPartition


def compare_phi_values(
    actual: float, expected: float, tolerance: int | None = None
) -> tuple[bool, str]:
    """
    Compare two phi values using configured precision.

    Args:
        actual: Actual phi value
        expected: Expected phi value
        tolerance: Precision digits (uses config.numerics.precision if None)

    Returns:
        Tuple of (values_equal, error_message)
    """
    if tolerance is None:
        tolerance = config.numerics.precision

    equal = utils.eq(actual, expected)
    if not equal:
        diff = abs(actual - expected)
        msg = (
            f"Phi values differ: {actual} != {expected} "
            f"(diff: {diff:.2e}, tolerance: 10^-{tolerance})"
        )
        return False, msg
    return True, ""


def compare_partitions(
    actual: SystemPartition | NullCut | None,
    expected: SystemPartition | NullCut | None,
) -> tuple[bool, str]:
    """
    Compare two partition objects.

    Args:
        actual: Actual partition
        expected: Expected partition

    Returns:
        Tuple of (partitions_equal, error_message)
    """
    if actual is None and expected is None:
        return True, ""

    if type(actual) is not type(expected):
        msg = (
            f"Partition types differ: "
            f"{type(actual).__name__} != {type(expected).__name__}"
        )
        return False, msg

    if actual != expected:
        msg = f"Partitions differ:\n  Actual: {actual}\n  Expected: {expected}"
        return False, msg

    return True, ""


def compare_repertoires(actual: Any, expected: Any) -> tuple[bool, str]:
    """
    Compare two RepertoireIrreducibilityAnalysis objects.

    Args:
        actual: Actual RIA
        expected: Expected RIA

    Returns:
        Tuple of (repertoires_equal, error_message)
    """
    if actual is None and expected is None:
        return True, ""

    if (actual is None) != (expected is None):
        msg = (
            f"One repertoire is None: "
            f"actual={actual is None}, expected={expected is None}"
        )
        return False, msg

    # Compare phi values
    phi_equal, phi_msg = compare_phi_values(actual.phi, expected.phi)
    if not phi_equal:
        msg = f"Repertoire phi differs: {phi_msg}"
        return False, msg

    # Compare mechanisms
    if actual.mechanism != expected.mechanism:
        msg = f"Repertoire mechanisms differ: {actual.mechanism} != {expected.mechanism}"
        return False, msg

    # Compare purviews
    if actual.purview != expected.purview:
        msg = f"Repertoire purviews differ: {actual.purview} != {expected.purview}"
        return False, msg

    return True, ""


def extract_sia_components(sia: SystemIrreducibilityAnalysis) -> dict[str, Any]:
    """
    Extract key components from a SystemIrreducibilityAnalysis for independent testing.

    This function extracts values in a serialization-independent way, making
    tests more robust to data model changes.

    Args:
        sia: SystemIrreducibilityAnalysis object

    Returns:
        Dictionary with extracted components:
            - phi: float value
            - partition_type: type name of partition
            - partition_str: string representation of partition
            - has_cause: whether cause RIA exists
            - has_effect: whether effect RIA exists
            - cause_phi: cause RIA phi if present
            - effect_phi: effect RIA phi if present
            - has_system_state: whether system state exists
            - current_state: current state tuple
            - node_indices: node indices tuple
    """
    components = {
        "phi": float(sia.phi),
        "partition_type": type(sia.partition).__name__,
        "partition_str": str(sia.partition),
        "has_cause": sia.cause is not None,
        "has_effect": sia.effect is not None,
        "has_system_state": sia.system_state is not None,
        "current_state": sia.current_state,
        "node_indices": sia.node_indices,
    }

    # Extract cause/effect phi values if present
    if sia.cause is not None:
        components["cause_phi"] = float(sia.cause.phi)
        components["cause_mechanism"] = sia.cause.mechanism
        components["cause_purview"] = sia.cause.purview

    if sia.effect is not None:
        components["effect_phi"] = float(sia.effect.phi)
        components["effect_mechanism"] = sia.effect.mechanism
        components["effect_purview"] = sia.effect.purview

    return components


def extract_phi_structure_components(result: Any) -> dict[str, Any]:
    """
    Extract key components from a phi_structure result for independent testing.

    Args:
        result: CauseEffectStructure result object

    Returns:
        Dictionary with extracted components:
            - num_distinctions: count of distinctions
            - distinction_mechanisms: set of mechanism tuples
            - has_relations: whether relations exist
            - num_relations: count of relations (if applicable)
    """
    components = {}

    # Extract distinctions
    if hasattr(result, "distinctions"):
        distinctions = result.distinctions
        components["num_distinctions"] = len(distinctions)
        components["distinction_mechanisms"] = {tuple(d.mechanism) for d in distinctions}
    else:
        components["num_distinctions"] = 0
        components["distinction_mechanisms"] = set()

    # Extract relations
    if hasattr(result, "relations"):
        components["has_relations"] = result.relations is not None
        if result.relations is not None:
            # Handle different relation types
            if hasattr(result.relations, "__len__"):
                components["num_relations"] = len(result.relations)
            else:
                components["num_relations"] = None
    else:
        components["has_relations"] = False
        components["num_relations"] = 0

    return components


def diff_sia_results(
    actual: SystemIrreducibilityAnalysis, expected: SystemIrreducibilityAnalysis
) -> str:
    """
    Generate a human-readable diff of two SIA results.

    This is useful for pytest failure messages, showing exactly which
    attributes differ and by how much.

    Args:
        actual: Actual SIA result
        expected: Expected SIA result

    Returns:
        Formatted diff string
    """
    lines = []

    # Compare each attribute from _sia_attributes
    for attr in actual._sia_attributes:
        actual_val = getattr(actual, attr, None)
        expected_val = getattr(expected, attr, None)

        if actual_val != expected_val:
            lines.append(f"\n  {attr}:")
            lines.append(f"    Actual:   {actual_val}")
            lines.append(f"    Expected: {expected_val}")

            # Add more detail for specific types
            if attr == "phi":
                diff = abs(float(actual_val) - float(expected_val))
                lines.append(f"    Diff:     {diff:.2e}")

            elif attr == "partition":
                lines.append(f"    Actual type:   {type(actual_val).__name__}")
                lines.append(f"    Expected type: {type(expected_val).__name__}")

            elif (
                attr in ["cause", "effect"]
                and actual_val is not None
                and expected_val is not None
            ):
                # Compare RIA phi values
                if hasattr(actual_val, "phi") and hasattr(expected_val, "phi"):
                    phi_diff = abs(float(actual_val.phi) - float(expected_val.phi))
                    lines.append(f"    {attr}.phi diff: {phi_diff:.2e}")

    if not lines:
        return "No differences found"

    return "".join(lines)


def assert_sia_equal_detailed(
    actual: SystemIrreducibilityAnalysis,
    expected: SystemIrreducibilityAnalysis,
    *,
    check_phi_only: bool = False,
    check_partition: bool = True,
    check_cause_effect: bool = True,
    tolerance: int | None = None,
):
    """
    Compare two SIA objects with detailed diagnostics.

    This assertion function provides much more detailed failure messages
    than a simple equality check, showing exactly which components differ.

    Args:
        actual: Actual SIA result
        expected: Expected SIA result
        check_phi_only: If True, only check phi value (ignore structure)
        check_partition: Whether to check partition equality
        check_cause_effect: Whether to check cause/effect repertoires
        tolerance: Precision for phi comparison (uses config.numerics.precision if None)

    Raises:
        AssertionError: If SIA objects differ, with detailed diff message
    """
    errors = []

    # Always check phi
    phi_equal, phi_msg = compare_phi_values(actual.phi, expected.phi, tolerance)
    if not phi_equal:
        errors.append(f"Phi: {phi_msg}")

    if check_phi_only:
        # Short-circuit if only checking phi
        if errors:
            pytest.fail("\n".join(errors))
        return

    # Check partition
    if check_partition:
        partition_equal, partition_msg = compare_partitions(
            actual.partition, expected.partition
        )
        if not partition_equal:
            errors.append(f"Partition: {partition_msg}")

    # Check cause/effect repertoires
    if check_cause_effect:
        cause_equal, cause_msg = compare_repertoires(actual.cause, expected.cause)
        if not cause_equal:
            errors.append(f"Cause: {cause_msg}")

        effect_equal, effect_msg = compare_repertoires(actual.effect, expected.effect)
        if not effect_equal:
            errors.append(f"Effect: {effect_msg}")

    # If any errors, fail with detailed message
    if errors:
        full_diff = diff_sia_results(actual, expected)
        error_msg = "\n\nSIA comparison failed:\n" + "\n".join(errors)
        error_msg += "\n\nFull diff:" + full_diff
        pytest.fail(error_msg)


def assert_phi_structure_equal_detailed(
    actual: Any,
    expected: Any,
    *,
    check_distinctions: bool = True,
    check_relations: bool = True,
    check_system_state: bool = False,
):
    """
    Compare two phi_structure results with detailed diagnostics.

    Args:
        actual: Actual phi_structure result
        expected: Expected phi_structure result
        check_distinctions: Whether to check distinction counts/mechanisms
        check_relations: Whether to check relation counts
        check_system_state: Whether to check system state

    Raises:
        AssertionError: If results differ, with detailed diff message
    """
    errors = []

    actual_components = extract_phi_structure_components(actual)
    expected_components = extract_phi_structure_components(expected)

    # Check distinctions
    if check_distinctions:
        if (
            actual_components["num_distinctions"]
            != expected_components["num_distinctions"]
        ):
            errors.append(
                f"Number of distinctions differs: "
                f"{actual_components['num_distinctions']} != "
                f"{expected_components['num_distinctions']}"
            )

        # Check distinction mechanisms
        actual_mechanisms = actual_components["distinction_mechanisms"]
        expected_mechanisms = expected_components["distinction_mechanisms"]
        if actual_mechanisms != expected_mechanisms:
            missing = expected_mechanisms - actual_mechanisms
            extra = actual_mechanisms - expected_mechanisms
            if missing:
                errors.append(f"Missing distinction mechanisms: {missing}")
            if extra:
                errors.append(f"Extra distinction mechanisms: {extra}")

    # Check relations
    if check_relations:
        if actual_components["has_relations"] != expected_components["has_relations"]:
            errors.append(
                f"Relations presence differs: "
                f"{actual_components['has_relations']} != "
                f"{expected_components['has_relations']}"
            )

        if actual_components.get("num_relations") != expected_components.get(
            "num_relations"
        ):
            errors.append(
                f"Number of relations differs: "
                f"{actual_components.get('num_relations')} != "
                f"{expected_components.get('num_relations')}"
            )

    # If any errors, fail with detailed message
    if errors:
        error_msg = "\n\nCauseEffectStructure comparison failed:\n" + "\n".join(errors)
        pytest.fail(error_msg)
