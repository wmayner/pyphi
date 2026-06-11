import numpy as np
import pytest

import pyphi
from pyphi import Substrate
from pyphi import actual
from pyphi import config
from pyphi import jsonify
from pyphi.conf import presets

from . import example_substrates

# Disable JSON version validation for tests
# Test data may be generated with different pyphi versions than what CI derives
pyphi.config.validate_json_version = False

# Suppress tqdm progress bars in test output. The global gate cascades
# to every ``parallel_*_evaluation`` per-site ``progress`` flag (see
# ``pyphi.conf._helpers.parallel_kwargs``), so this single switch covers
# all parallel and sequential progress chrome at once.
pyphi.config.progress_bars = False


# IIT Version Configuration Overrides
# =============================================================================
#
# These config overrides allow tests to run under specific IIT version settings.
# Use them in test classes with an autouse fixture:
#
#     class TestMyFeatureIIT30:
#         @pytest.fixture(autouse=True)
#         def _apply_config(self):
#             with IIT_3_CONFIG:
#                 yield
#
#         def test_something(self):
#             ...

# IIT 3.0 configuration for regression tests.
# Sourced from the canonical ``presets.iit3`` so the test config and the
# library's documented preset cannot drift apart.
IIT_3_CONFIG = config.override(**presets.iit3)

# IIT 4.0 configuration (current defaults, made explicit for clarity)
# Use this when you want to explicitly test IIT 4.0 behavior
IIT_4_CONFIG = config.override(
    {"iit.version": "IIT_4_0_2023"},
    mechanism_phi_measure="GENERALIZED_INTRINSIC_DIFFERENCE",
    system_phi_measure="GENERALIZED_INTRINSIC_DIFFERENCE",
    system_partition_scheme="DIRECTED_SET_PARTITION",
)

# Pytest configuration
# =============================================================================


def pytest_addoption(parser):
    """Custom CLI options."""
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help=(
            "Regenerate golden fixture data (test/data/golden/v1/) from current "
            "code. Use after intentional formula changes; verify against "
            "published IIT results before committing the regenerated fixtures."
        ),
    )


def pytest_configure(config):
    """Register custom markers for test categorization.

    Markers:
    - golden: Golden reference test comparing full structure against JSON fixture
    - robust: Robust component-level test with intermediate checks
    - slow: Tier-2 test (skipped by default; opt in with the root conftest's
      ``--slow`` flag). Used by the golden suite to gate IIT 4.0 (2026) and
      large-substrate fixtures.
    - perf: Wall-time floor assertion on a hot-path fixture.
    """
    config.addinivalue_line(
        "markers",
        "golden: Golden reference test comparing full structure against JSON fixture",
    )
    config.addinivalue_line(
        "markers",
        "robust: Robust component-level test with intermediate checks",
    )
    config.addinivalue_line(
        "markers",
        "perf: Wall-time floor assertion on a hot-path fixture",
    )


def pytest_assertrepr_compare(op, left, right):
    """Custom assertion messages for SIA comparisons.

    Provides detailed diff output when SystemIrreducibilityAnalysis
    objects are compared, showing exactly which attributes differ.
    """
    # Import here to avoid circular imports
    try:
        from pyphi.formalism.iit4 import SystemIrreducibilityAnalysis
    except ImportError:
        return None

    # Check if we're comparing two SIA objects
    if (
        isinstance(left, SystemIrreducibilityAnalysis)
        and isinstance(right, SystemIrreducibilityAnalysis)
        and op == "=="
    ):
        try:
            from .test_helpers import diff_sia_results

            diff_output = diff_sia_results(left, right)
            return [
                "Comparing SystemIrreducibilityAnalysis objects:",
                *diff_output.split("\n"),
            ]
        except (ImportError, AttributeError):
            # If test_helpers not available, fall back to default
            return None

    return None


# Check if pyemd is available
# =============================================================================

try:
    import pyemd  # noqa: F401

    PYEMD_AVAILABLE = True
except ImportError:
    PYEMD_AVAILABLE = False

# Skip decorator for EMD tests
skip_if_no_pyemd = pytest.mark.skipif(
    not PYEMD_AVAILABLE,
    reason="pyemd not installed (install with: pip install pyphi[emd])",
)

# Test fixtures from example substrates
# =============================================================================


# Matlab standard substrate and systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def standard():
    return example_substrates.standard()


@pytest.fixture()
def s():
    return example_substrates.s()


@pytest.fixture()
def s_empty():
    return example_substrates.s_empty()


@pytest.fixture()
def s_single():
    return example_substrates.s_single()


@pytest.fixture()
def subsys_n0n2():
    return example_substrates.subsys_n0n2()


@pytest.fixture()
def subsys_n1n2():
    return example_substrates.subsys_n1n2()


@pytest.fixture
def s_expected_sia():
    with open("./test/data/sia/s.json") as f:
        expected = jsonify.load(f)
    return expected


# Noised standard example and systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def noised():
    return example_substrates.noised()


@pytest.fixture()
def s_noised():
    return example_substrates.s_noised()


@pytest.fixture()
def noisy_selfloop_single():
    return example_substrates.noisy_selfloop_single()


@pytest.fixture
def s_noised_expected_sia():
    with open("./test/data/sia/s_noised.json") as f:
        expected = jsonify.load(f)
    return expected


# Simple substrate and systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def simple():
    return example_substrates.simple()


@pytest.fixture()
def simple_subsys_all_off():
    return example_substrates.simple_subsys_all_off()


@pytest.fixture()
def simple_subsys_all_a_just_on():
    return example_substrates.simple_subsys_all_a_just_on()


# Big substrate and systems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def big():
    return example_substrates.big()


@pytest.fixture()
def big_subsys_all():
    return example_substrates.big_subsys_all()


@pytest.fixture()
def big_subsys_0_thru_3():
    return example_substrates.big_subsys_0_thru_3()


@pytest.fixture
def big_subsys_0_thru_3_expected_sia():
    with open("./test/data/sia/big_subsys_0_thru_3.json") as f:
        expected = jsonify.load(f)
    return expected


# Trivially reducible substrate
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def reducible():
    return example_substrates.reducible()


@pytest.fixture()
def rule152():
    return example_substrates.rule152()


@pytest.fixture()
def rule152_s():
    return example_substrates.rule152_s()


@pytest.fixture
def rule152_s_expected_sia():
    with open("./test/data/sia/rule152_s.json") as f:
        expected = jsonify.load(f)
    return expected


# Systems with complete graphs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def s_complete():
    return example_substrates.s_complete()


@pytest.fixture()
def big_subsys_all_complete():
    return example_substrates.big_subsys_all_complete()


@pytest.fixture
def big_subsys_all_complete_expected_sia():
    with open("./test/data/sia/big_subsys_all_complete.json") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def rule152_s_complete():
    return example_substrates.rule152_s_complete()


@pytest.fixture()
def eights_complete():
    return example_substrates.eights_complete()


# Macro/Micro substrates
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def macro():
    return example_substrates.macro()


@pytest.fixture()
def macro_s():
    return example_substrates.macro_s()


@pytest.fixture
def macro_s_expected_sia():
    with open("./test/data/sia/macro_s.json") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def micro():
    return example_substrates.micro()


@pytest.fixture()
def micro_s():
    return example_substrates.micro_s()


@pytest.fixture
def micro_s_expected_sia():
    with open("./test/data/sia/micro_s.json") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def micro_s_all_off():
    return example_substrates.micro_s_all_off()


# Actual causation fixtures
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture
def transition():
    """An OR gate with two inputs. The OR gate is ON, others are OFF."""
    # fmt: off
    tpm = np.array([
        [0, 0.5, 0.5],
        [0, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
        [1, 0.5, 0.5],
    ])
    cm = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ])
    # fmt: on
    substrate = Substrate(tpm, cm)
    before_state = (0, 1, 1)
    after_state = (1, 0, 0)
    return actual.Transition(substrate, before_state, after_state, (1, 2), (0,))
