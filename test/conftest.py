import numpy as np
import pytest

import pyphi
from pyphi import Network
from pyphi import actual
from pyphi import config
from pyphi import jsonify

from . import example_networks

# Disable JSON version validation for tests
# Test data may be generated with different pyphi versions than what CI derives
pyphi.config.VALIDATE_JSON_VERSION = False


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

# IIT 3.0 configuration for regression tests
# These settings replicate the IIT 3.0 computational approach.
IIT_3_CONFIG = config.override(
    FORMALISM="IIT_3_0",
    REPERTOIRE_DISTANCE="EMD",
    PARTITION_TYPE="BI",
    SYSTEM_PARTITION_TYPE="DIRECTED_BI",
    ACTUAL_CAUSATION_MEASURE="PMI",
    PURVIEW_TIE_RESOLUTION=["PHI", "PURVIEW_SIZE"],
)

# IIT 4.0 configuration (current defaults, made explicit for clarity)
# Use this when you want to explicitly test IIT 4.0 behavior
IIT_4_CONFIG = config.override(
    FORMALISM="IIT_4_0_2023",
    REPERTOIRE_DISTANCE="GENERALIZED_INTRINSIC_DIFFERENCE",
    SYSTEM_PARTITION_TYPE="SET_UNI/BI",
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
    """
    config.addinivalue_line(
        "markers",
        "golden: Golden reference test comparing full structure against JSON fixture",
    )
    config.addinivalue_line(
        "markers",
        "robust: Robust component-level test with intermediate checks",
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

# Test fixtures from example networks
# =============================================================================


# Matlab standard network and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def standard():
    return example_networks.standard()


@pytest.fixture()
def s():
    return example_networks.s()


@pytest.fixture()
def s_empty():
    return example_networks.s_empty()


@pytest.fixture()
def s_single():
    return example_networks.s_single()


@pytest.fixture()
def subsys_n0n2():
    return example_networks.subsys_n0n2()


@pytest.fixture()
def subsys_n1n2():
    return example_networks.subsys_n1n2()


@pytest.fixture
def s_expected_sia():
    with open("./test/data/sia/s.json") as f:
        expected = jsonify.load(f)
    return expected


# Noised standard example and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def noised():
    return example_networks.noised()


@pytest.fixture()
def s_noised():
    return example_networks.s_noised()


@pytest.fixture()
def noisy_selfloop_single():
    return example_networks.noisy_selfloop_single()


@pytest.fixture
def s_noised_expected_sia():
    with open("./test/data/sia/s_noised.json") as f:
        expected = jsonify.load(f)
    return expected


# Simple network and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def simple():
    return example_networks.simple()


@pytest.fixture()
def simple_subsys_all_off():
    return example_networks.simple_subsys_all_off()


@pytest.fixture()
def simple_subsys_all_a_just_on():
    return example_networks.simple_subsys_all_a_just_on()


# Big network and subsystems
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def big():
    return example_networks.big()


@pytest.fixture()
def big_subsys_all():
    return example_networks.big_subsys_all()


@pytest.fixture()
def big_subsys_0_thru_3():
    return example_networks.big_subsys_0_thru_3()


@pytest.fixture
def big_subsys_0_thru_3_expected_sia():
    with open("./test/data/sia/big_subsys_0_thru_3.json") as f:
        expected = jsonify.load(f)
    return expected


# Trivially reducible network
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def reducible():
    return example_networks.reducible()


@pytest.fixture()
def rule152():
    return example_networks.rule152()


@pytest.fixture()
def rule152_s():
    return example_networks.rule152_s()


@pytest.fixture
def rule152_s_expected_sia():
    with open("./test/data/sia/rule152_s.json") as f:
        expected = jsonify.load(f)
    return expected


# Subsystems with complete graphs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def s_complete():
    return example_networks.s_complete()


@pytest.fixture()
def s_noised_complete():
    return example_networks.s_noised_complete()


@pytest.fixture()
def big_subsys_all_complete():
    return example_networks.big_subsys_all_complete()


@pytest.fixture
def big_subsys_all_complete_expected_sia():
    with open("./test/data/sia/big_subsys_all_complete.json") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def rule152_s_complete():
    return example_networks.rule152_s_complete()


@pytest.fixture()
def eights_complete():
    return example_networks.eights_complete()


# Macro/Micro networks
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


@pytest.fixture()
def macro():
    return example_networks.macro()


@pytest.fixture()
def macro_s():
    return example_networks.macro_s()


@pytest.fixture
def macro_s_expected_sia():
    with open("./test/data/sia/macro_s.json") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def micro():
    return example_networks.micro()


@pytest.fixture()
def micro_s():
    return example_networks.micro_s()


@pytest.fixture
def micro_s_expected_sia():
    with open("./test/data/sia/micro_s.json") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def micro_s_all_off():
    return example_networks.micro_s_all_off()


@pytest.fixture()
def propagation_delay():
    return example_networks.propagation_delay()


@pytest.fixture()
def differentiation_example_micro_1():
    return example_networks.differentiation_example_micro_1()


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
    network = Network(tpm, cm)
    before_state = (0, 1, 1)
    after_state = (1, 0, 0)
    return actual.Transition(network, before_state, after_state, (1, 2), (0,))
