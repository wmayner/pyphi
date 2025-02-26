from . import example_networks
import pytest
from pyphi import jsonify

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
    with open("./test/data/sia/s.json", mode="rt") as f:
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
    with open("./test/data/sia/s_noised.json", mode="rt") as f:
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
    with open("./test/data/sia/big_subsys_0_thru_3.json", mode="rt") as f:
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
    with open("./test/data/sia/rule152_s.json", mode="rt") as f:
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
    with open("./test/data/sia/big_subsys_all_complete.json", mode="rt") as f:
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
    with open("./test/data/sia/macro_s.json", mode="rt") as f:
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
    with open("./test/data/sia/micro_s.json", mode="rt") as f:
        expected = jsonify.load(f)
    return expected


@pytest.fixture()
def micro_s_all_off():
    return example_networks.micro_s_all_off()


@pytest.fixture()
def propagation_delay():
    return example_networks.propagation_delay()
