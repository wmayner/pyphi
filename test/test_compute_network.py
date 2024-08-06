import pickle

import pytest

from pyphi import Network, Subsystem, compute, config, constants, utils

from .test_big_phi import check_sia, standard_answer


def test_possible_complexes(s):
    assert list(compute.network.possible_complexes(s.network, s.state)) == [
        Subsystem(s.network, s.state, (0, 1, 2)),
        Subsystem(s.network, s.state, (1, 2)),
        Subsystem(s.network, s.state, (0, 2)),
        Subsystem(s.network, s.state, (0, 1)),
        Subsystem(s.network, s.state, (1,)),
    ]


def test_complexes_standard(s):
    complexes = list(compute.network.complexes(s.network, s.state), parallel=False)
    check_sia(complexes[0], standard_answer)


# TODO!! add more assertions for the smaller subsystems
@pytest.mark.slow
def test_all_complexes_standard(s):
    complexes = list(compute.network.all_complexes(s.network, s.state))
    check_sia(complexes[0], standard_answer)


@config.override(PARALLEL_CUT_EVALUATION=False)
def test_all_complexes_parallelization(use_iit_3_config, s):
    with config.override(PARALLEL_COMPLEX_EVALUATION=False):
        serial = compute.network.all_complexes(s.network, s.state)

    with config.override(PARALLEL_COMPLEX_EVALUATION=True):
        parallel = compute.network.all_complexes(s.network, s.state)

    assert sorted(serial) == sorted(parallel)


# TODO fix this horribly outdated mess that never worked in the first place :P
@pytest.mark.veryslow
def test_rule152_complexes_no_caching(rule152):
    net = rule152
    # Mapping from index of a PyPhi subsystem in network.subsystems to the
    # index of the corresponding subsystem in the Matlab list of subsets
    perm = {
        0: 0,
        1: 1,
        2: 3,
        3: 7,
        4: 15,
        5: 2,
        6: 4,
        7: 8,
        8: 16,
        9: 5,
        10: 9,
        11: 17,
        12: 11,
        13: 19,
        14: 23,
        15: 6,
        16: 10,
        17: 18,
        18: 12,
        19: 20,
        20: 24,
        21: 13,
        22: 21,
        23: 25,
        24: 27,
        25: 14,
        26: 22,
        27: 26,
        28: 28,
        29: 29,
        30: 30,
    }
    with open("test/data/rule152_results.pkl", "rb") as f:
        results = pickle.load(f)

    # Don't use concept caching for this test.
    constants.CACHE_CONCEPTS = False

    for state, result in results.items():
        # Unpack the state from the results key.
        # Generate the network with the state we're testing.
        net = Network(rule152.tpm, state, cm=rule152.cm)
        # Comptue all the complexes, leaving out the first (empty) subsystem
        # since Matlab doesn't include it in results.
        complexes = list(compute.network.complexes(net))[1:]
        # Check the phi values of all complexes.
        zz = [
            (sia.phi, result["subsystem_phis"][perm[i]])
            for i, sia in list(enumerate(complexes))
        ]
        diff = [
            utils.eq(sia.phi, result["subsystem_phis"][perm[i]])
            for i, sia in list(enumerate(complexes))
        ]
        assert all(
            utils.eq(sia.phi, result["subsystem_phis"][perm[i]])
            for i, sia in list(enumerate(complexes))[:]
        )
        # Check the major complex in particular.
        major = compute.subsystem.major_complex(net)
        # Check the phi value of the major complex.
        assert utils.eq(major.phi, result["phi"])
        # Check that the nodes are the same.
        assert (
            major.subsystem.node_indices
            == complexes[result["major_complex"] - 1].subsystem.node_indices
        )
        # Check that the concept's phi values are the same.
        result_concepts = [c for c in result["concepts"] if c["is_irreducible"]]
        z = list(zip([c.phi for c in major.ces], [c["phi"] for c in result_concepts]))
        diff = [i for i in range(len(z)) if not utils.eq(z[i][0], z[i][1])]
        assert all(
            list(
                utils.eq(c.phi, result_concepts[i]["phi"])
                for i, c in enumerate(major.ces)
            )
        )
        # Check that the minimal cut is the same.
        assert major.cut == result["cut"]
