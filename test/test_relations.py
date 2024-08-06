import numpy as np
import pytest

from pyphi import compute, config, examples, jsonify, relations
from pyphi.models import FlatCauseEffectStructure


def all_array_equal(x, y):
    return all(np.array_equal(a, b) for a, b in zip(x, y))


def overlap(purviews):
    return set.intersection(*map(set, purviews))


cases = [
    # specified_states, purviews, overlap_states, congruent_overlap
    (
        [
            np.array(
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0],
                    [1, 1, 1, 0, 1],
                ]
            ),
            np.array(
                [
                    [0, 0, 0, 0, 1],
                    [0, 1, 1, 0, 0],
                    [0, 0, 1, 0, 1],
                ]
            ),
        ],
        [
            (0, 2, 4, 5),
            (1, 2, 3, 4, 5),
            (0, 1, 2, 4, 5),
        ],
        [
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        ],
        [
            (2, 4),
            (4, 5),
        ],
    ),
    (
        [np.array([[0], [1]]), np.array([[0]])],
        [(1,), (1,)],
        [np.array([[0], [1]]), np.array([[0]])],
        [(1,)],
    ),
]

specified_states, purviews, overlap_states, congruent_overlaps = zip(*cases)
overlaps = list(map(overlap, purviews))


def test_only_nonsubsets():
    result = relations.only_nonsubsets(
        [
            {0},
            {1},
            {0, 1, 2},
            {1, 2, 3},
            {0, 2, 3, 4},
            {1, 2, 3, 4},
        ]
    )
    answer = [
        {0, 1, 2},
        {1, 2, 3, 4},
        {0, 2, 3, 4},
    ]
    assert set(map(frozenset, result)) == set(map(frozenset, answer))


@pytest.mark.parametrize(
    "specified_states,purviews,answer", zip(specified_states, purviews, overlap_states)
)
def test_overlap_states(specified_states, purviews, answer):
    result = relations.overlap_states(specified_states, purviews, overlap(purviews))
    assert all_array_equal(result, answer)


def test_congruent_overlap_empty():
    assert relations.congruent_overlap((), ()) == []


@pytest.mark.parametrize(
    "overlap_states,overlap,answer", zip(overlap_states, overlaps, congruent_overlaps)
)
def test_congruent_overlap(overlap_states, overlap, answer):
    result = relations.congruent_overlap(
        overlap_states,
        overlap,
    )
    assert all_array_equal(result, answer)


NETWORKS = ["grid3", "basic", "pqr", "xor", "rule110", "fig4", "fig5a", "fig5b"]


@pytest.mark.parametrize("case_name", NETWORKS)
@config.override(
    REPERTOIRE_DISTANCE="ID",
    PARTITION_TYPE="TRI",
    PARALLEL_CONCEPT_EVALUATION=False,
    PARALLEL_CUT_EVALUATION=False,
    PARALLEL_COMPLEX_EVALUATION=False,
    RELATION_ALLOW_DUPLICATE_PURVIEWS=True,
    RELATION_COMPUTATION="EXACT",
    RELATION_POTENTIAL_PURVIEWS="ALL",
    RELATION_PHI_SCHEME="AGGREGATE_DISTINCTION_RELATIVE_DIFFERENCES",
)
def test_maximally_irreducible_relation(case_name):
    with open(f"test/data/relations/relations_{case_name}.json", mode="rt") as f:
        answers = jsonify.load(f)
    for r in answers:
        assert r == r.relata.maximally_irreducible_relation()


@pytest.mark.slow
@pytest.mark.parametrize("case_name", NETWORKS)
@config.override(
    REPERTOIRE_DISTANCE="ID",
    PARTITION_TYPE="TRI",
    PARALLEL_CONCEPT_EVALUATION=False,
    PARALLEL_CUT_EVALUATION=False,
    PARALLEL_COMPLEX_EVALUATION=False,
    RELATION_ALLOW_DUPLICATE_PURVIEWS=True,
    RELATION_COMPUTATION="EXACT",
    RELATION_POTENTIAL_PURVIEWS="ALL",
    RELATION_PHI_SCHEME="AGGREGATE_DISTINCTION_RELATIVE_DIFFERENCES",
)
def test_all_relations(case_name):
    with open(f"test/data/relations/ces_{case_name}.json", mode="rt") as f:
        answer_ces = jsonify.load(f)
    # Compute and check CES
    subsystem = getattr(examples, f"{case_name}_subsystem")()
    ces = FlatCauseEffectStructure(compute.ces(subsystem))
    assert ces == answer_ces

    with open(f"test/data/relations/relations_{case_name}.json", mode="rt") as f:
        answers = jsonify.load(f)
    # Compute and check relations
    # TODO(4.0) config.override doesn't seem to work with joblib parallel?
    results = list(relations.relations(subsystem, ces, parallel=False))
    assert set(results) == set(answers)
