import pytest

from pyphi import combinatorics
from pyphi import config
from pyphi import examples
from pyphi import jsonify
from pyphi import relations
from pyphi.formalism import iit3
from pyphi.formalism import iit4 as new_big_phi
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure


def test_only_nonsubsets():
    """Test only_nonsubsets (moved to pyphi.combinatorics)."""
    result = combinatorics.only_nonsubsets(
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


# NOTE: The following tests were removed because they tested IIT 3.0 functions
# that no longer exist:
# - test_overlap_states: relations.overlap_states was removed
# - test_congruent_overlap_empty: relations.congruent_overlap was removed
# - test_congruent_overlap: relations.congruent_overlap was removed
# - test_maximally_irreducible_relation: uses old relations API


def test_null_relations_is_empty():
    """NullRelations has zero phi, zero relations, empty iteration."""
    from pyphi.relations import NullRelations

    nr = NullRelations()
    assert nr.sum_phi() == 0
    assert nr.num_relations() == 0
    assert list(nr) == []


def test_null_relations_to_json_round_trips():
    from pyphi import jsonify
    from pyphi.relations import NullRelations

    nr = NullRelations()
    encoded = jsonify.loads(jsonify.dumps(nr))
    assert isinstance(encoded, NullRelations)
    assert encoded.sum_phi() == 0


NETWORKS = ["grid3", "basic", "xor", "rule110", "fig4"]


@pytest.mark.parametrize("case_name", NETWORKS)
@config.override(
    parallel=False,
)
def test_all_relations(case_name):
    with open(f"test/data/relations/ces_{case_name}.json") as f:
        answer_ces = jsonify.load(f)
    # Compute and check CES
    system = getattr(examples, f"{case_name}_system")()
    ces = iit3.ces(system)
    assert ces == answer_ces

    with open(f"test/data/relations/relations_{case_name}.json") as f:
        answers = jsonify.load(f)
    # Compute and check relations
    # TODO(4.0) config.override doesn't seem to work with joblib parallel?
    results = list(
        relations.relations(
            new_big_phi.ces(
                system,
                system_measure=resolve_system_measure(
                    config.formalism.iit.system_phi_measure
                ),
                specification_measure=resolve_mechanism_measure(
                    config.formalism.iit.specification_measure
                ),
            ).distinctions
        )
    )
    assert set(results) == set(answers)
