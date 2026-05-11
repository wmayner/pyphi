import pytest

from pyphi import combinatorics
from pyphi import config
from pyphi import examples
from pyphi import jsonify
from pyphi import relations
from pyphi.formalism import iit3
from pyphi.formalism import iit4 as new_big_phi
from pyphi.metrics.distribution import resolve_mechanism_metric
from pyphi.metrics.distribution import resolve_system_metric


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
            new_big_phi.phi_structure(
                system,
                system_metric=resolve_system_metric(
                    config.formalism.iit.system_phi_measure
                ),
                specification_metric=resolve_mechanism_metric(
                    config.formalism.iit.specification_measure
                ),
            ).distinctions
        )
    )
    assert set(results) == set(answers)
