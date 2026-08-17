import pytest

from pyphi import combinatorics
from pyphi import config
from pyphi import examples
from pyphi import relations
from pyphi import serialize
from pyphi.formalism import iit3
from pyphi.formalism import iit4 as new_big_phi
from pyphi.measures.distribution import resolve_mechanism_measure
from pyphi.measures.distribution import resolve_system_measure
from test.conftest import IIT_4_CONFIG


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


def test_null_relations_serialize_round_trips():
    from pyphi.relations import NullRelations

    nr = NullRelations()
    encoded = serialize.loads(serialize.dumps(nr))
    assert isinstance(encoded, NullRelations)
    assert encoded.sum_phi() == 0


def test_null_relations_len_is_zero():
    """len(NullRelations()) returns 0, matching the sister classes' contract."""
    from pyphi.relations import NullRelations

    assert len(NullRelations()) == 0


@pytest.fixture(autouse=True)
def _pin_formalism():
    """Pin IIT 4.0 (2023) with concrete relations: the golden CES/relation
    files in this module are 2023-sourced and compared relation-by-relation,
    so the comparisons must not depend on the ambient default. (Under the
    2026 default, deterministic fixtures cap to φ_s = 0 and their
    congruence-resolved structures are empty.)"""
    with IIT_4_CONFIG, config.override(relation_computation="CONCRETE"):
        yield


NETWORKS = ["grid3", "basic", "xor", "rule110", "fig4"]


@pytest.mark.parametrize("case_name", NETWORKS)
@config.override(
    parallel=False,
)
def test_all_relations(case_name):
    with open(f"test/data/relations/ces_{case_name}.json") as f:
        answer_ces = serialize.load(f)
    # Compute and check CES
    system = getattr(examples, f"{case_name}_system")()
    ces = iit3._compute_distinctions(system)
    assert ces == answer_ces

    with open(f"test/data/relations/relations_{case_name}.json") as f:
        answers = serialize.load(f)
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


@pytest.mark.parametrize("case_name", ["basic", "xor"])
@config.override(parallel=False)
def test_analytical_relations_sum_matches_concrete(case_name):
    """``AnalyticalRelations.sum_phi()`` equals the concrete relation-phi sum.

    The analytical sum (Albantakis et al. 2023, S3) yields the total relation
    small-phi without enumerating concrete relations, which the
    paper-reproduction suite relies on to obtain Phi for larger systems (e.g.
    IIT 4.0 Fig 6C, and feasibly 6D's ~1.5M relations). This guards that the two
    routes agree.
    """
    system = getattr(examples, f"{case_name}_system")()
    distinctions = new_big_phi.ces(
        system,
        system_measure=resolve_system_measure(config.formalism.iit.system_phi_measure),
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    ).distinctions
    concrete_sum = sum(float(r.phi) for r in relations.relations(distinctions))
    analytical_sum = float(relations.AnalyticalRelations(distinctions).sum_phi())
    assert analytical_sum == pytest.approx(concrete_sum)


@config.override(parallel=False)
def test_analytical_relations_iteration_raises_guided_error():
    """Iterating or indexing ``AnalyticalRelations`` raises a ``TypeError`` that
    points to the enumerable alternatives, instead of a bare "not iterable"."""
    distinctions = new_big_phi.ces(
        examples.basic_system(),
        system_measure=resolve_system_measure(config.formalism.iit.system_phi_measure),
        specification_measure=resolve_mechanism_measure(
            config.formalism.iit.specification_measure
        ),
    ).distinctions
    rels = relations.AnalyticalRelations(distinctions)

    for op in (lambda: list(rels), lambda: iter(rels), lambda: rels[0]):
        with pytest.raises(TypeError) as excinfo:
            op()
        message = str(excinfo.value)
        assert "strongest" in message
        assert "materialize" in message
        assert "CONCRETE" in message

    # The closed-form count still works — only enumeration is unavailable.
    assert isinstance(rels.num_relations(), int)


def test_analytical_relations_rejects_unsupported_kwargs():
    import pytest

    from pyphi.relations import analytical_relations

    with pytest.raises(TypeError, match="analytical relation computation"):
        analytical_relations((), max_degree=2)


def test_analytical_relations_have_no_len():
    """Analytical relations hold no relation objects (iteration raises), so
    they define no length at any scale; the count is ``num_relations()``.
    ``len()`` on a closed form would otherwise work at small sizes and raise
    at production sizes — a size-dependent API."""
    import pyphi
    from pyphi.conf import presets
    from pyphi.relations import AnalyticalRelations
    from pyphi.system import System

    with pyphi.config.override(
        **presets.by_name["IIT_4_0_2026"], progress_bars=False, parallel=False
    ):
        s = System.from_substrate(examples.basic_substrate(), (1, 0, 0), None)
        ar = AnalyticalRelations(s.distinctions())
    assert isinstance(ar.num_relations(), int)
    with pytest.raises(TypeError, match="len"):
        len(ar)
    # Truthiness must not fall back to the missing __len__.
    assert bool(ar)


def test_relation_face_orders_by_phi():
    """max()/sorted() on faces follow φ, not frozenset subset comparison."""
    lo = relations.RelationFace(["a"], phi=0.1)
    hi = relations.RelationFace(["b"], phi=0.9)
    assert lo < hi
    assert hi > lo
    assert (hi < lo) is False
    assert max([lo, hi]) is hi
    assert max([hi, lo]) is hi
    # Equality and hashing remain set semantics.
    assert lo == relations.RelationFace(["a"], phi=0.1)
    assert hash(lo) == hash(relations.RelationFace(["a"], phi=0.1))
    assert lo != hi


def test_relation_orders_by_phi():
    """max()/sorted() on relations follow φ, not frozenset subset comparison."""
    lo = relations.Relation(["a"])
    hi = relations.Relation(["b"])
    # Seed the cached phi property; ordering must read it.
    lo.__dict__["phi"] = 0.1
    hi.__dict__["phi"] = 0.9
    assert lo < hi
    assert hi > lo
    assert max([lo, hi]) is hi
    assert max([hi, lo]) is hi
    # Equality and hashing remain set semantics.
    assert lo == relations.Relation(["a"])
    assert lo != hi
