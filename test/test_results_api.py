"""The shape of result objects: one CES type, tuple-valued collections,
frozenset relation purviews, value equality, and the removed aliases."""

import importlib
from pathlib import Path

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.models import CauseEffectStructure
from pyphi.relations import NullRelations
from pyphi.substrate_generator import ising

RC3_IIT3_FIXTURE = (
    Path(__file__).parent / "data" / "serialize" / "rc3_xor_iit3_analysis.json"
)


def test_analysis_ces_has_one_type_under_every_version():
    s, st = examples.xor_substrate(), (0, 0, 0)
    for formalism in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"):
        ces = pyphi.analyze(s, st, formalism=formalism).ces
        assert isinstance(ces, CauseEffectStructure)
        assert len(ces.distinctions) == len(tuple(ces.distinctions))
        assert isinstance(ces.distinctions.phis, tuple)


def test_compute_ces_has_the_same_type_under_iit3():
    ces = pyphi.analyze(
        examples.xor_substrate(), (0, 0, 0), formalism="IIT_3_0", compute="ces"
    )
    assert isinstance(ces, CauseEffectStructure)
    assert isinstance(ces.relations, NullRelations)


def test_iit3_analysis_reads_like_every_other_version():
    analysis = pyphi.analyze(examples.xor_substrate(), (0, 0, 0), formalism="IIT_3_0")
    ces = analysis.ces
    assert isinstance(ces.relations, NullRelations)
    assert ces.sia is analysis.sia
    assert len(ces.distinctions) == 3
    assert [d.phi for d in ces.distinctions] == list(ces.distinctions.phis)
    assert ces.distinctions.phis == (0.5, 0.5, 0.5)
    assert analysis.phi == pytest.approx(1.875)
    card = str(analysis)
    assert "Distinctions" in card
    assert "Relations" not in card
    assert "Σφ_r" not in card
    assert analysis._repr_html_()
    frame = analysis.to_pandas()
    assert frame["n_distinctions"].item() == 3
    assert np.isnan(frame["sum_phi_r"].item())


def test_iit3_ces_round_trips(tmp_path):
    ces = pyphi.analyze(examples.xor_substrate(), (0, 0, 0), formalism="IIT_3_0").ces
    pyphi.save(ces, tmp_path / "c.json")
    assert pyphi.load(tmp_path / "c.json") == ces
    assert str(ces)


def test_iit3_analysis_round_trips(tmp_path):
    analysis = pyphi.analyze(examples.xor_substrate(), (0, 0, 0), formalism="IIT_3_0")
    pyphi.save(analysis, tmp_path / "a.json")
    loaded = pyphi.load(tmp_path / "a.json")
    assert loaded.ces == analysis.ces
    assert loaded.phi == analysis.phi
    assert loaded.formalism == "IIT_3_0"


def test_rc3_iit3_analysis_loads_as_a_cause_effect_structure():
    """An IIT 3.0 analysis saved by 2.0.0rc3 stored its CES as the bare
    distinctions; it loads as a CauseEffectStructure with empty relations."""
    loaded = pyphi.load(RC3_IIT3_FIXTURE)
    fresh = pyphi.analyze(examples.xor_substrate(), (0, 0, 0), formalism="IIT_3_0")
    assert isinstance(loaded.ces, CauseEffectStructure)
    assert isinstance(loaded.ces.relations, NullRelations)
    assert loaded.ces.sia is loaded.sia
    assert loaded.formalism == "IIT_3_0"
    assert loaded.phi == pytest.approx(fresh.phi)
    assert loaded.ces.distinctions.phis == pytest.approx(fresh.ces.distinctions.phis)
    assert loaded.ces.distinctions.mechanisms == fresh.ces.distinctions.mechanisms
    assert "Distinctions" in str(loaded)


def test_distinction_collections_are_tuples():
    distinctions = pyphi.analyze(examples.xor_substrate(), (0, 0, 0)).ces.distinctions
    assert isinstance(distinctions.phis, tuple)
    assert isinstance(distinctions.mechanisms, tuple)
    cause = distinctions.purviews(pyphi.Direction.CAUSE)
    assert isinstance(cause, tuple)
    both = distinctions.purviews(pyphi.Direction.both())
    assert both == cause + distinctions.purviews(pyphi.Direction.EFFECT)


def test_relation_purview_is_always_frozenset():
    a = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1))
    assert {type(r.purview) for r in a.ces.relations.materialize()} == {frozenset}


def _axis():
    weights = np.array([[-0.2, 0.7, 0.2], [0.7, -0.2, 0.0], [0.0, -0.8, 0.2]])
    return pyphi.weight_axis([ising.probability] * 3, weights, (0, 1), temperature=0.25)


def test_landscape_section_equality_and_hash():
    axis = _axis()
    one = pyphi.landscape_section(axis, (1, 0, 0), [0.5, 0.6], progress=False)
    two = pyphi.landscape_section(axis, (1, 0, 0), [0.5, 0.6], progress=False)
    assert one == two
    with pytest.raises(TypeError):
        hash(one)


def test_perturbation_is_unhashable():
    perturbation = pyphi.perturb(_axis(), (1, 0, 0), 0.7)
    with pytest.raises(TypeError):
        hash(perturbation)


@pytest.mark.parametrize(
    "module,name",
    [
        ("pyphi.models", "Concept"),
        ("pyphi.convert", "b2l"),
        ("pyphi.convert", "sbs2sbn"),
    ],
)
def test_removed_aliases_name_the_replacement(module, name):
    with pytest.raises(AttributeError, match=r"replaced|renamed"):
        getattr(importlib.import_module(module), name)


@pytest.mark.parametrize(
    "old,new", [("null_concept", "null_distinction"), ("apply_cut", "apply_partition")]
)
def test_removed_system_names_name_the_replacement(old, new):
    system = examples.xor_system()
    with pytest.raises(AttributeError, match=new):
        getattr(system, old)


def test_system_card_says_partition():
    system = examples.xor_system()
    card = str(system.apply_partition(system.sia().partition))
    assert "Partition" in card
    assert "Cut" not in card


def test_analytical_relations_to_pandas_summarizes():
    a = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1))
    relations = a.ces.relations
    df = relations.to_pandas()
    assert len(df) > 0
    assert list(df.columns) == ["degree", "num_relations", "sum_phi"]
    assert df["num_relations"].sum() == relations.num_relations()
    assert df["sum_phi"].sum() == pytest.approx(relations.sum_phi())
