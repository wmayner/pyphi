import json

import numpy as np
import pytest

from pyphi import actual
from pyphi import examples
from pyphi import serialize
from pyphi.actual import Transition
from pyphi.conf.snapshot import ConfigSnapshot
from pyphi.direction import Direction
from pyphi.formalism.iit4 import NullSystemIrreducibilityAnalysis
from pyphi.models.complex import Complex
from pyphi.models.complex import ExcludedCandidate
from pyphi.models.explanation import NullResultReason
from pyphi.substrate import Substrate

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


def make_transition():
    sub = examples.actual_causation_substrate()
    return Transition(
        sub,
        before_state=(1, 1),
        after_state=(1, 1),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_substrate_round_trips(fmt):
    obj = examples.basic_substrate()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_system_round_trips(fmt):
    obj = examples.basic_system()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_round_trips(fmt):
    obj = make_transition()
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_system_round_trips(fmt):
    for obj in (make_transition().cause_system, make_transition().effect_system):
        restored = round_trip(obj, fmt)
        assert type(restored) is type(obj)
        assert restored == obj
        assert restored.direction == obj.direction


def test_transition_system_save_load_preserves_type(tmp_path):
    # Regression: ``save`` used to delegate to the underlying ``System``, so
    # ``load`` returned a System -- silent loss of the transition data.
    from pyphi.actual import TransitionSystem

    obj = make_transition().cause_system
    path = tmp_path / "ts.json"
    obj.save(path)
    restored = TransitionSystem.load(path)
    assert type(restored) is TransitionSystem
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_account_round_trips(fmt):
    obj = actual.account(make_transition())
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is type(obj)


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_sia_round_trips(fmt):
    obj = actual.sia(make_transition())
    restored = round_trip(obj, fmt)
    assert restored == obj


@pytest.mark.parametrize("fmt", FORMATS)
def test_complex_round_trips(fmt):
    obj = Complex(
        sia=NullSystemIrreducibilityAnalysis(node_indices=(0, 1)),
        substrate=examples.basic_substrate(),
        is_maximal=True,
        excluded=(ExcludedCandidate((0,), 0.1),),
    )
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert type(restored) is Complex


@pytest.mark.parametrize("fmt", FORMATS)
def test_excluded_candidate_round_trips_gated_and_certified(fmt):
    """Gated candidates (phi=None) from the certified prune must serialize,
    and the certification record (ii_ceiling, gated) must survive the
    round-trip -- for both gated and measured candidates.
    """
    gated = ExcludedCandidate((0, 1), None, ii_ceiling=0.25, gated=True)
    restored = round_trip(gated, fmt)
    assert restored == gated
    assert restored.phi is None
    assert restored.gated is True
    assert restored.ii_ceiling == 0.25

    measured = ExcludedCandidate((0, 1), 0.1, ii_ceiling=0.5)
    restored = round_trip(measured, fmt)
    assert restored == measured
    assert restored.ii_ceiling == 0.5


@pytest.mark.parametrize("fmt", FORMATS)
def test_kary_substrate_round_trips(fmt):
    obj = examples.gomez_p53_mdm2_substrate()  # ternary p53 + binary Mdm2 units
    restored = round_trip(obj, fmt)
    assert restored == obj
    assert list(restored.node_labels) == list(obj.node_labels)
    assert restored.factored_tpm.state_space == obj.factored_tpm.state_space


@pytest.mark.parametrize("fmt", FORMATS)
def test_transition_preserves_noise_background(fmt):
    # OR gate driven by a noised background unit: the EFFECT ratio is
    # nonzero only when noise_background survives the round-trip.
    substrate = Substrate(np.array([[0, 0], [1, 1], [1, 1], [1, 1]]))
    obj = Transition(substrate, (1, 1), (1, 1), (0,), (1,), noise_background=True)
    restored = round_trip(obj, fmt)
    assert restored.noise_background is True
    assert restored == obj
    original_ratio = obj._ratio(Direction.EFFECT, (0,), (1,))
    assert restored._ratio(Direction.EFFECT, (0,), (1,)) == original_ratio
    assert original_ratio != 0.0


@pytest.mark.parametrize("fmt", FORMATS)
def test_null_ac_sia_round_trips(fmt):
    sia = actual._null_ac_sia(make_transition(), Direction.CAUSE)
    restored = round_trip(sia, fmt)
    assert restored == sia
    assert len(restored.account) == 0
    assert len(restored.partitioned_account) == 0


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_ria_preserves_node_labels(fmt):
    s = actual.sia(make_transition())
    link = next(iter(s.account))
    assert link.ria.node_labels is not None
    restored = round_trip(s, fmt)
    rlink = next(iter(restored.account))
    assert rlink.ria.node_labels == link.ria.node_labels


@pytest.mark.parametrize("fmt", FORMATS)
def test_ac_sia_preserves_reasons_ties_config_provenance(fmt):
    t = make_transition()
    a = actual._null_ac_sia(t, Direction.CAUSE, reasons=[NullResultReason.NO_SYSTEM])
    b = actual._null_ac_sia(t, Direction.CAUSE, alpha=0.5)
    a.set_ties([a, b])
    restored = round_trip(a, fmt)
    assert restored.reasons == [NullResultReason.NO_SYSTEM]
    peers = [p for p in restored.ties if p is not restored]
    assert len(peers) == 1
    assert isinstance(restored.config, ConfigSnapshot)
    assert restored.provenance == a.provenance


def test_ac_sia_loads_without_new_fields():
    sia = actual._null_ac_sia(
        make_transition(), Direction.CAUSE, reasons=[NullResultReason.NO_SYSTEM]
    )
    data = json.loads(serialize.dumps(sia, format="json"))

    def strip(o):
        if isinstance(o, dict):
            for key in ("reasons", "config", "provenance", "tie_peers"):
                o.pop(key, None)
            for v in o.values():
                strip(v)
        elif isinstance(o, list):
            for item in o:
                strip(item)

    strip(data)
    restored = serialize.loads(json.dumps(data).encode(), format="json")
    assert restored.reasons == []
    assert restored.ties == (restored,)
    # Nothing stored: the constructor still snapshots load-time context.
    assert restored.config is not None
    assert restored.provenance is not None


@pytest.mark.parametrize("fmt", FORMATS)
def test_reducible_causal_link_round_trips(fmt):
    # Two independent copy units: the joint mechanism is reducible, so its
    # causal link carries a null RIA with purview/partition/probability None.
    sub = Substrate(np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float))
    t = Transition(
        sub,
        before_state=(1, 1),
        after_state=(1, 1),
        cause_indices=(0, 1),
        effect_indices=(0, 1),
    )
    link = t.find_causal_link(Direction.CAUSE, (0, 1))
    assert link.alpha == 0.0
    assert link.ria.probability is None
    restored = round_trip(link, fmt)
    assert restored == link
    assert restored.ria.purview == link.ria.purview
    assert restored.ria.partition == link.ria.partition
    assert restored.ria.probability is None
    assert restored.ria.partitioned_probability is None
    assert restored.ria.reasons == link.ria.reasons


@pytest.mark.parametrize("fmt", FORMATS)
def test_substrate_round_trip_preserves_factored_tpm_labels(fmt):
    obj = examples.basic_substrate()
    assert obj.factored_tpm.node_labels is not None
    restored = round_trip(obj, fmt)
    assert restored.factored_tpm.node_labels == obj.factored_tpm.node_labels
