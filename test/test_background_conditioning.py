"""Tests for the cause-side background-conditioning conventions."""

import numpy as np
import pytest

from pyphi import config
from pyphi.system import System

from . import example_substrates

STATE = (1, 0, 0)
SYSTEM_NODES = (0, 1)

# Manually derived on the discriminating substrate; also pinned by the
# genuine PyPhi 1.2.0 oracle (test/data/iit3-canonical/).
CAUSE_REP_MARGINALIZED = [0.40566037735849053, 0.5943396226415094]
CAUSE_REP_CONDITIONED = [0.1, 0.9]


@pytest.fixture()
def substrate():
    return example_substrates.noisy_or_background_substrate()


def _system(substrate, **kwargs):
    return System(substrate, STATE, node_indices=SYSTEM_NODES, **kwargs)


class TestCauseRepertoireConventions:
    def test_default_is_marginalized(self, substrate):
        rep = _system(substrate).cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_MARGINALIZED)

    def test_conditioned_convention_via_config(self, substrate):
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            rep = _system(substrate).cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_CONDITIONED)

    def test_system_field_pins_convention_over_config(self, substrate):
        pinned = _system(substrate, background_conditioning="CAUSAL_MARGINALIZATION")
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            rep = pinned.cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_MARGINALIZED)

    def test_invalid_field_value_rejected(self, substrate):
        with pytest.raises(ValueError, match="background_conditioning"):
            _system(substrate, background_conditioning="PAST_STATE")

    def test_effect_side_is_convention_invariant(self, substrate):
        default = _system(substrate).effect_repertoire((0,), (1,)).squeeze()
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            conditioned = _system(substrate).effect_repertoire((0,), (1,)).squeeze()
        assert np.array_equal(default, conditioned)
        assert default == pytest.approx([0.0, 1.0])

    def test_full_substrate_system_is_convention_invariant(self, substrate):
        full = System(substrate, STATE)
        baseline = full.cause_repertoire((0,), (1,))
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            conditioned = System(substrate, STATE).cause_repertoire((0,), (1,))
        assert np.array_equal(baseline, conditioned)


class TestCacheFreshness:
    def test_same_system_object_respects_config_flip(self, substrate):
        # The option is read at compute time: a System built (and computed
        # on) under one convention must produce the other convention's
        # values inside an override — through both the System-level caches
        # and the kernel memo cache.
        s = _system(substrate)
        before = s.cause_repertoire((0,), (1,)).squeeze()
        assert before == pytest.approx(CAUSE_REP_MARGINALIZED)
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            inside = s.cause_repertoire((0,), (1,)).squeeze()
        after = s.cause_repertoire((0,), (1,)).squeeze()
        assert inside == pytest.approx(CAUSE_REP_CONDITIONED)
        assert after == pytest.approx(CAUSE_REP_MARGINALIZED)

    def test_apply_cut_shares_marginals_across_conventions(self, substrate):
        from pyphi.direction import Direction
        from pyphi.models.partitions import DirectedBipartition

        s = _system(substrate)
        _ = s.cause_repertoire((0,), (1,))
        cut = s.apply_cut(
            DirectedBipartition(Direction.CAUSE, (0,), (1,), node_labels=s.node_labels)
        )
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            rep = cut.cause_repertoire((0,), (1,)).squeeze()
        assert rep == pytest.approx(CAUSE_REP_CONDITIONED)


class TestKernelOperation:
    def test_conditioned_factors_match_eq4_on_conditioned_tpm(self, substrate):
        # The direct construction equals the Eq. 4 machinery run on the
        # background-conditioned TPM (the weight degenerates to exactly 1).
        from pyphi.core.tpm.marginalization import _cause_marginal_factored
        from pyphi.core.tpm.marginalization import cause_conditioned

        tpm = substrate.factored_tpm
        background = {2: STATE[2]}
        direct = cause_conditioned(tpm, SYSTEM_NODES, background)
        via_eq4 = _cause_marginal_factored(
            tpm.condition(background), STATE, SYSTEM_NODES
        )
        for i in SYSTEM_NODES:
            assert np.array_equal(direct.factor(i), via_eq4.factor(i))


class TestValueSemantics:
    def test_eq_hash_fingerprint_distinguish_pinned_systems(self, substrate):
        plain = _system(substrate)
        pinned = _system(substrate, background_conditioning="CONDITION_CURRENT_STATE")
        assert plain != pinned
        assert plain._fingerprint != pinned._fingerprint
        same = _system(substrate)
        assert plain == same
        assert hash(plain) == hash(same)

    def test_serialization_round_trip_preserves_pin(self, substrate):
        from pyphi import serialize

        pinned = _system(substrate, background_conditioning="CONDITION_CURRENT_STATE")
        restored = serialize.loads(serialize.dumps(pinned))
        assert restored == pinned
        assert restored.background_conditioning == "CONDITION_CURRENT_STATE"
        plain = _system(substrate)
        assert serialize.loads(serialize.dumps(plain)) == plain


class TestActualCausationInsulation:
    def test_ac_account_invariant_under_the_knob(self, substrate):
        # A transition over a proper subset of the substrate: background
        # unit C is outside the transition, the exact situation where the
        # knob would otherwise leak into AC cause repertoires.
        from pyphi import actual

        def account_alphas():
            transition = actual.Transition(
                substrate,
                before_state=(1, 0, 0),
                after_state=(0, 1, 0),
                cause_indices=(0, 1),
                effect_indices=(0, 1),
            )
            account = actual.account(transition)
            return sorted(
                (link.direction, tuple(link.mechanism), float(link.alpha))
                for link in account
            )

        baseline = account_alphas()
        with config.override(background_conditioning="CONDITION_CURRENT_STATE"):
            flipped = account_alphas()
        assert flipped == baseline
        assert len(baseline) > 0


class TestFromSubstrateKwargs:
    def test_from_substrate_forwards_background_conditioning(self, substrate):
        factory = System.from_substrate(
            substrate,
            STATE,
            SYSTEM_NODES,
            background_conditioning="CONDITION_CURRENT_STATE",
        )
        assert factory.background_conditioning == "CONDITION_CURRENT_STATE"
        direct = _system(substrate, background_conditioning="CONDITION_CURRENT_STATE")
        rep_factory = factory.cause_repertoire((0,), (1,)).squeeze()
        rep_direct = direct.cause_repertoire((0,), (1,)).squeeze()
        assert rep_factory == pytest.approx(rep_direct)

    def test_from_substrate_rejects_unknown_kwargs(self, substrate):
        with pytest.raises(TypeError):
            System.from_substrate(substrate, STATE, SYSTEM_NODES, bogus_kwarg=1)
