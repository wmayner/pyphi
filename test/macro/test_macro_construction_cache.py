"""Tests for the macro-construction intermediate cache."""

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import pyphi
from pyphi import config
from pyphi.conf import presets
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.macro import tpm as macro_tpm_module
from pyphi.macro.tpm import macro_tpms
from pyphi.macro.units import MacroUnit
from pyphi.macro.units import coarse_grain
from pyphi.substrate import Substrate
from test.macro.test_macro_goldens import BBX_UNITS
from test.macro.test_macro_goldens import CG_UNITS
from test.macro.test_macro_tpm import CG_TPM
from test.macro.test_macro_tpm import _bbx_micro_tpm

CG_STATE = (0, 0, 0, 0)
BBX_ONES = (1,) * 8

# Same footprints as CG_UNITS, different mapping.
CG_VARIANT_UNITS = tuple(
    MacroUnit(unit.constituents, 1, coarse_grain(2, on_counts={1, 2}))
    for unit in CG_UNITS
)


@pytest.fixture(autouse=True)
def _fresh_cache():
    macro_tpm_module._CONSTRUCTION_CACHE.clear()
    yield
    macro_tpm_module._CONSTRUCTION_CACHE.clear()


class TestConfigOption:
    def test_default_on(self):
        assert InfrastructureConfig().cache_macro_construction is True
        assert pyphi.config.infrastructure.cache_macro_construction is True

    def test_validation_rejects_non_bool(self):
        with pytest.raises(ValueError):
            InfrastructureConfig(cache_macro_construction="yes")

    def test_top_level_override_routes(self):
        with pyphi.config.override(cache_macro_construction=False):
            assert pyphi.config.infrastructure.cache_macro_construction is False
        assert pyphi.config.infrastructure.cache_macro_construction is True


class TestReuse:
    def test_mapped_variants_share_the_prefix(self):
        cache = macro_tpm_module._CONSTRUCTION_CACHE
        substrate = Substrate(CG_TPM)
        with config.override(**presets.iit4_2023):
            macro_tpms(substrate, CG_UNITS, (CG_STATE,))
            # 2 units, each: sequence miss + transition miss.
            assert (cache.misses, cache.hits) == (4, 0)
            assert cache.size == 4
            macro_tpms(substrate, CG_VARIANT_UNITS, (CG_STATE,))
            # Same footprints and grain: both sequence lookups hit;
            # Step 1 never reruns.
            assert (cache.misses, cache.hits) == (4, 2)

    def test_apportionment_separates_keys(self):
        cache = macro_tpm_module._CONSTRUCTION_CACHE
        substrate = Substrate(_bbx_micro_tpm())
        plain = BBX_UNITS[0]
        apportioned = MacroUnit(
            plain.constituents,
            plain.update_grain,
            plain.mapping,
            background_apportionment=(4, 5, 6, 7),
        )
        with config.override(**presets.iit4_2023):
            macro_tpms(substrate, (plain,), (BBX_ONES, BBX_ONES))
            misses_after_plain = cache.misses
            macro_tpms(substrate, (apportioned,), (BBX_ONES, BBX_ONES))
        # Same footprint and grain, different patron structure: the
        # apportioned construction must NOT reuse the plain entries.
        assert cache.misses == misses_after_plain + 2
        assert cache.hits == 0

    def test_flag_off_bypasses_cache_entirely(self):
        cache = macro_tpm_module._CONSTRUCTION_CACHE
        substrate = Substrate(CG_TPM)
        with config.override(**presets.iit4_2023, cache_macro_construction=False):
            macro_tpms(substrate, CG_UNITS, (CG_STATE,))
            macro_tpms(substrate, CG_VARIANT_UNITS, (CG_STATE,))
        assert cache.size == 0
        assert (cache.misses, cache.hits) == (0, 0)


class TestSweepReuse:
    def test_default_sweep_shares_across_variants(self, monkeypatch):
        """Mirrors the measured redundancy: the default complexes() sweep on
        the Example 1 substrate performs ~162 per-unit constructions over 6
        distinct (footprint, grain) keys."""
        from pyphi.macro.search import SearchBounds
        from pyphi.macro.search import complexes

        calls = {"n": 0}
        real = macro_tpm_module._discounted_on_probabilities

        def counting(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr(macro_tpm_module, "_discounted_on_probabilities", counting)
        with config.override(**presets.iit4_2023, progress_bars=False):
            with config.override(cache_macro_construction=False):
                result_off = complexes(Substrate(CG_TPM), CG_STATE, SearchBounds())
            off_calls = calls["n"]
            calls["n"] = 0
            macro_tpm_module._CONSTRUCTION_CACHE.clear()
            result_on = complexes(Substrate(CG_TPM), CG_STATE, SearchBounds())
            on_calls = calls["n"]
        # Step 1 runs once per distinct key instead of once per construction.
        assert 0 < on_calls < off_calls
        assert off_calls >= 5 * on_calls
        # And the sweep outcome is identical, exactly.
        assert len(result_on.records) == len(result_off.records)
        for on_record, off_record in zip(
            result_on.records, result_off.records, strict=True
        ):
            assert on_record.system == off_record.system
            assert on_record.phi == off_record.phi
        assert result_on.complexes == result_off.complexes
        assert result_on.ties == result_off.ties


def _factor_bytes(factored_pair):
    """All factor arrays of a (T_c, T_e) pair as bytes, order-stable."""
    return tuple(
        (tpm.factor(i) + 0.0).tobytes()
        for tpm in factored_pair
        for i in range(tpm.n_nodes)
    )


BBX_APPORTIONED_UNIT = MacroUnit(
    BBX_UNITS[0].constituents,
    BBX_UNITS[0].update_grain,
    BBX_UNITS[0].mapping,
    background_apportionment=(4, 5, 6, 7),
)

CONSTRUCTION_CASES = {
    "cg": (lambda: Substrate(CG_TPM), CG_UNITS, (CG_STATE,)),
    "bbx_grain2": (
        lambda: Substrate(_bbx_micro_tpm()),
        BBX_UNITS,
        (BBX_ONES, BBX_ONES),
    ),
    "bbx_apportioned": (
        lambda: Substrate(_bbx_micro_tpm()),
        (BBX_APPORTIONED_UNIT,),
        (BBX_ONES, BBX_ONES),
    ),
}


class TestByteIdentity:
    @pytest.mark.parametrize("name", sorted(CONSTRUCTION_CASES))
    def test_cache_on_off_and_hit_paths_agree_exactly(self, name):
        make_substrate, units, history = CONSTRUCTION_CASES[name]
        cache = macro_tpm_module._CONSTRUCTION_CACHE
        with config.override(**presets.iit4_2023):
            with config.override(cache_macro_construction=False):
                off = _factor_bytes(macro_tpms(make_substrate(), units, history))
            substrate = make_substrate()
            cold = _factor_bytes(macro_tpms(substrate, units, history))
            hits_before = cache.hits
            warm = _factor_bytes(macro_tpms(substrate, units, history))
            assert cache.hits > hits_before  # the second build hit the cache
        assert cold == off
        assert warm == off


def _perturbed_cg_tpm(value):
    tpm = np.array(CG_TPM, copy=True)
    tpm[0, 0] = value
    return tpm


class TestIsolation:
    def test_substrates_with_identical_unit_keys_never_share(self):
        """Two substrates, same units/footprint/grain/apportionment — the
        only key difference is the substrate fingerprint."""
        a = Substrate(CG_TPM)
        b = Substrate(_perturbed_cg_tpm(0.123))
        assert a._fingerprint != b._fingerprint
        with config.override(**presets.iit4_2023):
            macro_tpms(a, CG_UNITS, (CG_STATE,))
            b_cached = _factor_bytes(macro_tpms(b, CG_UNITS, (CG_STATE,)))
            with config.override(cache_macro_construction=False):
                b_fresh = _factor_bytes(
                    macro_tpms(
                        Substrate(_perturbed_cg_tpm(0.123)), CG_UNITS, (CG_STATE,)
                    )
                )
                a_fresh = _factor_bytes(
                    macro_tpms(Substrate(CG_TPM), CG_UNITS, (CG_STATE,))
                )
        assert b_cached == b_fresh  # b never picked up a's entries
        assert b_cached != a_fresh  # the perturbation is visible through it


class TestLifetime:
    def test_entries_evicted_when_substrate_dies(self):
        cache = macro_tpm_module._CONSTRUCTION_CACHE
        # A fingerprint unique to this test, so no fixture keeps it alive.
        substrate = Substrate(_perturbed_cg_tpm(0.321))
        with config.override(**presets.iit4_2023):
            macro_tpms(substrate, CG_UNITS, (CG_STATE,))
        assert cache.size > 0
        del substrate
        for _ in range(5):
            gc.collect()
            if cache.size == 0:
                break
        assert cache.size == 0


class TestConcurrency:
    def test_concurrent_variant_construction_is_consistent(self):
        """Concurrent constructions sharing cache keys produce the same
        bytes as a cache-off build (exercised under the free-threaded CI
        lane; a benign double-compute is allowed, corruption is not)."""
        substrate = Substrate(CG_TPM)
        variant_sets = [CG_UNITS, CG_VARIANT_UNITS] * 4
        with config.override(**presets.iit4_2023):
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(
                    pool.map(
                        lambda units: _factor_bytes(
                            macro_tpms(substrate, units, (CG_STATE,))
                        ),
                        variant_sets,
                    )
                )
            with config.override(cache_macro_construction=False):
                expected = {
                    id(units): _factor_bytes(macro_tpms(substrate, units, (CG_STATE,)))
                    for units in (CG_UNITS, CG_VARIANT_UNITS)
                }
        for units, result in zip(variant_sets, results, strict=True):
            assert result == expected[id(units)]
