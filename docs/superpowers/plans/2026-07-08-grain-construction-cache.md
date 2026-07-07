# Macro-Construction Intermediate Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement lever 1 of
`docs/superpowers/specs/2026-07-07-grain-discovery.md` (§3.1): a cache of the
mapping-independent Steps 1–2 intermediates of the macro TPM construction
(Marshall et al. 2024, Eqs. 26–31), keyed on
`(constituent footprint, update grain τ, apportionment key)` per substrate, so
that repeated grain-search candidates sharing a footprint and grain reuse the
expensive mapping-independent prefix instead of recomputing it per mapped
variant. Result-preserving: macro TPMs and sweep results must be byte-identical
with the cache on and off.

**Architecture:** A module-level
:class:`~pyphi.cache.content.ContentCache` in `pyphi/macro/tpm.py`, following
the existing `_PURVIEW_CACHE` pattern in `pyphi/substrate.py` exactly: entries
keyed on `(substrate._fingerprint, args)`, GC-driven eviction via
`observe(substrate, fingerprint)`, memory-pressure guard via
`cache_utils.memory_full()`, thread-safe by construction. Two entry kinds share
the cache: the discounted transition matrix (Step 1, key excludes τ) and the
per-τ sequence-class distribution (Steps 2+4a, key includes τ). The mapping
enters the construction only at Step 4's cheap compression, which stays
uncached. A new infrastructure config option `cache_macro_construction`
(default on) gates the cache; when off, the construction computes directly with
no cache reads or writes. No search-loop code changes: `macro/search.py`'s
per-run `system_cache` (whole constructed systems, keyed on canonical units
including mappings) is untouched and sits above this cache.

**Tech Stack:** Python 3.13, numpy, pytest. No new dependencies.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start — suggested: `macro-construction-cache`; base on
  the current working branch).
- Float comparisons in tests use `pytest.approx` (default tolerance) — never
  `==` on φ values — **except** in this plan's equivalence tests, whose entire
  claim is byte-identity: those deliberately use `.tobytes()` equality on
  arrays and exact `==` on φ values. Do not "fix" them to `approx`.
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task.
- Docstrings describe final state only — no migration narrative, no planning
  artifacts (no task numbers, no "lever 1", no spec references, no
  design-alternative discussion).
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- The final verification (Task 5) must run `uv run pytest` **with no path
  argument** at least once (bare paths skip the doctest sweep).

## Background for implementers (read once)

**The construction.** `pyphi/macro/tpm.py::macro_tpms(substrate, units,
micro_history)` builds the macro cause/effect TPMs. Its per-unit loop
(currently lines ~316–319) is:

```python
for j, unit in enumerate(units):
    on_probabilities = _discounted_on_probabilities(factored, units, j, patron)
    transition = _full_transition_matrix(on_probabilities)
    macro_prob_full = _unit_macro_probabilities(transition, unit)
    ...
```

The mapping-independent prefix is the first three calls up to (but not
including) the mapping compression:

1. `_discounted_on_probabilities` (Step 1, Eqs. 26–30): reads the substrate's
   factored TPM, the updating unit's `micro_constituents` (footprint), and the
   *patron structure* — `_patron_units(units)` plus, per patron unit `k`, the
   keep-set `set(units[k].micro_constituents) |
   set(units[k].background_apportionment)` (Eq. 29). Every other index falls to
   the global mean (Eq. 28) regardless of whether it is another system unit or
   unapportioned background. It does **not** read the unit's mapping, its
   grain, or `micro_history`.
2. `_full_transition_matrix` (Eq. 30): a pure function of the Step 1 output;
   `(2**n, 2**n)` — the Θ(4^n) floor.
3. `_unit_sequence_distributions` (Steps 2+4a, Eqs. 31, 35–36): reads the
   transition matrix, `unit.micro_constituents`, and `unit.micro_grain` (τ).
   Still mapping- and history-independent. Output `(2**n, 2**(m·τ))`.

The mapping enters only in `_unit_macro_probabilities`, which compresses the
sequence distribution through `unit.micro_mapping` — cheap. `micro_history`
enters only at Step 3 (`_background_weights_cause` / `_background_weights_effect`),
after the cached region.

Therefore the complete key for the sequence distribution is
`(substrate content, footprint, τ, apportionment key)` where the apportionment
key is the full `(background index → patron keep-set)` map; under the default
`apportionment="NONE"` it is empty. The transition matrix's key drops τ.
`micro_constituents` is a sorted tuple, so hierarchical (nested) units with
the same micro footprint and composed grain correctly share entries — Steps
1–2 read only the derived footprint and grain, never the nesting or mapping.

**Why it pays.** Measured in the spec: the mapping-independent share of
construction cost grows from 20% of construction at n = 4 micro units to 95%
at n = 8, and a default `complexes()` sweep on the 2024 paper's Example 1
performed 162 per-unit constructions over only **6 distinct
(footprint, grain) keys** — 27× key redundancy. The existing per-run
`system_cache` in `macro/search.py` cannot capture this: it keys on the full
canonical unit tuple *including mappings*, so mapped variants never share.

**The cache mechanism to reuse.** `pyphi/cache/content.py::ContentCache` —
already used at module level by `pyphi/substrate.py` (`_PURVIEW_CACHE`,
keyed on `_cm_fingerprint`) and per-function by
`pyphi/core/repertoire_algebra.py`. It provides: content-fingerprint keying
(two substrates with different TPMs can never collide —
`Substrate._fingerprint` is a blake2b-256 digest of alphabet sizes, all factor
bytes, and connectivity, exactly what `Substrate.__eq__` compares); refcounted
GC-driven eviction (`observe(source, fingerprint)` + weakref finalizers, so
entries die with the last live substrate carrying the fingerprint); a
`memory_full()` insertion guard tied to
`maximum_cache_memory_percentage`; thread safety (locked eviction
bookkeeping, lock-free hot path — a concurrent double-compute is benign
because the computation is deterministic); and automatic registration in the
cache-policy registry. It is covered by `test/cache/test_content_cache.py`
and `test/cache/test_content_cache_threadsafe.py` (the P6a free-threaded lane
runs these). A bespoke dict-with-eviction would have to re-solve eviction,
memory guarding, and locking; ContentCache is the smallest correct mechanism
here.

**Parallel search note.** Under the parallel macro search
(`parallel_macro_system_evaluation`), constructions happen in the driver
process (`_system_of_cached` builds every `MacroSystem` before dispatch;
workers only run `sia()`), so a module-level cache captures the full sharing.
Worker processes hold their own empty module cache and never construct. The
cache changes no values, so the parallel ≡ sequential invariant is untouched.

**Goldens.** `test/macro/test_macro_goldens.py` pins φ values at 1e-13; the
constructions it exercises (`CG_UNITS`, `BBX_UNITS`, the apportioned bbx unit)
are the equivalence fixtures for this plan. The perf-counter gate
(`test/integration/test_perf_counters.py`) pins call counts for
formalism/AC fixtures only — its pinned frames are `conf/:override`,
`relations.py:relations`, `repertoire_algebra.py:{cause,effect}_repertoire`,
and `system.py:find_mip`, and no pinned fixture runs macro code — so no pin
regeneration is expected (Task 5 verifies this explicitly).

Quick fixtures: `test/macro/test_macro_tpm.py` exports `CG_TPM` (the Example 1
micro TPM) and `_bbx_micro_tpm()`; `test/macro/test_macro_goldens.py` exports
`CG_UNITS` and `BBX_UNITS`. Cross-importing between test modules is the
established pattern in `test/macro/`.

---

### Task 1: Config option `cache_macro_construction`

**Files:**
- Modify: `pyphi/conf/infrastructure.py` (field + validation)
- Modify: `pyphi_config_3.0.yml` (reference config, next to the other cache options at lines ~50–51)
- Test: `test/macro/test_macro_construction_cache.py` (create; config tests only for now)

**Interfaces:**
- Produces: `config.infrastructure.cache_macro_construction: bool = True`,
  overridable top-level (`pyphi.config.override(cache_macro_construction=False)`)
  like every infrastructure option. Consumed by Task 2.

- [ ] **Step 1: Write the failing tests**

Create `test/macro/test_macro_construction_cache.py`:

```python
"""Tests for the macro-construction intermediate cache."""

import pytest

import pyphi
from pyphi.conf.infrastructure import InfrastructureConfig


class TestConfigOption:
    def test_default_on(self):
        assert InfrastructureConfig().cache_macro_construction is True
        assert pyphi.config.infrastructure.cache_macro_construction is True

    def test_validation_rejects_non_bool(self):
        with pytest.raises((TypeError, ValueError)):
            InfrastructureConfig(cache_macro_construction="yes")

    def test_top_level_override_routes(self):
        with pyphi.config.override(cache_macro_construction=False):
            assert pyphi.config.infrastructure.cache_macro_construction is False
        assert pyphi.config.infrastructure.cache_macro_construction is True
```

Note: check how `_check_bool` failures surface (`TypeError` vs `ValueError`)
by reading a neighboring option's validation, and narrow the `pytest.raises`
to the actual exception type.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/macro/test_macro_construction_cache.py -v`
Expected: FAIL with `AttributeError`/`TypeError` (unknown field).

- [ ] **Step 3: Implement**

In `pyphi/conf/infrastructure.py`, add after `cache_potential_purviews`
(line ~88):

```python
    # When True (default), the macro TPM construction caches its
    # mapping-independent Steps 1-2 intermediates (the discounted transition
    # matrix and the per-grain sequence-class distributions) per substrate, so
    # candidate units sharing a footprint, update grain, and apportionment
    # structure reuse them. Results are identical either way; set False to
    # disable the cache entirely (no reads or writes).
    cache_macro_construction: bool = True
```

And in `__post_init__`, next to the other cache checks:

```python
        _check_bool("cache_macro_construction", self.cache_macro_construction)
```

In `pyphi_config_3.0.yml`, add under the other cache lines:

```yaml
  cache_macro_construction: true
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/macro/test_macro_construction_cache.py test/conf/ -v`
Expected: new tests pass; no existing config-layer test breaks (the field-name
collision checks in `test/conf/test_config_layers.py` iterate
`fields(InfrastructureConfig)` — a new non-colliding name is fine).

- [ ] **Step 5: Commit**

No changelog fragment yet — the option is inert until Task 2 wires it; the
fragment there covers both.

```bash
git add pyphi/conf/infrastructure.py pyphi_config_3.0.yml test/macro/test_macro_construction_cache.py
git commit -m "Add cache_macro_construction infrastructure option"
```

---

### Task 2: The construction-intermediate cache in `pyphi/macro/tpm.py`

**Files:**
- Modify: `pyphi/macro/tpm.py` (add `_CONSTRUCTION_CACHE`, `_apportionment_key`,
  `_sequence_distributions_of`; rewire the `macro_tpms` loop; change
  `_unit_macro_probabilities` to take the sequence distribution)
- Test: `test/macro/test_macro_construction_cache.py` (extend)
- Create: `changelog.d/macro-construction-cache.optimization.md`

**Interfaces:**
- Produces:
  - `pyphi.macro.tpm._CONSTRUCTION_CACHE: ContentCache` (name
    `"macro.construction"`) — test-visible hit/miss/size counters and
    `clear()`.
  - `_apportionment_key(units) -> tuple` — the Step 1 patron structure:
    sorted `(background_index, patron_keep_set)` pairs; `()` under empty
    apportionment.
  - `_sequence_distributions_of(substrate, units, j, patron) -> np.ndarray` —
    Steps 1–2 for updating unit `j`, cached when
    `config.infrastructure.cache_macro_construction` is on, computed directly
    otherwise.
  - `_unit_macro_probabilities(sequence_dist, unit)` — signature change from
    `(transition, unit)`; Step 4 compression only. No callers outside the
    module (verified: `test/macro/test_macro_tpm.py` imports only
    `_discounted_on_probabilities` and `_full_transition_matrix`, both
    unchanged).
- Consumes: `cache_macro_construction` (Task 1),
  `pyphi.cache.content.ContentCache`, `Substrate._fingerprint`,
  `pyphi.utils.np_immutable`.

- [ ] **Step 1: Write the failing tests**

Append to `test/macro/test_macro_construction_cache.py`:

```python
import numpy as np

from pyphi import config
from pyphi.conf import presets
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

        monkeypatch.setattr(
            macro_tpm_module, "_discounted_on_probabilities", counting
        )
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
```

Note on the exact counts in `test_mapped_variants_share_the_prefix`: they
encode the two-entry-kinds design (one `sequence` and one `transition` entry
per distinct key on first construction; `sequence` hits on the variant). If
the implementation legitimately changes the entry layout, update the counts
with a comment — do not weaken to inequalities.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/macro/test_macro_construction_cache.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute
'_CONSTRUCTION_CACHE'`.

- [ ] **Step 3: Implement in `pyphi/macro/tpm.py`**

Add imports (`ContentCache` at module scope is safe — `pyphi/substrate.py`,
which this module already imports transitively, does the same):

```python
from pyphi import utils
from pyphi.cache.content import ContentCache
```

Add after `_patron_units`:

```python
_CONSTRUCTION_CACHE = ContentCache("macro.construction")


def _apportionment_key(units) -> tuple:
    """The patron structure Step 1 reads (Eq. 29), as a canonical key.

    For each apportioned background index, its patron unit's keep-set
    (the patron's micro constituents plus its apportionment), sorted by
    background index. Empty when no unit is apportioned — Step 1 then
    depends only on the substrate and the updating unit's footprint.
    """
    key = []
    for unit in units:
        if unit.background_apportionment:
            keep = tuple(
                sorted(
                    set(unit.micro_constituents)
                    | set(unit.background_apportionment)
                )
            )
            key.extend((w, keep) for w in unit.background_apportionment)
    return tuple(sorted(key))


def _sequence_distributions_of(substrate, units, j, patron) -> np.ndarray:
    """Steps 1-2 for updating unit ``j`` (Eqs. 26-31, 35-36), cached.

    These intermediates do not read the unit's mapping or the micro
    history: Step 1 depends on the substrate TPM, the unit's footprint,
    and the patron structure; Step 2 adds only the micro grain. They are
    cached per substrate content fingerprint under
    ``(footprint, micro_grain, apportionment key)`` — the discounted
    transition matrix separately under ``(footprint, apportionment key)``
    so different grains share it — and reused by every unit variant with
    the same key. Entries are evicted when the last live substrate
    carrying the fingerprint is garbage-collected. Cached arrays are
    immutable. Disabled (no reads or writes) when
    ``config.infrastructure.cache_macro_construction`` is off.
    """
    from pyphi.conf import config as _config

    unit = units[j]
    factored = substrate.factored_tpm

    def compute_transition():
        return utils.np_immutable(
            _full_transition_matrix(
                _discounted_on_probabilities(factored, units, j, patron)
            )
        )

    if not _config.infrastructure.cache_macro_construction:
        return _unit_sequence_distributions(compute_transition(), unit)

    fingerprint = substrate._fingerprint
    app_key = _apportionment_key(units)
    _CONSTRUCTION_CACHE.observe(substrate, fingerprint)

    def compute_sequence():
        transition = _CONSTRUCTION_CACHE.get_or_compute(
            fingerprint,
            ("transition", unit.micro_constituents, app_key),
            compute_transition,
        )
        return utils.np_immutable(_unit_sequence_distributions(transition, unit))

    return _CONSTRUCTION_CACHE.get_or_compute(
        fingerprint,
        ("sequence", unit.micro_constituents, unit.micro_grain, app_key),
        compute_sequence,
    )
```

Change `_unit_macro_probabilities` to consume the sequence distribution
(deleting its internal `_unit_sequence_distributions` call; the docstring
keeps the Eq. 35 reference and documents the new argument):

```python
def _unit_macro_probabilities(
    sequence_dist: np.ndarray, unit: MacroUnit
) -> np.ndarray:
    """Eq. 35: probability of each macro state of ``J`` per starting state.

    Compresses the ``U^J`` sequence-class distribution through the
    unit's composed truth table ``g_J``.

    Returns:
        np.ndarray: ``(2**n, 2)``.
    """
    table = np.asarray(unit.micro_mapping)
    return np.stack(
        [
            sequence_dist[:, table == 0].sum(axis=1),
            sequence_dist[:, table == 1].sum(axis=1),
        ],
        axis=1,
    )
```

Rewire the `macro_tpms` loop (lines ~316–319):

```python
    for j, unit in enumerate(units):
        sequence_dist = _sequence_distributions_of(substrate, units, j, patron)
        macro_prob_full = _unit_macro_probabilities(sequence_dist, unit)
```

(The rest of the loop body is unchanged. Also update the module docstring's
last sentence to mention that the mapping-independent Steps 1–2 intermediates
are cached per substrate — final-state prose only.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/macro/test_macro_construction_cache.py test/macro/ -v -m "not slow"`
Expected: all new tests pass; every existing (non-slow) macro test still
passes on the rewired path.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/macro-construction-cache.optimization.md <<'EOF'
The macro TPM construction now caches its mapping-independent Steps 1-2
intermediates (the discounted transition matrix and the sequence-class
distributions, Eqs. 26-31 of Marshall et al. 2024) per substrate, keyed on
the unit's footprint, update grain, and apportionment structure — so
grain-search candidates that differ only in their mapping reuse the
expensive construction prefix. Results are identical with the cache on or
off. Gated by the new infrastructure option `cache_macro_construction`
(default on); entries are evicted when the substrate is garbage-collected.
EOF
git add pyphi/macro/tpm.py test/macro/test_macro_construction_cache.py changelog.d/macro-construction-cache.optimization.md
git commit -m "Cache mapping-independent macro-construction intermediates"
```

---

### Task 3: Byte-identity equivalence across the golden constructions

The cache's correctness contract is byte-identity, not tolerance equality:
a cache hit returns the array a fresh deterministic computation would produce,
bit for bit. This task pins that directly on the three golden construction
fixtures (coarse-grain, black-box at grain 2, apportioned black-box — together
covering τ > 1, Eq. 29 apportionment, and both entry kinds).

**Files:**
- Test: `test/macro/test_macro_construction_cache.py` (extend)

**Interfaces:** none new; consumes Task 2.

- [ ] **Step 1: Write the tests**

Append:

```python
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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest test/macro/test_macro_construction_cache.py -v`
Expected: all pass. (These tests pass immediately after Task 2 — they are the
regression net for the equivalence claim, not TDD drivers. If any byte
comparison fails, the cache key is incomplete: stop and diagnose which input
of Steps 1–2 the key misses before touching anything else.)

- [ ] **Step 3: Run the full macro golden battery, slow lane included**

Run: `uv run pytest test/macro/ -v` (background if desired; the slow bbx
sweeps run in ~1 min total; leave `PYPHI_MACRO_FULL_SWEEP` unset)
Expected: all pass — the goldens reproduce through the cached path at their
committed 1e-13 pins.

- [ ] **Step 4: Commit**

```bash
git add test/macro/test_macro_construction_cache.py
git commit -m "Pin byte-identity of macro construction with cache on and off"
```

---

### Task 4: Isolation, lifetime, and concurrency

**Files:**
- Test: `test/macro/test_macro_construction_cache.py` (extend)

**Interfaces:** none new; consumes Task 2.

- [ ] **Step 1: Write the tests**

Append:

```python
import gc
from concurrent.futures import ThreadPoolExecutor


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
        gc.collect()
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
```

Notes for the implementer:
- If the eviction assertion is flaky because a finalizer needs a second
  collection pass, mirror the pattern in
  `test/cache/test_content_cache.py`'s eviction tests (loop `gc.collect()` a
  bounded number of times) rather than sleeping.
- `config.override` inside threads: the concurrency test reads config from
  worker threads. If the override context manager is not visible across
  threads (check how `test/cache/test_content_cache_threadsafe.py` handles
  config), set the preset around the whole test with the override applied
  before the pool starts, as written above — the workers only *read*.

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest test/macro/test_macro_construction_cache.py -v`
Expected: all pass. Isolation is delivered by the fingerprint key component;
if `test_substrates_with_identical_unit_keys_never_share` fails, the key is
missing the substrate component — stop and fix, do not weaken the test.

- [ ] **Step 3: Commit**

```bash
git add test/macro/test_macro_construction_cache.py
git commit -m "Test substrate isolation, GC eviction, and concurrency of the construction cache"
```

---

### Task 5: Perf gates, docs close-out, and full verification

**Files:**
- Modify: `CLAUDE.md` (caching options list)
- Modify: `ROADMAP.md` (Status Dashboard)

**Interfaces:** none (verification + documentation).

- [ ] **Step 1: Perf-counter and perf-budget gates**

Run: `uv run pytest test/integration/test_perf_counters.py test/integration/test_perf_budget.py -q`

Expected: **all pass with zero pin changes.** The pinned fixtures exercise no
macro code and the pinned frames (`conf/:override`, relations, repertoires,
`find_mip`) are not called by the construction cache. If any pin fails,
**stop**: this means the change leaked into a non-macro path — diagnose the
cause. Regenerating pins (`uv run python scripts/gen_perf_counts.py`) is only
acceptable after the cause is understood and the regenerated diff is reviewed
and called out to the user explicitly, like a φ golden. Never regenerate
silently.

- [ ] **Step 2: Documentation**

- `CLAUDE.md`: in the "Caching (`config.infrastructure`)" list, add:
  `- **`cache_macro_construction`**: Cache mapping-independent macro-construction intermediates (default: true)`
- `ROADMAP.md`: add a Status Dashboard row following the neighboring rows'
  column format:

```markdown
| Macro-construction intermediate cache | ✅ landed | — | The macro TPM construction's mapping-independent Steps 1–2 intermediates (Eqs. 26–31) are cached per substrate on `(footprint, update grain, apportionment key)` beside the search's per-run system cache, so grain-search candidates differing only in mapping share the expensive construction prefix. Byte-identical results cache-on/off (pinned); GC-driven eviction via `ContentCache`; `cache_macro_construction` config option. |
```

- [ ] **Step 3: Full test suite (no path argument — includes the doctest sweep)**

Run: `uv run pytest -x -q`
Expected: all pass. If an unrelated-looking test fails, diagnose before
touching anything — other sessions may have concurrent working-tree changes;
only fix failures traceable to this plan's commits.

Then run the macro directory once more on its own for a focused signal:
`uv run pytest test/macro/ -q`

- [ ] **Step 4: Pre-commit hooks over the changed files**

Run: `uv run pre-commit run --files $(git diff --name-only $(git merge-base HEAD <base-branch>) | tr '\n' ' ')` — substitute the branch this worktree was created from.
Expected: all hooks pass (ruff, pyright, file checks). Fix findings and
follow-up commit as appropriate.

- [ ] **Step 5: Commit the documentation updates**

```bash
git add CLAUDE.md ROADMAP.md
git commit -m "Record the macro-construction intermediate cache in docs"
```

---

## Self-review notes

- **Key completeness.** Step 1 reads exactly: the factored TPM (covered by
  `Substrate._fingerprint`), the updating unit's `micro_constituents`, and the
  patron map with keep-sets (`_apportionment_key`); every other index falls to
  the global mean identically whether it is another system unit or
  unapportioned background (Eq. 28 branch). Step 2 adds `micro_constituents`
  (already keyed) and `micro_grain`. `micro_history` and the mapping are read
  only after the cached region (Steps 3–4), so their exclusion from the key is
  a code-reading fact, not an assumption; the byte-identity tests (Task 3) and
  the sweep equality (Task 2) guard it empirically.
- **Cache mechanism choice.** `ContentCache` over a bespoke per-run dict:
  it already provides fingerprint isolation, GC-driven eviction, the memory
  guard, thread safety (free-threaded lane covered by existing tests), and
  registry visibility — and the `_PURVIEW_CACHE` precedent makes the wiring
  one function. A per-run dict threaded through `MacroSystem.from_micro`
  would change public constructor signatures and would not serve direct
  `from_micro` callers (who produce exactly the 27× redundancy the spec
  measured when evaluating mapped variants by hand).
- **Cross-substrate collision is impossible by construction:**
  `Substrate._fingerprint` digests alphabet sizes, all factor bytes, and
  connectivity (what `__eq__` compares); Task 4 tests it anyway with two
  substrates differing in a single TPM entry.
- **Determinism/parallel invariant:** the cache changes no values (hit
  returns what the deterministic computation produced), constructions run in
  the driver process under parallel search, and no evaluation-order dependence
  is introduced — so the parallel ≡ sequential guarantee of the macro search
  is untouched.
- **Exact-count tests** (Task 2) pin the two-entry layout deliberately; the
  sweep test uses a ratio (≥5×, measured 27×) so it survives benign changes in
  search enumeration while still failing if sharing breaks.
- The grain-discovery exploration document is an untracked file and is not
  modified by this plan; its §3.1 design is implemented as specified
  (per-substrate keying strengthens the per-run scoping it sketched).
- Risk accepted: cached arrays are shared between constructions, so they are
  frozen with `np_immutable` at insertion; downstream code only reads them
  (einsum / masked sums), and any future in-place mutation would raise
  immediately rather than corrupt.
