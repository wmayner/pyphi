# PyPhi 2.0.0 Final Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Settle the public API of PyPhi 2.0.0 and fix every verified finding from the post-rc3 review, so the next tag can be the final release.

**Architecture:** Twelve sequential tasks on one worktree branch. API renames come first because later tasks build on the new names; correctness fixes next; then results, errors, the MCP server and the docs. Each task leaves the full fast suite green and commits.

**Tech Stack:** Python 3.13, uv, pytest (+ doctests over `pyphi/`), Sphinx/MyST docs (`just docs`, `-W`), towncrier fragments, msgspec serialization.

**Spec:** `docs/superpowers/specs/2026-10-11-final-polish-design.md` (decisions D1–D9). Finding IDs (`api-2`, `res-1`, …) refer to the review results file `results.json`, copied into the worktree at `.claude/review-2026-10-11.json` (untracked) before Task 1.

**Plan granularity.** Each task states the required behavior, the exact tests that pin it, and the checks to run. It does not spell out implementation code: writing that would mean reading roughly forty modules into the planning session. Implementers read the code they change; the tests below are the contract.

## Global Constraints

- Python 3.13+ only; no backward-compatibility code for older Pythons.
- Names that exist only in the 2.0 release candidates are renamed with no alias. Removed or renamed 1.x names raise `AttributeError` naming the replacement and get a migration-guide row.
- Files saved by rc1–rc3 must still load (renamed serialized fields: the decoder accepts the old name).
- No computed φ value changes unless the task says so and a test states the old and new value.
- Every test that asserts a φ value pins its formalism with a complete preset: `IIT_3_CONFIG` / `IIT_4_CONFIG` from `test/conftest.py`, or `pyphi.config.override(**pyphi.iit4_2026)` for the default theory. Never set `iit.version` alone.
- Terminology: "intrinsic-information requirement", never "cap". φ_s is `analysis.phi`, Φ is `analysis.big_phi`. States are little-endian.
- The docs present the default theory as "IIT"; earlier versions appear only on `docs/howto/earlier-versions.md` and in the configuration and migration references.
- Before writing public prose (docs, changelog fragments, docstrings), apply the `writing-naturally` skill.
- User-facing changes get a towncrier fragment in `changelog.d/` (`<name>.<type>.md`).
- Commit messages match the existing log: an imperative sentence summary, a body saying what changed and why, the session's attribution lines. Stage only your own files; other sessions may have changes in the tree.
- Verification: `uv run pytest -q -p no:cacheprovider > <log> 2>&1`, then read the summary line. Never pipe a test run through `tail`/`grep` to judge it. The bare invocation is required at least once per task because it runs the doctests.
- `ruff format` only the `.py` files you touched (it also rewrites code blocks in `pyphi/mcp/content/*.md`). No unsafe ruff fixes.
- Perf pins: if `test/integration/test_perf_counters.py` fails, run `uv run python scripts/gen_perf_counts.py` only after confirming every moved count is explained by the task; state the moved frames and their direction in the commit message.

## Review Focus

1. **Files saved by rc1–rc3** (analyses, SweepResults) must load after the field and schema changes in Tasks 3 and 8. Owning tasks add a round-trip test from a fixture written by the pre-change code.
2. **Multi-valued (k-ary) purview labels.** The label fix in Task 3 must write k-ary specified states correctly (for example `P₂` for value 2), not only binary upper/lower case. Task 3 pins a k-ary case.
3. **IIT 3.0 results after unifying `analysis.ces`** (Task 7): `len()`, iteration, `.phis`, display and save/load round-trip must all work. Task 7 pins each.
4. **Internal callers of the now-strict methods** (Task 2): campaign runners, the MCP server and the grain search pass options through `Substrate`/`System` methods. They must not start raising. Task 2 runs `test/campaign`, `test/mcp` and `test/macro` explicitly in addition to the suite.
5. **Config writes inside workers** (Task 4): `install_snapshot` in parallel workers replaces layers wholesale. Validation added to the write paths must not reject valid snapshots or slow the per-chunk install measurably. Task 4 runs `test/parallel` and the perf pins.

---

### Task 0: Worktree and baseline

- [ ] **Step 1:** Create the worktree: `git worktree add .claude/worktrees/final-polish -b claude/2-0-final-polish main`. In the worktree, `uv sync --group dev --group docs --all-extras` (see memory note: in worktrees `uv run` uses the worktree's `.venv`).
- [ ] **Step 2:** Copy the review results into the worktree as `.claude/review-2026-10-11.json` (untracked).
- [ ] **Step 3:** Baseline: full fast suite (expect 4774 passed, 296 skipped) and record the count.

### Task 1: Rename the shadowed modules and make `__all__` explicit (D3, D6)

**Files:**
- Move: `pyphi/analyze.py` → `pyphi/analysis.py`, `pyphi/sweep.py` → `pyphi/sweeps.py`, `pyphi/optimize.py` → `pyphi/optimization.py` (use `git mv`).
- Modify: `pyphi/__init__.py` (imports, explicit `__all__`, delete `_SHADOWED_BY_FUNCTION`), `pyphi/models/__init__.py` (`__all__` without underscore names or registries), `pyphi/actual.py` (`__all__`), every importer (`git grep -n "pyphi.analyze\b\|from pyphi.analyze\|from .analyze\|pyphi.sweep\b\|from .sweep\|pyphi.optimize\b\|from .optimize"`), `docs/reference/api.md`, docs that name the module paths (`docs/howto/read-result.md`, `docs/howto/landscape.md:151`, `docs/howto/campaigns.md`), `scripts/`, `benchmarks/benchmarks/`, `test/test_import_surface.py`.
- Test: `test/test_import_surface.py`.

**Interfaces:**
- Produces: modules `pyphi.analysis` (`analyze`, `Analysis`), `pyphi.sweeps` (`sweep`, `SweepResult`), `pyphi.optimization` (`optimize`, `OptimizationResult`). Top-level `pyphi.analyze`, `pyphi.sweep`, `pyphi.optimize` remain the functions.

- [ ] **Step 1: Write the failing tests** in `test/test_import_surface.py`:

```python
import importlib
import types

import pyphi


def test_entry_point_modules_are_modules_and_functions_stay_on_top():
    for module, function, cls in [
        ("analysis", "analyze", "Analysis"),
        ("sweeps", "sweep", "SweepResult"),
        ("optimization", "optimize", "OptimizationResult"),
    ]:
        mod = importlib.import_module(f"pyphi.{module}")
        assert isinstance(mod, types.ModuleType)
        assert getattr(mod, function) is getattr(pyphi, function)
        assert getattr(mod, cls) is getattr(pyphi, cls)


def test_star_import_does_not_rebind_common_names():
    namespace = {"log": "mine", "cache": "mine", "utils": "mine"}
    exec("from pyphi import *", namespace)
    assert namespace["log"] == "mine"
    assert namespace["cache"] == "mine"
    assert namespace["utils"] == "mine"


def test_all_lists_only_public_names():
    internal = {"deferred", "registry", "protocols", "utils", "node",
                "combinatorics", "serializable", "data_structures",
                "constants", "log", "cache", "TransitionSystem"}
    assert not internal & set(pyphi.__all__)
    assert all(hasattr(pyphi, name) for name in pyphi.__all__)
    for name in ("analyze", "sweep", "optimize", "Analysis", "SweepResult",
                 "OptimizationResult", "Substrate", "System"):
        assert name in pyphi.__all__


def test_models_all_has_no_private_names():
    assert not [n for n in pyphi.models.__all__ if n.startswith("_")]


def test_actual_declares_its_public_surface():
    assert "Transition" in pyphi.actual.__all__
```

- [ ] **Step 2:** Run them; expect failures on the module type and `__all__`.
- [ ] **Step 3:** `git mv` the three modules; update every importer found by the grep above; write the explicit `__all__` lists from `docs/reference/api.md` (public modules listed there plus the top-level functions and classes currently exported). Keep the 1.x `_RENAMED` hints in `pyphi.__getattr__`.
- [ ] **Step 4:** Add `pyphi.Analysis`, `pyphi.SweepResult`, `pyphi.OptimizationResult` and the three new modules to `docs/reference/api.md`; fix `landscape.md:151` to `pyphi.OptimizationResult.load`.
- [ ] **Step 5:** Full fast suite; `just docs` (succeeds with `-W`; confirm `docs/_build/html/reference/_autosummary/pyphi.analysis.html` exists and documents `Analysis`). Grep build and CI files for the old paths (`.github/`, `justfile`, `pyproject.toml`, `.pre-commit-config.yaml`).
- [ ] **Step 6:** Fragment `changelog.d/module-names.change.md`; commit.

### Task 2: `subset` everywhere and no silently ignored keyword arguments (D1, D2)

**Files:**
- Modify: `pyphi/substrate.py` (`sia`, `ces`, `phi_structure`, `complexes`, `maximal_complex`, `irreducible_sias`, `all_sias`), `pyphi/system.py` (`System.__init__` signature, delete `from_substrate`, `sia`, `ces`, `distinctions`, `mic`, `mie`, `phi`), `pyphi/formalism/iit4/__init__.py` and `pyphi/formalism/iit3/__init__.py` (`sia(**kwargs)` entry points), `pyphi/actual.py` (`events`, `true_events` parameters), every caller of `System.from_substrate` and of `indices=`/`nodes=`/`node_indices=` as constructor arguments (`git grep -n "from_substrate\|indices=\|nodes="`), docs and MCP content that show these calls.
- Test: new `test/test_entry_point_arguments.py`.

**Interfaces:**
- Consumes: Task 1 module names.
- Produces: `Substrate.sia(state, *, subset=None, formalism=None, <explicit formalism options>)`; same for `ces`, `phi_structure`; `Substrate.complexes(state, *, candidates=None, formalism=None, parallel=None, progress=None)` and the same for `maximal_complex`, `irreducible_sias`, `all_sias`; `System(substrate, state, *, subset=None, partition=None, external_indices=None, background_conditioning=None, background_state=None)`; `System.sia(*, formalism=None, ...)`, `System.ces(*, formalism=None, ...)`. Unknown keywords raise `TypeError` whose message names the argument.

- [ ] **Step 1: Write the failing tests:**

```python
import pytest

import pyphi
from pyphi import examples


def test_substrate_sia_takes_subset_like_analyze():
    with pyphi.config.override(**pyphi.iit4_2026):
        s = examples.iit4_2023_fig1a_substrate()
        expected = pyphi.analyze(s, (0, 1, 1), subset=(0, 1)).phi
        sia = s.sia((0, 1, 1), subset=(0, 1))
    assert sia.node_indices == (0, 1)
    assert sia.phi == pytest.approx(expected)


@pytest.mark.parametrize("call", [
    lambda s: s.sia((0, 1, 1), partition_schme="X"),
    lambda s: s.sia((0, 1, 1), indices=(0, 1)),
    lambda s: s.complexes((0, 1, 1), bogus=1),
    lambda s: pyphi.System(s, (0, 1, 1)).sia(bogus=1),
    lambda s: pyphi.System(s, (0, 1, 1)).distinctions(bogus=1),
    lambda s: pyphi.System(s, (0, 1, 1)).mic((0,), bogus=1),
])
def test_unknown_keywords_raise(call):
    with pytest.raises(TypeError, match="bogus|partition_schme|indices"):
        call(examples.iit4_2023_fig1a_substrate())


def test_complexes_honors_formalism():
    s = examples.basic_noisy_selfloop_substrate()
    with pyphi.config.override(**pyphi.iit3):
        expected = [(c.node_indices, round(c.phi, 6)) for c in s.complexes((1, 0, 0))]
    got = [(c.node_indices, round(c.phi, 6))
           for c in s.complexes((1, 0, 0), formalism="IIT_3_0")]
    assert got == expected


def test_system_constructor_takes_subset_keyword_only():
    s = examples.basic_substrate()
    assert pyphi.System(s, (1, 0, 0), subset=(0, 1)).node_indices == (0, 1)
    with pytest.raises(TypeError):
        pyphi.System(s, (1, 0, 0), (0, 1))
    assert not hasattr(pyphi.System, "from_substrate")
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. Where a method forwards options to the formalism, accept the formalism's documented option names explicitly or validate against them; resolve `formalism=` with `presets.canonical` and apply its preset with `config.override` for the call, as `analyze` does.
- [ ] **Step 4:** Run `test/campaign`, `test/mcp`, `test/macro`, `test/parallel` explicitly (Review Focus 4), then the full fast suite.
- [ ] **Step 5:** Update docs/MCP content/skill that call `System.from_substrate`, `indices=` or `nodes=`. Fragment `changelog.d/subset-everywhere.change.md`. Commit.

### Task 3: Intrinsic specification naming and purview labels in the specified state (D5)

**Files:**
- Modify: `pyphi/models/state_specification.py`, `pyphi/models/ria.py` (`purview_label`, `purview_state` → `current_state`), `pyphi/models/distinction.py:238-247`, `pyphi/models/distinctions.py:120`, `pyphi/visualize/projection/__init__.py:196`, `pyphi/formalism/iit4/__init__.py` and `formalism.py` (attribute uses), `pyphi/core/repertoire_algebra.py`, `pyphi/system.py`, `pyphi/serialize/schema.py` and its codec (renamed field, old name accepted on decode), `pyphi/analysis.py` (`_system_rows`: omit `ii(s)` under `IIT_4_0_2023`), `pyphi/display/*` labels, `explain()` labels; docs: `docs/theory/system-integration.md`, `overview.md`, `phi-structure.md`, `docs/reference/glossary.md`, `docs/howto/read-result.md`, `docs/howto/earlier-versions.md`, MCP content. Update tests that use the old attribute (`git grep -n "intrinsic_information\|purview_state" -- test`), keeping `sia.intrinsic_information` (ii(s)).
- Test: new `test/models/test_specified_state_labels.py`; serialization round-trip in `test/serialize/`.

**Interfaces:**
- Produces: `StateSpecification.intrinsic_specification`, `.runner_up_intrinsic_specification`; `MICE/RIA.current_state` (was `purview_state`); `sia.intrinsic_information` unchanged (ii(s)).

- [ ] **Step 1: Write the failing tests:**

```python
import numpy as np
import pytest

import pyphi
from pyphi import examples


def _labels(analysis):
    return {
        d.mechanism: (d.cause_purview_label, d.effect_purview_label)
        for d in analysis.ces.distinctions
    }


def test_purview_labels_use_the_specified_state():
    # basic_substrate in (1, 1, 0): current and specified states differ.
    with pyphi.config.override(**pyphi.iit4_2026):
        a = pyphi.analyze(examples.basic_substrate(), (1, 1, 0))
    assert _labels(a) == {
        (1,): ("C", "A"),
        (2,): ("ab", "b"),
        (0, 1): ("C", "Ac"),
    }
    for d in a.ces.distinctions:
        assert d.cause_purview_label == d.cause.specified_state.purview_label


def test_kary_purview_label_uses_the_specified_value():
    # Build a 2-unit substrate with a 3-valued unit whose specified state differs
    # from its current state; assert the label carries the specified value.
    # (Implementer: construct with Substrate(marginals=..., state_space=...) so that
    # the effect of unit 0 specifies value 2 while unit 0 is currently 0, and
    # assert the label equals the StateSpecification.purview_label for it.)
    ...


def test_per_direction_quantity_is_called_intrinsic_specification():
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1),
                            subset=(0, 1), compute="sia")
    cause = sia.system_state.cause
    assert cause.intrinsic_specification == pytest.approx(0.843228, abs=1e-6)
    assert not hasattr(cause, "intrinsic_information")
    assert sia.intrinsic_information == pytest.approx(sia.phi)  # ii(s) binds here


def test_2023_analysis_card_has_no_ii_row():
    a = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1),
                      subset=(0, 1), formalism="IIT_4_0_2023")
    assert "ii(s)" not in str(a)


def test_rc3_saved_state_specification_still_loads(tmp_path):
    # Fixture: a SIA saved by the pre-change code (write it in Step 0 of this task
    # from the Task 2 commit and commit it under test/data/serialize/).
    ...
```

The two `...` tests must be completed by the implementer before Step 2; their assertions are stated in the comments. In Step 0, before changing any code, save `pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0,1,1), subset=(0,1))` under the 2026 preset to `test/data/serialize/rc3_fig1a_ab_analysis.json` with the current code, and commit it with the tests.

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Rename attributes and serialized fields (decoder accepts the old name); switch every purview label to the specified state (`StateSpecification.purview_label` already does this); rename `purview_state` to `current_state`; relabel cards and `explain()`; drop the `ii(s)` row for 2023 results.
- [ ] **Step 4:** Fix the docs listed above: the specified state maximizes intrinsic specification (Mayner, Marshall & Tononi 2026, Eq. 12); ii(s) appears only as the min term; `phi-structure.md` gets separate rows for i_spec (`sia.intrinsic_specification`) and ii(s) (`sia.intrinsic_information`).
- [ ] **Step 5:** Full fast suite and `just docs`. Fragment `changelog.d/intrinsic-specification-names.change.md` and `changelog.d/purview-labels.fix.md`. Commit.

### Task 4: Validate config writes on every path (D8, findings mcp-1, err-3, port-4, err-11, drift-5, port-11)

**Files:**
- Modify: `pyphi/conf/_global.py` (`__setitem__`, `__setattr__` layer and sub-namespace replacement branches, ambiguity message), the frozen layer classes in `pyphi/conf/formalism.py`, `infrastructure.py`, `numerics.py` (assigning a field raises `ConfigurationError` that shows the supported write), `pyphi/conf/_field_routing.py` (`RENAMED_FIELDS` filled from the migration table; near-match suggestions with `difflib.get_close_matches`), `docs/migration/migration-2.0.md`, `docs/whats-new-in-2.0.md` (uppercase sentence), `pyphi/mcp/content/configuration.md`.
- Test: `test/conf/test_config_writes.py`.

- [ ] **Step 1: Write the failing tests:**

```python
import pytest

import pyphi
from pyphi.conf import ConfigurationError


@pytest.fixture(autouse=True)
def _restore():
    with pyphi.config.override():
        yield


def test_version_alone_raises_on_item_assignment():
    with pytest.raises(ConfigurationError):
        pyphi.config["iit.version"] = "IIT_4_0_2023"


def test_unknown_version_raises_at_assignment():
    with pytest.raises(ConfigurationError):
        pyphi.config["iit.version"] = "IIT4"


def test_ambiguity_message_recommends_a_preset():
    with pytest.raises(ConfigurationError, match="iit4_2023|formalism="):
        pyphi.config.version = "IIT_4_0_2023"


def test_writing_through_a_layer_says_how_to_set_it():
    with pytest.raises(ConfigurationError, match=r"config\.precision = "):
        pyphi.config.numerics.precision = 6


def test_typo_suggests_the_option():
    with pytest.raises(ConfigurationError, match="progress_bars"):
        pyphi.config.progres_bars = False


def test_presets_and_overrides_still_apply():
    with pyphi.config.override(**pyphi.iit3):
        assert pyphi.config.formalism.iit.version == "IIT_3_0"
    pyphi.config.precision = 6
    assert pyphi.config.precision == 6
```

- [ ] **Step 2:** Run; expect failures.
- [ ] **Step 3:** Implement. `install_snapshot` (worker path) may skip validation, because snapshots come from a validated parent config; keep it fast.
- [ ] **Step 4:** Run `test/conf`, `test/parallel`, `test/integration/test_perf_counters.py`, then the full fast suite.
- [ ] **Step 5:** Docs: migration guide says to set options with a top-level write, `config[...]` or `override`; whats-new describes the uppercase behavior accurately; MCP configuration reference matches. Fragment `changelog.d/config-write-validation.fix.md`. Commit.

### Task 5: Grain search excludes reducible candidates; `ComplexesResult` card (D8, findings api-1, res-5)

**Files:**
- Modify: `pyphi/macro/search.py` (`complexes`, pruned path), `pyphi/condensation.py` only if the filter belongs there for every caller (check `Substrate.complexes`, which filters through `irreducible_sias` already), `ComplexesResult` display (`pyphi/macro/`), `pyphi/display/` as needed.
- Test: `test/macro/test_grain_search_complexes.py`.

- [ ] **Step 1: Write the failing tests:**

```python
import pyphi
from pyphi import examples


def test_grain_search_returns_no_complex_when_every_candidate_is_reducible():
    with pyphi.config.override(**pyphi.iit4_2026):
        for substrate, state in [(examples.basic_substrate(), (1, 0, 0)),
                                 (examples.xor_substrate(), (0, 0, 0))]:
            assert substrate.complexes(state) == ()
            result = pyphi.analyze(substrate, state, grains=True)
            assert tuple(result.complexes) == ()


def test_grain_search_complexes_all_have_positive_phi():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1),
                               grains=True)
    assert result.complexes
    assert all(c.phi > 0 for c in result.complexes)


def test_complexes_result_prints_compactly():
    with pyphi.config.override(**pyphi.iit4_2026):
        result = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1),
                               grains=True)
    assert len(repr(result)) < 3000
    assert hasattr(result, "_repr_html_")
```

- [ ] **Step 2–4:** Run (fail), implement the φ_s > 0 filter (`numerics.is_positive`) before the exclusion cascade in both paths and give `ComplexesResult` a card (complexes with φ_s, units and grains; record count; tie count; formalism), run `test/macro` and the full fast suite. Run the slow macro reproductions: `uv run pytest -m slow --slow -q -p no:cacheprovider test/macro test/integration/test_paper_reproduction.py -k "marshall or macro or grain"` and read the summary.
- [ ] **Step 5:** Fragment `changelog.d/grain-search-reducible.fix.md`; commit.

### Task 6: `explain()` runner-up gap and how the MIP is chosen (D8, finding res-4)

**Files:**
- Modify: `pyphi/formalism/iit4/__init__.py:551-558` and `:946-957`, `pyphi/models/partitions.py` (`concise_partition`: directed partitions over the same blocks print differently, for example with the severed direction), `docs/howto/read-result.md:65-66`, `docs/theory/system-integration.md:75`.
- Test: `test/formalism/test_explain_runner_up.py`.

- [ ] **Step 1: Write the failing tests:**

```python
import pyphi
from pyphi import examples


def _findings(sia):
    return {f.kind: f.value for f in sia.explain().findings}


def test_runner_up_gap_is_not_negative_when_the_requirement_binds():
    with pyphi.config.override(**pyphi.iit4_2026):
        sia = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1),
                            subset=(0, 1), compute="sia")
    f = _findings(sia)
    assert f["gap"] >= 0
    assert f["runner_up"] != str(f.get("winning_partition", ""))
```

(Implementer: the finding kinds are those `explain()` emits today; adjust the keys to the actual kind names, keeping both assertions.)

- [ ] **Step 2–4:** Run (fail); compute the gap on values without the intrinsic-information requirement, as `partition_margin` does; make `concise_partition` distinguish directed partitions; correct both docs passages (the MIP is chosen on normalized φ before the requirement applies). Full fast suite and `just docs`.
- [ ] **Step 5:** Fragment `changelog.d/explain-runner-up.fix.md`; commit.

### Task 7: Results API: one `ces` type, tuples, frozensets, equality, aliases, example names (D7)

**Files:**
- Modify: `pyphi/analysis.py` (`Analysis.ces`), `pyphi/formalism/iit3/__init__.py` (build a `CauseEffectStructure` with empty relations), `pyphi/models/distinctions.py` (`phis`, `mechanisms`, purview properties return tuples), `pyphi/relations.py` (`Relation.purview` frozenset + docstring), `pyphi/landscape.py` (`LandscapeSection.__eq__`, `__hash__ = None` on `LandscapeSection` and `Perturbation`), `pyphi/relations.py` or `pyphi/models` (`AnalyticalRelations.to_pandas` returns the closed-form summary), `pyphi/models/__init__.py` and `pyphi/system.py` (remove `Concept`, `null_concept`; `apply_cut` → `apply_partition`; card row "Partition"), `pyphi/convert.py` (remove the twelve short names), `pyphi/examples.py` (rename the fifteen `fig*` functions to `iit3_2014_fig*`; `__getattr__` map for the 1.x names `fig1a`…`fig16`, `fig6/8/9/10/14`, `actual_causation`, `prevention`), the example registry, every caller (`git grep`), docs, MCP content (`list_examples` names), migration guide rows.
- Test: `test/test_results_api.py`.

- [ ] **Step 1: Write the failing tests:**

```python
import pytest

import pyphi
from pyphi import examples
from pyphi.models import CauseEffectStructure


def test_analysis_ces_has_one_type_under_every_version():
    s, st = examples.xor_substrate(), (0, 0, 0)
    for formalism in ("IIT_3_0", "IIT_4_0_2023", "IIT_4_0_2026"):
        ces = pyphi.analyze(s, st, formalism=formalism).ces
        assert isinstance(ces, CauseEffectStructure)
        assert len(ces.distinctions) == len(tuple(ces.distinctions))
        assert isinstance(ces.distinctions.phis, tuple)


def test_iit3_ces_round_trips(tmp_path):
    ces = pyphi.analyze(examples.xor_substrate(), (0, 0, 0), formalism="IIT_3_0").ces
    pyphi.save(ces, tmp_path / "c.json")
    assert pyphi.load(tmp_path / "c.json") == ces
    assert str(ces)


def test_relation_purview_is_always_frozenset():
    a = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1))
    assert {type(r.purview) for r in a.ces.relations.materialize()} == {frozenset}


def test_landscape_section_equality_and_hash():
    axis = pyphi.weight_axes("sigmoid", [[0, 0.8], [0.8, 0]], [(0, 1)])
    one = pyphi.landscape_section(lambda t: axis([t]), (1, 0), [0.5, 0.6], progress=False)
    two = pyphi.landscape_section(lambda t: axis([t]), (1, 0), [0.5, 0.6], progress=False)
    assert one == two
    with pytest.raises(TypeError):
        hash(one)


@pytest.mark.parametrize("old,new", [
    ("fig1a_substrate", "iit3_2014_fig1a_substrate"),
    ("fig1a", "iit3_2014_fig1a_substrate"),
    ("fig4", "iit3_2014_fig4_substrate"),
])
def test_old_example_names_point_to_new(old, new):
    with pytest.raises(AttributeError, match=new):
        getattr(examples, old)


@pytest.mark.parametrize("module,name", [
    ("pyphi.models", "Concept"), ("pyphi.convert", "b2l"), ("pyphi.convert", "sbs2sbn"),
])
def test_removed_aliases_name_the_replacement(module, name):
    import importlib
    with pytest.raises(AttributeError, match="replaced|renamed"):
        getattr(importlib.import_module(module), name)


def test_analytical_relations_to_pandas_summarizes():
    a = pyphi.analyze(examples.iit4_2023_fig1a_substrate(), (0, 1, 1), subset=(0, 1))
    df = a.ces.relations.to_pandas()
    assert len(df) > 0
```

(Implementer: adapt `weight_axes` arguments to its real signature if needed; the assertions are the contract.)

- [ ] **Step 2–4:** Run (fail); implement; full fast suite; `just docs`; run the slow IIT 3.0 reproductions: `uv run pytest -m slow --slow -q -p no:cacheprovider -k "iit3 or 2014"`.
- [ ] **Step 5:** Migration guide rows for every removed or renamed name; fragments `changelog.d/results-api.change.md`, `changelog.d/example-names.change.md`; commit.

### Task 8: Analysis entry-point vocabulary and long-format sweeps (D4, D9)

**Files:**
- Modify: `pyphi/analysis.py` (`parallel` replaces `parallel_kwargs`; `progress`; `grains=False`), `pyphi/sweeps.py` (`compute` vocabulary incl. `None` and `"distinctions"`; drop `seed`; long-format df; `big_phi`/`sum_phi_d`; axis validation; numpy inputs; compact card; normalize rc-saved results on load), `pyphi/optimization.py` (`direction` → `goal`; validate state/subset/bounds before scipy, finding err-1), `pyphi/landscape.py`, `pyphi/estimate.py` (`phi_posterior`), `pyphi/cost.py`, `pyphi/substrate.py` (`complexes(parallel=, progress=)`), `pyphi/macro/search.py` (`complexes(*, bounds, parallel, prune, progress)`), `pyphi/actual.py` (`Transition` keyword-only tail), `pyphi/system.py` (`distinctions(*, congruent)`), campaign code that calls these, MCP server tools, docs (`sweep.md` incl. `skipped` tuple shape, `landscape.md`, `parallel.md`, `optimize` docs), `Analysis.to_pandas` columns.
- Test: `test/test_entry_point_vocabulary.py`; sweep round-trip with an rc3 fixture.

- [ ] **Step 1: Write the failing tests:**

```python
import inspect

import numpy as np
import pytest

import pyphi
from pyphi import examples

ENTRY_POINTS = [pyphi.analyze, pyphi.sweep, pyphi.optimize, pyphi.landscape_section,
                pyphi.perturb, pyphi.phi_posterior, pyphi.estimate_analysis]


def test_shared_arguments_are_keyword_only_and_named_alike():
    for f in ENTRY_POINTS:
        params = inspect.signature(f).parameters
        for name in ("subset", "subsets", "formalism", "formalisms", "compute",
                     "parallel", "progress", "seed"):
            if name in params:
                assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, (f, name)
        assert "parallel_kwargs" not in params
        assert "direction" not in params


def test_compute_vocabulary_is_shared():
    s = examples.basic_substrate()
    for compute in (None, "sia", "ces", "distinctions"):
        pyphi.analyze(s, (1, 0, 0), compute=compute)
        pyphi.sweep(s, states=[(1, 0, 0)], compute=compute, progress=False)
        pyphi.estimate_analysis(s, compute=compute)


def test_sweep_df_is_long_format_whatever_the_axis_sizes():
    s = examples.basic_substrate()
    one = pyphi.sweep(s, states=[(1, 0, 0)], progress=False).df
    two = pyphi.sweep(s, states=[(1, 0, 0), (0, 0, 1)], progress=False).df
    for df in (one, two):
        assert {"substrate", "formalism", "subset", "state"} <= set(df.columns)
        assert isinstance(df.index, __import__("pandas").RangeIndex)


def test_sweep_full_rows_report_big_phi():
    df = pyphi.sweep(examples.basic_substrate(), states=[(1, 1, 0)], compute=None,
                     progress=False).df
    assert {"phi", "big_phi", "sum_phi_d", "sum_phi_r"} <= set(df.columns)


def test_sweep_accepts_numpy_states_and_validates_axes():
    s = examples.basic_substrate()
    pyphi.sweep(s, states=np.array([[1, 0, 0], [0, 0, 1]]), progress=False)
    pyphi.sweep(s, states=tuple(np.array([1, 0, 0])), progress=False)
    pyphi.sweep(s, states=[(1, 0, 0)], subsets=(0, 1, 2), progress=False)
    with pytest.raises(ValueError):
        pyphi.sweep(s, states=[], progress=False)
    with pytest.raises(ValueError, match="all"):
        pyphi.sweep(s, states="All", progress=False)


def test_sweep_repr_is_compact():
    r = pyphi.sweep(examples.iit4_2023_fig1a_substrate(), states="all", progress=False)
    assert repr(r).count("\n") < 40


def test_analyze_grains_false_means_no_search():
    a = pyphi.analyze(examples.basic_substrate(), (1, 0, 0), grains=False)
    assert isinstance(a, pyphi.Analysis)
```

- [ ] **Step 2–4:** Run (fail); implement; update campaign code, MCP tools and their tests for the renamed arguments; run `test/campaign`, `test/mcp`, then the full fast suite; `just docs`.
- [ ] **Step 5:** Fragment `changelog.d/entry-point-vocabulary.change.md`, `changelog.d/sweep-long-format.change.md`; commit.

### Task 9: Intrinsic-unit competitor set honors τ_J (D8, finding theory-2)

**Files:**
- Modify: `pyphi/macro/search.py:876-915` (`_f_for_unit`), the "single verdict per decomposition" cache key, `pyphi/macro/criteria.py:8-10` docstring, `docs/theory/macro-units.md:97-101`, `docs/theory/computational-complexity.md:321-326`.
- Test: `test/macro/test_competitor_grain.py`.

- [ ] **Step 1: Write the failing test** (from the finding's reproduction):

```python
from pyphi import examples
from pyphi.macro import MacroUnit, SearchBounds, coarse_grain
from pyphi.macro.search import competing_systems


def test_competitors_have_update_grain_at_most_the_units():
    sub = examples.grid3_substrate()
    gamma = MacroUnit((0, 1, 2), update_grain=1, mapping=coarse_grain(3, {2, 3}))
    competitors = competing_systems(
        sub, gamma, ((0, 0, 0), (0, 0, 0)),
        SearchBounds(max_update_grain=2, max_constituents=3),
    )
    assert competitors
    assert all(u.micro_grain <= 1 for system in competitors for u in system)
```

(Implementer: match `competing_systems`' real return shape when iterating units.)

- [ ] **Step 2–4:** Run (fail); restrict members to `micro_grain ≤ unit.micro_grain`; key the verdict cache on (decomposition, τ_J); run `test/macro` and the slow macro reproductions (as in Task 5); confirm no pinned value moves under the default `max_update_grain=1`. Correct the three passages.
- [ ] **Step 5:** Fragment `changelog.d/intrinsic-unit-competitors.fix.md`; commit.

### Task 10: Error messages and first-run polish

**Findings:** err-2, err-4, err-5, err-6, err-7/mcp-10, err-8, err-10/mcp-15, res-8, port-7, drift-6, polish-1, drift-11 (docs note), port-11.

**Files:** `pyphi/substrate.py` and `pyphi/core/tpm/` (state-by-state checks: finite, [0,1], row sums, before the independence test; range message names unit, input state and the user's entry; `FactoredTPM.__array__` returning state-by-node ON probabilities for binary substrates and raising a `TypeError` naming `.to_pandas()`/`.to_joint()` for k-ary), `pyphi/analysis.py` and `pyphi/cost.py` (first argument type check; `System` accepted by using its substrate, state and node indices), `pyphi/validate.py` (node labels must be `str`), `pyphi/serialize/` (`load` wraps msgspec errors, names the path, detects 1.x jsonify files), `pyphi/formalism/iit4/` null SIA (records the system state; reason row "Why φ_s = 0: not strongly connected" on the Analysis card), `pyphi/substrate.py` (`alphabet=` accepts a tuple of sizes), `pyphi/_progress.py` (no `TqdmWarning` without ipywidgets: fall back to `tqdm.std`), the `ot` import (suppress POT's `SyntaxWarning` at import), `docs/migration/migration-2.0.md`.

**Test:** extend `test/test_first_run.py` with one test per finding, each asserting the message text or behavior named in the finding (for example: `pytest.raises(InvalidTPM, match="row")` for a state-by-state TPM `np.eye(4) * 0.9`; `pyphi.analyze(examples.basic_system(), (1,0,0))` either works or raises `TypeError` mentioning `substrate`; `Substrate(tpm, node_labels=range(2))` raises `ValueError` mentioning `str`; loading a 1.x jsonify dict raises an error mentioning "1.x"; `analyze(basic_substrate(), (1,0,0), subset=(0,1))` card contains "not strongly connected" and not "None"; `np.asarray(basic_substrate().tpm).shape == (8, 3)`; `import pyphi` in a subprocess with `tqdm.auto` patched to simulate no ipywidgets emits no `TqdmWarning`).

- [ ] **Steps:** write the tests (fail) → implement → full fast suite → fragment `changelog.d/error-messages.change.md` → commit.

### Task 11: MCP server

**Findings:** mcp-2, mcp-3, mcp-4, mcp-5, mcp-6, mcp-9, mcp-13, mcp-15, and content mcp-7, mcp-8, mcp-11, mcp-12, mcp-14.

**Files:** `pyphi/mcp/server.py`, `pyphi/mcp/install.py`, `pyphi/mcp/agents.py`, `pyphi/cost.py` (early refusal: once the memoized partition count for a smaller fully connected system exceeds the limit, stop), `pyphi/campaign/scope.py` (reject unknown scope keys; accept `max_purview_order_by_mechanism_order`), `pyphi/mcp/content/*.md`, `pyphi/mcp/skills/`, `docs/howto/mcp-server.md`, `docs/howto/ai-assistants.md`, `pyphi/mcp/prompts.py`.

**Test:** `test/mcp/` additions: the guard refuses a fully connected 10-unit substrate in under 10 seconds; `estimate_cost(..., subset=[...])` prices the subset; an unknown scope key raises; `inspect(ref)` with no path is under 5 KB and lists paths; `plot` on a SIA-only result with `kind="repertoires"` raises a `ValueError` naming `compute`; `analyze(formalism="iit3", confirm_large=True)` reports `"IIT_3_0"`; `detail="bogus"` raises; `install --print --no-iit-expert` prints no plugin line and lists the CLAUDE.md change; uninstall after `--no-iit-expert` prints no plugin hint.

- [ ] **Steps:** write the tests (fail) → implement → `test/mcp` and the full fast suite → content fixes (tool list, prompts, default labels `n0, n1, …`, k-ary TPM shape, "plot writes figures under the system temp directory", φₛ not Φₛ, a real example pair of values) → fragment → commit.

### Task 12: Documentation corrections (Sonnet)

**Findings:** api-20, port-2, port-3, port-5, port-6, port-10, port-12, port-13/drift-1, drift-2, drift-3, drift-4, drift-8, drift-9, drift-10, res-7, res-10, polish-2, and a final sweep for the names renamed in Tasks 1–8.

**Files:** `docs/migration/migration-2.0.md` and `pyphi/mcp/content/migration.md` (rows for `compute.complexes` → `irreducible_sias`, `compute.condensed` → `complexes`, `compute.all_complexes` → `all_sias`; the real 1.x example names; result-attribute renames; option values BI/TRI/ALL and `PICK_SMALLEST_PURVIEW`; the actual-causation partition option; a Macro section stating that 1.x macro values do not reproduce; 1.x IIT 3.0 values agree to about 1e-5), `docs/howto/earlier-versions.md` (a session-wide recipe that works for every preset; link to `blob/main/pyphi_config_3.0.yml`), `pyphi_config_3.0.yml` header, `docs/theory/system-integration.md`, `docs/reference/glossary.md`, `pyphi/substrate.py` `complexes` docstring and `pyphi/mcp/content/interpreting-iit-results.md` (recursive definition of a complex), `docs/howto/tie-breaking.md` table, `docs/examples/IIT_4.0_demo.ipynb` setup cell (no `catch_warnings`; preset before `shortcircuit_sia`; outputs refreshed with `just notebook-outputs`), the six Colab notebooks (install cell, as in the demo), `docs/howto/cache.md`, `docs/howto/build-substrate.md`, `docs/tutorials/worked-example.md` (+0.7 each way), `pyphi/optimization.py` docstring ("the gate caveat").

**Test:** `just docs` succeeds; `test/docs/test_demo_notebook.py` passes; `git grep -n "from_substrate\|purview_state\|parallel_kwargs\|fig1a_substrate\b\|pyphi\.sweep\.\|Concept\b" -- docs README.md pyphi/mcp` returns only intended mentions (migration rows).

- [ ] **Steps:** edits → `just docs` → demo notebook outputs → fragment `changelog.d/docs-corrections.doc.md` → commit.

### Task 13: Final gates and whole-branch review

- [ ] **Step 1:** Full fast suite (bare `uv run pytest`), slow suite split in two runs (`-k "test_marshall_2023_fig3_condensation"` and its complement) with the machine kept awake, `just docs`, `uv build` + clean-venv import with `env -i PATH=/usr/bin:/bin`, perf pins.
- [ ] **Step 2:** One Opus review subagent over `git diff main...HEAD` against the spec: every decision implemented, no aliases left for rc-only names, migration rows complete, fragments present.
- [ ] **Step 3:** Fix what the review confirms; re-run the affected gates.
- [ ] **Step 4:** Report to the user with the gate results; merge to `main` (fast-forward) only after the user agrees. Push and tag only with consent.
