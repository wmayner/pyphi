# Formalism Pinning / Decoupling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple the test suite from the ambient default formalism so every φ-asserting test declares which formalism it validates, making a future default flip a small change with a known blast radius — with **zero φ-value changes** in this work.

**Architecture:** Unify all formalism pins onto the complete nested-preset representation (killing the partial-pin class at the root), then add narrowest-scope formalism pins to the currently-unpinned numeric tests. The "test" for this hygiene work is a two-ambient behavior check: the suite stays green under the current default (2023) with no value edits, and becomes green under a forced-2026 ambient (except the deliberately default-dependent Group-3 tests).

**Tech Stack:** pytest, pyphi config layering (`pyphi.config.override`, `pyphi/conf/presets.py`).

## Global Constraints

- **Zero φ-value changes.** No golden is regenerated; no expected value is edited. If any value moves, stop and investigate — it means a pin is not value-neutral.
- **Do not touch unrelated working-tree changes.** The tree has unrelated untracked files (experiments/, notebooks). Stage only files this plan modifies.
- **Never bypass pre-commit hooks.** No `--no-verify`.
- **Python 3.13+**, `uv run` for all commands.
- **Formalism pins use the nested-preset form**, never a hand-listed subset of `iit.*` fields. Complete `IITConfig` as a unit.
- **Verification uses `uv run pytest` with no path argument at least once** (Task 7) so the `pyphi/` doctest sweep runs.

## The two-ambient verifier

Several tasks verify under a forced-2026 ambient. Save this reusable harness to the scratchpad (not committed):

`/private/tmp/claude-501/-Users-will-projects-pyphi/1dd4bccc-73b8-46b4-9aab-8abb94fabc99/scratchpad/flip_verify.py`

```python
"""Run given pytest args under a forced IIT-4.0-2026 ambient default.

Usage: uv run python flip_verify.py <pytest args...>
Applies presets.iit4_2026 globally (tests that pin their own formalism nest
and restore); only genuinely-unpinned tests are affected.
"""
import sys
import pyphi
from pyphi.conf import presets

pyphi.config.override(**presets.iit4_2026).__enter__()
import pytest

sys.exit(pytest.main([*sys.argv[1:], "-q", "-o", "addopts=", "-p", "no:cacheprovider"]))
```

- **Run A (current default, 2023):** `uv run pytest <files> -q`
- **Run B (forced 2026 ambient):** `uv run python <scratchpad>/flip_verify.py <files>`

---

### Task 1: Unify formalism pins onto the nested-preset form (fixes Group 1)

**Files:**
- Modify: `test/golden/zoo.py:140-160` (`IIT_4_2023_CONFIG`, `IIT_4_2026_CONFIG`)
- Modify: `test/golden/zoo.py` inline `config_overrides={...}` dicts (audit; ~lines 256, 349, 425, 450)
- Modify: `test/conftest.py:49-56` (`IIT_4_CONFIG`)

**Interfaces:**
- Produces: `IIT_4_2023_CONFIG` / `IIT_4_2026_CONFIG` (zoo) and `IIT_4_CONFIG` (conftest) as complete, preset-sourced formalism pins reused by later tasks.

- [ ] **Step 1: Confirm the current failure (Run B on the goldens)**

Run: `uv run python <scratchpad>/flip_verify.py test/integration/test_golden_regression.py -m "not slow"`
Expected: FAIL — `test_golden_regression[basic_iit4_2023]`, `[basic_subset_iit4_2023]`, `[logistic3_k8_iit4_2023]`, `[xor_iit4_2023]` fail (the leaky pin lets the 2026 cap through).

- [ ] **Step 2: Rewrite the zoo constants to nested-preset form**

In `test/golden/zoo.py`, replace `IIT_4_2023_CONFIG` (currently the dotted dict at ~line 141) and `IIT_4_2026_CONFIG` (~line 156) with:

```python
# IIT 4.0 (2023) — Albantakis et al. 2023, GID measure, no ii(s) cap.
# Sourced from the canonical preset so every formalism field comes as a unit
# (a hand-listed subset silently inherits the rest from the ambient default —
# the partial-pin trap). ``progress_bars`` / ``parallel`` are test-runtime
# knobs, not formalism fields.
IIT_4_2023_CONFIG = {**presets.iit4_2023, "progress_bars": False, "parallel": False}

# IIT 4.0 (2026) — Mayner, Marshall, Tononi 2026. ii(s) = min(i_diff, i_spec)
# cap on system phi (INTRINSIC_INFORMATION); mechanism phi stays GID.
IIT_4_2026_CONFIG = {**presets.iit4_2026, "progress_bars": False, "parallel": False}
```

(`presets` is already imported at `zoo.py:45`.)

- [ ] **Step 3: Audit and convert inline formalism dicts in zoo.py**

Search: `grep -n "config_overrides={" test/golden/zoo.py`
For each inline dict that pins a 4.0 formalism by hand-listed `iit.*` keys, replace it with the matching constant (`IIT_4_2023_CONFIG` / `IIT_4_2026_CONFIG`) or nested-preset form. Leave dicts that are already nested-preset or that intentionally set a non-formalism field. Do not change IIT 3.0 dicts (already `IIT_3_CONFIG`).

- [ ] **Step 4: Source conftest.IIT_4_CONFIG from the preset**

In `test/conftest.py`, replace the `IIT_4_CONFIG` definition (lines 49-56) with:

```python
# IIT 4.0 (2023) configuration, complete and preset-sourced. Use this to pin a
# test to the 2023/GID formalism explicitly, independent of the ambient default.
IIT_4_CONFIG = config.override(**presets.iit4_2023)
```

- [ ] **Step 5: Run A — goldens green, no regen**

Run: `uv run pytest test/integration/test_golden_regression.py test/conf -q`
Expected: PASS, no fixtures regenerated (no `--regenerate-golden` passed).

- [ ] **Step 6: Run B — goldens now insulated**

Run: `uv run python <scratchpad>/flip_verify.py test/integration/test_golden_regression.py -m "not slow"`
Expected: PASS — the four `*_iit4_2023` goldens no longer leak the cap.

- [ ] **Step 7: Commit**

```bash
git add test/golden/zoo.py test/conftest.py
git commit -m "Complete formalism pins via nested-preset form in golden configs

The iit4_2023 golden config listed iit.version and the mechanism measure
but not system_phi_measure, so it inherited the system measure from the
ambient default. Source the 4.0 configs from the canonical presets (whole
IITConfig as a unit) like the existing IIT_3_CONFIG, closing the leak by
construction. Value-neutral under the current default."
```

---

### Task 2: Pin the function-level Group-2 files to 2023

**Files (computation happens in test bodies; formalism-specific tests self-override with `with` blocks):**
- Modify: `test/formalism/test_big_phi.py`
- Modify: `test/formalism/test_iit4.py`
- Modify: `test/models/test_complex_model.py`
- Modify: `test/models/test_explanation.py`
- Modify: `test/display/test_display.py`
- Modify: `test/test_connectivity_validation.py`
- Modify: `test/mcp/test_server.py` (extend the existing `_quiet` autouse fixture)

**Interfaces:**
- Consumes: `IIT_4_CONFIG` from `test.conftest` (Task 1).

- [ ] **Step 1: Confirm current failures (Run B)**

Run: `uv run python <scratchpad>/flip_verify.py test/formalism/test_big_phi.py test/formalism/test_iit4.py test/models/test_complex_model.py test/models/test_explanation.py test/display/test_display.py test/test_connectivity_validation.py test/mcp/test_server.py`
Expected: FAIL — the Group-2 tests in these files (per the measured list) fail under the 2026 ambient.

- [ ] **Step 2: Add a module-scoped 2023 pin to each file without an existing autouse quiet fixture**

For `test_big_phi.py`, `test_iit4.py`, `test_complex_model.py`, `test_explanation.py`, `test_display.py`, `test_connectivity_validation.py`: import the pin and add a module-level autouse fixture near the top of the module (after imports, before the first test):

```python
from test.conftest import IIT_4_CONFIG


@pytest.fixture(autouse=True)
def _pin_iit4_2023():
    """Pin the 2023/GID formalism for this module, so φ assertions do not
    depend on the ambient default. Tests that need another formalism override
    it locally with a ``with`` block, which nests inside this pin."""
    with IIT_4_CONFIG:
        yield
```

If a file already imports part of this (e.g. `test_big_phi.py` imports `IIT_3_CONFIG`), add the `IIT_4_CONFIG` import alongside; do not duplicate `import pytest`.

- [ ] **Step 3: Extend the existing `_quiet` fixture in `test/mcp/test_server.py`**

`mcp/test_server.py` already has an autouse `_quiet` (line ~31). Replace its body so it also pins 2023:

```python
@pytest.fixture(autouse=True)
def _quiet():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        yield
```

Add `from test.conftest import IIT_4_CONFIG` to the imports.

- [ ] **Step 4: Guard against autouse races**

For each file, confirm no in-file test class defines its own autouse formalism fixture (which would race with the new module-autouse). Search each file: `grep -nE "autouse|IIT_3_CONFIG|presets.iit3|II_CONFIG" <file>`. Formalism-specific tests must self-override via `with`/decorator (nests correctly), not via a competing class-autouse. If any file does have a competing class-autouse, move its pin to per-class scope instead (see Task 3 pattern). Expected here: all seven use `with`-block self-overrides, so the module pin is safe.

- [ ] **Step 5: Run A — unchanged**

Run: `uv run pytest test/formalism/test_big_phi.py test/formalism/test_iit4.py test/models/test_complex_model.py test/models/test_explanation.py test/display/test_display.py test/test_connectivity_validation.py test/mcp/test_server.py -q`
Expected: PASS, identical counts to before (no value changes).

- [ ] **Step 6: Run B — now green**

Run: `uv run python <scratchpad>/flip_verify.py test/formalism/test_big_phi.py test/formalism/test_iit4.py test/models/test_complex_model.py test/models/test_explanation.py test/display/test_display.py test/test_connectivity_validation.py test/mcp/test_server.py`
Expected: PASS — these files are now insulated from the ambient default.

- [ ] **Step 7: Commit**

```bash
git add test/formalism/test_big_phi.py test/formalism/test_iit4.py test/models/test_complex_model.py test/models/test_explanation.py test/display/test_display.py test/test_connectivity_validation.py test/mcp/test_server.py
git commit -m "Pin 2023 formalism in function-level IIT 4.0 numeric tests

These tests asserted phi values while riding the ambient default formalism.
Pin them explicitly to 2023/GID so their assertions no longer depend on
what the library default is. No value changes."
```

---

### Task 3: Pin the mixed-formalism files per class / per test

**Files (a single file exercises more than one formalism, some via class-autouse fixtures):**
- Modify: `test/formalism/test_complexes.py` (add per-class 2023 pin to the four IIT 4.0 classes)
- Modify: `test/formalism/test_iit4_sia_components.py` (per-class 2023 pin to `TestPhiValues`, `TestPartitionTypes`; per-test pin to `TestEq23IntrinsicInformationCap::test_gid_distance_unaffected`)

**Interfaces:**
- Consumes: `IIT_4_CONFIG` from `test.conftest` (Task 1).

- [ ] **Step 1: Confirm current failures (Run B)**

Run: `uv run python <scratchpad>/flip_verify.py test/formalism/test_complexes.py test/formalism/test_iit4_sia_components.py`
Expected: FAIL — `TestComplexWrapperIIT40`, `TestMaximalComplexWrapperIIT40`, `TestSubstrateMethodsIIT40`; `TestPhiValues::*`, `TestPartitionTypes::test_sia_standard_example_partition_type`, `TestEq23IntrinsicInformationCap::test_gid_distance_unaffected`.

- [ ] **Step 2: Add a per-class 2023 pin to the IIT 4.0 classes in `test_complexes.py`**

`test_complexes.py` already imports `IIT_3_CONFIG` and has `TestComplexesIIT30` with its own autouse `IIT_3_CONFIG` fixture (line ~40). Do **not** add a module-wide pin (it would race with that). Instead, add an autouse fixture to each IIT 4.0 class — `TestCauseEffectStructureIIT40` (line ~194), `TestSubstrateMethodsIIT40` (~237), `TestComplexWrapperIIT40` (~383), `TestMaximalComplexWrapperIIT40` (~411):

```python
    @pytest.fixture(autouse=True)
    def _pin_iit4_2023(self):
        with IIT_4_CONFIG:
            yield
```

Add `from test.conftest import IIT_4_CONFIG` to the imports.

- [ ] **Step 3: Pin the 2023 classes in `test_iit4_sia_components.py`**

Add the same per-class autouse fixture (Step 2 body) to `TestPhiValues` (line ~49) and `TestPartitionTypes` (line ~226). Add `from test.conftest import IIT_4_CONFIG` if not present.

- [ ] **Step 4: Pin the single GID test inside the cap class**

`TestEq23IntrinsicInformationCap` (line ~384) is internally mixed: most tests apply `II_CONFIG` (INTRINSIC_INFORMATION) explicitly, but `test_gid_distance_unaffected` (line ~498) asserts GID behavior and rides the ambient. Do **not** add a class-wide pin here (it would fight the II tests). Wrap just that one test's body:

```python
    def test_gid_distance_unaffected(self, s):
        with IIT_4_CONFIG:
            ...  # existing body, indented one level
```

(The II tests that apply `II_CONFIG` on top of an explicit 2023 pin still produce the cap — `system_phi_measure=INTRINSIC_INFORMATION` binds regardless of version label — so if a class-wide pin were ever preferred it would also work; the per-test wrap is the minimal, clearest change.)

- [ ] **Step 5: Run A — unchanged**

Run: `uv run pytest test/formalism/test_complexes.py test/formalism/test_iit4_sia_components.py -q`
Expected: PASS, identical counts.

- [ ] **Step 6: Run B — now green**

Run: `uv run python <scratchpad>/flip_verify.py test/formalism/test_complexes.py test/formalism/test_iit4_sia_components.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add test/formalism/test_complexes.py test/formalism/test_iit4_sia_components.py
git commit -m "Pin formalism per class/test in mixed-formalism files

test_complexes.py and test_iit4_sia_components.py each exercise more than
one formalism. Pin the 2023/GID classes and the one GID-specific cap test
at the narrowest scope, leaving the 2026-cap tests' own overrides intact.
No value changes."
```

---

### Task 4: Pin formalism inside module-scoped φ fixtures

**Files (φ is computed at module-fixture setup, which a function-scoped autouse pin does not cover):**
- Modify: `test/test_estimate.py` (`grid3_phi_posterior` module fixture + a function-autouse pin)
- Modify: `test/formalism/test_selection_margins.py` (`basic_sia` / `xor_sia` module fixtures + a function-autouse pin)

**Interfaces:**
- Consumes: `IIT_4_CONFIG` from `test.conftest` (Task 1).

- [ ] **Step 1: Confirm current failures (Run B)**

Run: `uv run python <scratchpad>/flip_verify.py test/test_estimate.py test/formalism/test_selection_margins.py`
Expected: FAIL — `test_epsilon_boundary_sensitivity`, `test_observational_twin_nonidentifiability`, `test_screen_engages_and_matches_unscreened`; `test_partition_margin_matches_brute_force`.

- [ ] **Step 2: Pin inside the module φ fixtures in `test_estimate.py`**

`grid3_phi_posterior` (line ~205) computes `phi_posterior(...)` inside `config.override(progress_bars=False)`. Extend that context to also pin 2023:

```python
@pytest.fixture(scope="module")
def grid3_phi_posterior(grid3_posterior):
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        return phi_posterior(grid3_posterior, (0, 0, 0), n_samples=40, seed=99)
```

Add `from test.conftest import IIT_4_CONFIG` to the imports.

- [ ] **Step 3: Also pin the test-body path in `test_estimate.py`**

Extend the existing `_quiet` autouse fixture (line ~15) so test bodies that compute under the ambient are covered too:

```python
@pytest.fixture(autouse=True)
def _quiet():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        yield
```

- [ ] **Step 4: Pin the module SIA fixtures in `test_selection_margins.py`**

`basic_sia` and `xor_sia` (lines ~26, ~32) compute `.sia()` inside `config.override(progress_bars=False)`. Extend both to pin 2023:

```python
@pytest.fixture(scope="module")
def basic_sia():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        return examples.basic_system().sia()


@pytest.fixture(scope="module")
def xor_sia():
    with IIT_4_CONFIG, pyphi.config.override(progress_bars=False):
        return examples.xor_system().sia()
```

Extend the `_quiet` autouse (line ~20) the same way as Step 3, and add `from test.conftest import IIT_4_CONFIG`. Note `test_selection_margins.py` line ~174 has a test that toggles `iit.version` between 2026 and 2023 via `with` blocks; those nest inside the module pin correctly — leave them.

- [ ] **Step 5: Run A — unchanged**

Run: `uv run pytest test/test_estimate.py test/formalism/test_selection_margins.py -q`
Expected: PASS, identical counts.

- [ ] **Step 6: Run B — now green**

Run: `uv run python <scratchpad>/flip_verify.py test/test_estimate.py test/formalism/test_selection_margins.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add test/test_estimate.py test/formalism/test_selection_margins.py
git commit -m "Pin formalism inside module-scoped phi fixtures

grid3_phi_posterior, basic_sia and xor_sia compute phi at module-fixture
setup, which a function-scoped autouse pin does not wrap. Pin 2023 inside
those fixtures (and the function-level _quiet path). No value changes."
```

---

### Task 5: Add the canonical default-formalism assertion (Group 3)

**Files:**
- Create/Modify: a single canonical assertion in `test/conf/test_config_layers.py`
- Reference (leave as-is, document): `test/conf/test_config_layers.py::TestGlobalConfigFacade::test_layered_reads_work`, `::test_legacy_uppercase_read_still_works`, `test/formalism/test_formalism_measure_threading.py::test_2023_omitted_metric_uses_default`

**Interfaces:**
- Produces: `test_default_formalism_is_iit4_2023` — the single findable "the library default is X" assertion, and the one place the future flip edits.

- [ ] **Step 1: Write the canonical assertion**

Add to `test/conf/test_config_layers.py` (module level, no formalism pin — it must read the true ambient default):

```python
def test_default_formalism_is_iit4_2023():
    """The library default formalism. This is the single canonical assertion
    of the shipping default; flipping the default is a deliberate edit here
    (plus the default-dependent facade tests it points to). Do NOT pin a
    formalism in this test — it must observe the real default."""
    from pyphi.conf import config

    iit = config.formalism.iit
    assert iit.version == "IIT_4_0_2023"
    assert iit.system_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
    assert iit.mechanism_phi_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
    assert iit.specification_measure == "GENERALIZED_INTRINSIC_DIFFERENCE"
```

- [ ] **Step 2: Run it (default) — passes**

Run: `uv run pytest test/conf/test_config_layers.py::test_default_formalism_is_iit4_2023 -q`
Expected: PASS.

- [ ] **Step 3: Confirm it is the intended red under a flip (Run B)**

Run: `uv run python <scratchpad>/flip_verify.py test/conf/test_config_layers.py::test_default_formalism_is_iit4_2023`
Expected: FAIL (asserts 2023, ambient is 2026). This is the deliberate signal, not a regression.

- [ ] **Step 4: Commit**

```bash
git add test/conf/test_config_layers.py
git commit -m "Add canonical default-formalism assertion

A single, findable assertion of the shipping default formalism, unpinned
so it observes the real default. Flipping the default is a deliberate edit
here."
```

---

### Task 6: Document the convention and the flip recipe

**Files:**
- Modify: `CLAUDE.md` (Testing Strategy section)
- Modify: `test/conftest.py` (the pattern-doc header near the `IIT_*_CONFIG` definitions)

- [ ] **Step 1: Add the convention note to `CLAUDE.md`**

In the Testing Strategy section, add a short subsection:

```markdown
#### Formalism pinning (tests that assert φ values)

A φ value is only meaningful relative to a formalism. Any test that asserts a
φ value must **pin its formalism explicitly** — never rely on the ambient
default. Pin with the complete preset-sourced context managers
(`IIT_3_CONFIG`, `IIT_4_CONFIG` in `test/conftest.py`, sourced from
`pyphi.conf.presets`), not a hand-listed subset of `iit.*` fields: setting
`iit.version` alone leaves the measures on the ambient default — the
partial-pin trap that silently recomputes under a different formalism when the
default changes. Tests that compute φ at module-fixture setup must pin inside
the fixture (a function-scoped autouse pin does not wrap module-fixture setup).

Exactly one test — `test_default_formalism_is_iit4_2023` — asserts the shipping
default; it is intentionally unpinned. To flip the default formalism: change
the default in `pyphi/conf/formalism.py`, update that assertion and the
default-dependent facade tests it names, and regenerate only the `docs/`
tutorial examples that demonstrate default behavior (CI doctests in `pyphi/`
compute no cap-sensitive φ).
```

- [ ] **Step 2: Update the conftest pattern header**

In `test/conftest.py`, near the `IIT_*_CONFIG` block (lines ~30-56), extend the usage comment to state: pins are preset-sourced and complete; a φ-asserting test must apply one; never hand-list `iit.version` alone.

- [ ] **Step 3: Verify docs build is unaffected**

Run: `uv run pytest --doctest-modules pyphi/ -q -o addopts=`
Expected: PASS (no source doctest computes a cap-sensitive φ; sanity check that nothing regressed).

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md test/conftest.py
git commit -m "Document formalism-pinning convention and the default-flip recipe"
```

---

### Task 7: Full two-ambient acceptance

**Files:** none (verification only).

- [ ] **Step 1: Run A — full suite green under the current default**

Run: `uv run pytest`
Expected: PASS across `pyphi/` (doctests) and `test/`, with **no golden regeneration and no expected-value edits anywhere in the diff**. (Slow lane may be run separately with `--slow`; if time-boxed, at minimum run `uv run pytest` without a path so the doctest sweep is included.)

- [ ] **Step 2: Run B — full fast lane green under forced 2026, except Group 3**

Run: `uv run python <scratchpad>/flip_verify.py test/ pyphi/ --doctest-modules -m "not slow"`
Expected: PASS except the deliberately default-dependent tests:
`test/conf/test_config_layers.py::TestGlobalConfigFacade::test_layered_reads_work`,
`::test_legacy_uppercase_read_still_works`,
`test/conf/test_config_layers.py::test_default_formalism_is_iit4_2023`,
`test/formalism/test_formalism_measure_threading.py::test_2023_omitted_metric_uses_default`.
These four are the entire, documented flip blast radius. Any *other* failure means a pin was missed — return to the relevant task.

- [ ] **Step 3: Confirm zero value changes in the diff**

Run: `git diff develop --stat` (or the base branch) and scan: only test files, `CLAUDE.md`, `zoo.py`, `conftest.py` changed; no `test/data/golden/**` changes.
Expected: no golden data files in the diff.

- [ ] **Step 4: Update ROADMAP if this work is tracked there**

If the ROADMAP has a row for this (or the Wave-7 ii-gate that motivated it), note the decoupling landed and that a default flip is now a bounded follow-up.

- [ ] **Step 5: Final commit (if Step 4 changed ROADMAP)**

```bash
git add ROADMAP.md
git commit -m "Note formalism-pinning decoupling landed; default flip now bounded"
```

---

## Self-Review

**Spec coverage:**
- Root fix (unify on preset form) → Task 1. ✓
- Group 1 (leaky goldens) → Task 1. ✓
- Group 2 (unpinned numeric, ~39) → Tasks 2 (function-level), 3 (mixed-formalism), 4 (module φ fixtures). ✓
- Group 3 (canonical default assertion) → Task 5. ✓
- CLAUDE.md convention + flip recipe → Task 6. ✓
- Two-run acceptance → Task 7 (and per-task Run A/Run B). ✓
- Non-goals (actual flip, 2026 coverage expansion, ii-gate) → not in any task. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"; each pin shows the exact fixture code and insertion target. Line numbers are approximate ("~") because unrelated edits may shift them; the anchoring symbol names are exact.

**Type/name consistency:** `IIT_4_CONFIG` (conftest, Task 1) is the single pin reused by Tasks 2–4. `flip_verify.py` path is constant. `test_default_formalism_is_iit4_2023` (Task 5) is referenced identically in Tasks 6–7.

**Known soft spots for the executor:**
- Task 2 Step 4 (autouse-race guard) and Task 3's per-class placement depend on each file's actual override style; the Run A/Run B pair per task is the backstop — if Run A changes a value, a pin was placed too broadly.
- If any file's module pin makes a currently-passing test change value under Run A, narrow the pin to per-class/per-test (Task 3 pattern).
