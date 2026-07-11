# Formalism pinning: decoupling the test suite from the ambient default

**Date:** 2026-07-11
**Status:** Design, approved for planning
**Scope:** Test-suite hygiene. Zero φ-value changes. No user-facing behavior change.

## Problem

The IIT formalism a computation uses is read from the ambient configuration
(`config.formalism.iit.version` and the three measure fields). The library
default is currently `IIT_4_0_2023` with `GENERALIZED_INTRINSIC_DIFFERENCE`
(GID) on all measures (`pyphi/conf/formalism.py:82-85`).

A large part of the test suite asserts φ values without pinning a formalism —
those tests silently depend on the ambient default being 2023/GID. Changing the
default (for example to the 2026 intrinsic-information-cap formalism) therefore
recomputes those tests under a different formalism and breaks them, with no way
to tell from the test whether the math regressed or the default merely moved.

A φ value is only meaningful relative to a formalism. A test that asserts a
number must declare which formalism it validates. Where it does not, the
assertion's meaning is ambiguous and silently follows the default.

### Measured blast radius

Running the fast lane with the ambient default forced to the 2026 preset
(`presets.iit4_2026`), in-process, so that tests which pin their own formalism
nest and restore correctly and only genuinely-unpinned tests break:

**45 failed, 3490 passed, 7 skipped** (fast lane, `-m "not slow"`).

The 45 failures fall into three groups.

**Group 1 — leaky "pinned" fixtures (the latent bug).** Fixtures that *look*
pinned but set only a subset of the formalism fields, inheriting the rest from
the ambient default:

- `test/integration/test_golden_regression.py::test_golden_regression[*_iit4_2023]`
  (4 parametrizations), via `test/golden/zoo.py::IIT_4_2023_CONFIG`.

`IIT_4_2023_CONFIG` (`zoo.py:141`) sets `iit.version`, `mechanism_phi_measure`,
and `system_partition_scheme` — but **not `system_phi_measure`**. It has been
correct only because the ambient default's system measure is GID. Flip the
default's system measure to `INTRINSIC_INFORMATION` and every "iit4_2023" golden
silently computes under the 2026 cap while claiming to be the 2023 formalism.
`test/formalism/test_iit4.py::test[*]` and
`test/formalism/test_iit4_sia_components.py::TestEq23IntrinsicInformationCap::test_gid_distance_unaffected`
are likely the same partial-pin shape; the plan confirms each.

**Group 2 — unpinned IIT-4.0 numeric tests (the bulk, ~39).** Compute φ and
assert values while riding ambient = 2023/GID. The table below is the raw
measured failure count per file. Whether a given failure is a partial pin
(Group 1) or no pin at all (Group 2) is a diagnostic detail confirmed during
planning — a few rows here (notably `test_iit4.py` and the
`TestEq23IntrinsicInformationCap` case inside `test_iit4_sia_components.py`) may
resolve to Group 1. The remedy is identical either way: a complete,
preset-sourced pin.

| File | Count |
|------|-------|
| `test/formalism/test_big_phi.py` | 9 |
| `test/formalism/test_iit4_sia_components.py` | 7 |
| `test/models/test_complex_model.py` | 5 |
| `test/formalism/test_complexes.py` | 4 |
| `test/test_estimate.py` | 3 |
| `test/mcp/test_server.py` | 3 |
| `test/formalism/test_iit4.py` | 3 |
| `test/models/test_explanation.py` | 2 |
| `test/display/test_display.py` | 1 |
| `test/formalism/test_selection_margins.py` | 1 |
| `test/test_connectivity_validation.py` | 1 |

All IIT 3.0 tests pass, because they already pin — they have to, since the
default is 4.0. It is the 4.0 tests that assumed the default was theirs.

**Group 3 — config/default plumbing (2).** Tests that *should* reflect whatever
the default is:

- `test/conf/test_config_layers.py::TestGlobalConfigFacade::test_layered_reads_work`
- `test/formalism/test_formalism_measure_threading.py::test_2023_omitted_metric_uses_default`

(Exact per-test group assignment is finalized during planning; the counts above
are the measured failure list, not an estimate.)

## Goal and non-goals

**Goal.** Decouple the suite from the ambient default so that flipping the
library default later is a small change with a known, empty blast radius. Every
φ-asserting test declares its formalism; a future default change touches exactly
one canonical assertion and nothing else silently.

**This work makes zero φ-value changes.** Group-2 tests already assert correct
2023 values; pinning them to 2023 freezes what they already assert. No goldens
are regenerated and no expected values are edited. 2023 remains a supported
formalism, so retaining this coverage is correct.

**Non-goals (explicitly out of scope):**

- **The actual default flip.** Changing `conf/formalism.py` to default to 2026
  is a separate follow-up, made trivial by this work (change the default, flip
  the one canonical assertion, done).
- **Expanding 2026 flagship coverage.** The `*_iit4_2026` golden tier already
  covers the 2026 formalism; assessing whether it is *sufficient* is a separate
  question, not part of decoupling.
- **The ii-gated grain scheduler** (the Wave 7 item this discussion started
  from). Independent.

## The root fix

The defect is not "some tests forgot to pin." It is that **pinning is done by
hand-rolled partial dicts that can drift out of sync with the formalism they
name.** Two such dicts exist — `conftest.IIT_4_CONFIG` and
`zoo.IIT_4_2023_CONFIG` — each independently listing formalism fields, and one
of them omitted a field.

The fix is to source every formalism pin from the canonical presets in
`pyphi/conf/presets.py`, which define each formalism as a whole `IITConfig`
namespace. Overriding with a preset replaces the entire `iit` sub-namespace as a
unit, so no individual field can be forgotten:

```python
with pyphi.config.override(**presets.iit4_2023):
    ...  # version=2023, sys=GID, mech=GID, spec=GID, partition=DIRECTED_SET_PARTITION
```

Verified from a 2026 ambient (the leaky scenario): the override resets all three
measures to GID and the partition scheme to `DIRECTED_SET_PARTITION`, and
restores the 2026 ambient on exit. `IITConfig()` defaults
(`system_phi_measure=GID`, `system_partition_scheme=DIRECTED_SET_PARTITION`)
match exactly what `IIT_4_2023_CONFIG` currently hand-sets, so replacing the
hand-rolled dict with a preset source is provably value-neutral.

Infrastructure knobs used only for test runtime (`progress_bars`, `parallel`)
are layered on top of the preset separately; they are not formalism fields.

This removes the entire partial-pin class at the root rather than patching
individual symptoms.

## Work items

1. **Consolidate the pins onto presets.**
   Redefine `conftest.IIT_4_CONFIG`, `zoo.IIT_4_2023_CONFIG`, and
   `zoo.IIT_4_2026_CONFIG` to derive from `presets.iit4_2023` / `presets.iit4_2026`
   (with the infrastructure layer applied separately). Confirm value-neutrality
   by checking `IITConfig` defaults against the fields the dicts currently set.
   This fixes Group 1.

2. **Pin Group 2 to 2023.**
   Add autouse fixtures applying the complete 2023 pin, at the narrowest scope
   that is uniformly one formalism: module-scoped where an entire file is 2023;
   per-class where a file mixes formalisms. `test_iit4_sia_components.py` is the
   known mixed case — `TestPhiValues` (2023) alongside
   `TestEq23IntrinsicInformationCap` (the 2026 cap) — so it must not take a
   blanket module pin. Reuse the existing `with IIT_4_CONFIG: yield` fixture
   pattern already documented in `conftest.py`. No marker framework.

3. **One canonical default assertion.**
   Consolidate Group 3 into a single test that asserts
   `config.formalism.iit.version` and the measure fields equal the library
   default. This is the one intended failure on a future flip — the single,
   loud, deliberate signal that the default moved.

4. **Document the convention and the flip recipe.**
   - A short note in the project `CLAUDE.md` (Testing Strategy) and the
     `conftest.py` pattern header: φ-asserting tests pin their formalism via
     `presets.*`; never set `iit.version` alone, because that leaves the
     measures on the ambient default — the partial-pin trap that silently
     recomputes under a different formalism when the default changes.
   - A three-step "how to flip the default formalism" recipe: change the default
     in `conf/formalism.py`; flip the one canonical default assertion; regenerate
     only the documentation examples (`docs/` tutorials) that legitimately
     demonstrate default behavior. CI doctests in `pyphi/` are unaffected (no
     source doctest computes a cap-sensitive φ; the one φ-valued doctest is
     `# doctest: +SKIP`).

## Acceptance criteria

Two full-suite runs prove the deliverable:

- **Run A — default unchanged (2023).** The suite is green with **no golden
  regeneration and no expected-value edits**. Proves the work is zero-change.
- **Run B — default forced to 2026 in-process.** The suite is **still green**,
  except the single canonical default assertion (Group 3), which is expected to
  reflect the ambient. Proves the suite is decoupled from the ambient default —
  the actual flip-safety test.

The measurement harness for Run B already exists (applies `presets.iit4_2026` as
the ambient default globally, then runs the fast lane). The gap between today's
Run B (45 failures) and a green Run B is the whole deliverable.

Verification uses `uv run pytest` with no path argument at least once, so the
doctest sweep over `pyphi/` is exercised (bare-path invocations skip it).

## Risks and edge cases

- **Mixed-formalism files.** A blanket module pin on a file that also tests the
  2026 cap would break the cap tests. Pin at class/function scope in those files.
  `test_iit4_sia_components.py` is the identified case; the plan scans for others.
- **Value-neutrality must be verified, not assumed.** Item 1 changes how configs
  are constructed; Run A (no expected-value edits) is the guard. If any value
  moves, the preset defaults diverge from a hand-set field and that divergence is
  investigated before proceeding — it would itself be a latent inconsistency.
- **Presets carry infrastructure defaults.** Preset override replaces the whole
  `iit` namespace; `progress_bars` / `parallel` live outside it and are layered
  separately so test-runtime behavior (no progress bars, sequential) is preserved.
