# Migration Guide (2.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write `docs/migration/migration-2.0.md`, the single topic-organized
migration guide defined in
`docs/superpowers/specs/2026-07-08-migration-guide-design.md`, discharging
ship-criterion #5.

**Architecture:** One MyST Markdown page under the Migration section, organized
as orientation → rename table → five per-topic subsections, each audience-tagged
(**[1.x]** / **[4.0-branch]** / **[both]**), with static before/after snippets.
The page is a static reference (not an executed notebook); every 2.0 "after"
snippet is verified against the current API before it appears.

**Tech Stack:** MyST Markdown, Sphinx `-W` build.

## Global Constraints

- **Static page, not executed.** No `{code-cell}` blocks and no jupytext front
  matter; use plain fenced ```python blocks. The "before" (pre-2.0 API) cannot
  run under 2.0.
- **Every 2.0 snippet verified.** Before writing an "after" snippet, confirm the
  call against the current code (`env -u VIRTUAL_ENV uv run python -c "…"`).
- **Sourced from the change history**, not memory: the ROADMAP's landed-change
  records, `changelog.d/`, and git.
- Builds under `uv run --all-extras --group docs sphinx-build -W --keep-going -b
  html docs docs/_build/html`.
- Commit trailer on every commit; never `--no-verify`; stage only named files.

## Verified facts (confirmed against the current code)

Renames (old → new), all confirmed present/absent:

- `pyphi.Network` → `pyphi.Substrate` (`Network` gone)
- `pyphi.Subsystem` → `pyphi.System` (`Subsystem` gone)
- `pyphi.compute` (module) → `pyphi.analyze` (`compute` gone)
- `subsystem.cause_tpm` / `effect_tpm` → `system.cause_marginal` /
  `effect_marginal` (+ `proper_cause_marginal` / `proper_effect_marginal`);
  the old names are gone
- `pyphi.jsonify` → `pyphi.serialize` (+ top-level `pyphi.save` / `pyphi.load`);
  `jsonify` gone

Entry point: `pyphi.analyze(substrate, state)` → `Analysis` with `.phi`, `.ces`,
`.sia`, `.system`.

Formalism: `pyphi.analyze(substrate, state, formalism="IIT_3_0" | "IIT_4_0_2023"
| "IIT_4_0_2026")`; the config field is `pyphi.config.formalism.iit.version`
(default `"IIT_4_0_2023"`). In 1.x this was a single `IIT_VERSION` toggle
defaulting to IIT 3.0 — so **the default formalism changed from IIT 3.0 to IIT
4.0**.

Config: 2.0 uses a **layered nested YAML** with top-level keys `formalism`
(sub-namespaces `iit` / `actual_causation`), `infrastructure`, `numerics`.
Legacy flat YAML (e.g. `PRECISION: 6`) is rejected on load with a rename map
(`pyphi/conf/_io.py`). Runtime access: `pyphi.config.numerics.precision`, etc.,
and a top-level write like `pyphi.config.precision = 6` routes to the right
layer.

Serialization: `pyphi.jsonify` (and per-class `to_json`/`from_json`) deleted;
`pyphi.serialize` provides `dumps`/`loads` (bytes) and `save`/`load` (path/file,
format inferred from extension), plus `.save()`/`.load()` on result types. It is
a **format break with no standalone converter** — results saved in the old
jsonify format cannot be loaded and must be recomputed (ROADMAP, 2026-06-25).
`save(obj, target, *, format=None)`, `load(target, *, format=None)`.

---

## Task 1: Write the migration guide

**Files:**
- Create: `docs/migration/migration-2.0.md`
- Modify: `docs/migration/index.md` (toctree)

- [ ] **Step 1: Re-verify every 2.0 "after" call.** Run the checks from
  "Verified facts" against the current code before writing, in case the API
  moved:

  ```bash
  env -u VIRTUAL_ENV uv run python -c "import pyphi, inspect; \
    print(hasattr(pyphi,'Substrate'), hasattr(pyphi,'System'), hasattr(pyphi,'analyze'), \
          hasattr(pyphi,'save'), hasattr(pyphi,'load'), not hasattr(pyphi,'Network'), \
          not hasattr(pyphi,'compute'), not hasattr(pyphi,'jsonify')); \
    print(inspect.signature(pyphi.analyze)); print(inspect.signature(pyphi.save))"
  ```
  Expected: all booleans `True`; the two signatures print. If any is `False`,
  correct that row's snippet before writing it.

- [ ] **Step 2: Write `docs/migration/migration-2.0.md`.** A static MyST page
  (no front matter, no `{code-cell}`), with these sections. Each "after" snippet
  must be one of the verified calls above.

  - **`# Migrating to PyPhi 2.0`** — orientation: 2.0 is a breaking release
    implementing IIT 4.0, with IIT 3.0 retained through configuration; there are
    no deprecation shims, so pre-2.0 code must be updated. Define the tags:
    **[1.x]** = released 1.x (IIT 3.0 era), **[4.0-branch]** = the IIT 4.0
    feature branch, **[both]**.

  - **`## Renames at a glance`** — a Markdown table, columns *Old* / *New* /
    *Affects*, rows:
    | Old | New | Affects |
    | --- | --- | --- |
    | `pyphi.Network` | `pyphi.Substrate` | [1.x] |
    | `pyphi.Subsystem` | `pyphi.System` | [1.x] |
    | `pyphi.compute.*` | `pyphi.analyze(...)` | [1.x] |
    | `subsystem.cause_tpm` | `system.cause_marginal` | [both] |
    | `subsystem.effect_tpm` | `system.effect_marginal` | [both] |
    | `pyphi.jsonify` | `pyphi.serialize` / `pyphi.save` / `pyphi.load` | [both] |
    | `pyphi.config.IIT_VERSION` | `pyphi.config.formalism.iit.version` | [both] |

    Add a one-line note that `cause_marginal`/`effect_marginal` are the causal
    marginals of IIT 4.0 (the old `cause_tpm` name was a misnomer — it was never
    a TPM), and reference the [theory section](../theory/substrate-and-system.md).

  - **`## Building and analyzing`** [1.x] — before/after: `pyphi.Network(tpm, cm)`
    + `pyphi.Subsystem(network, state, nodes)` + `pyphi.compute.big_phi(subsystem)`
    → `pyphi.Substrate(tpm, cm=cm)` + `pyphi.analyze(substrate, state)` and its
    `.phi` / `.ces` / `.sia`. Show the shape only; keep snippets short.

  - **`## Choosing a formalism`** [both] — the old single `IIT_VERSION` toggle
    (default IIT 3.0) becomes per-call `analyze(..., formalism="IIT_4_0_2023")`
    or the config field `pyphi.config.formalism.iit.version`; **the default is
    now IIT 4.0 (2023)**, so results differ from a 1.x default run unless
    `formalism="IIT_3_0"` is requested. Cross-reference
    [formalism versions](../theory/formalism-versions.md).

  - **`## Configuration`** [both] — flat YAML (`PRECISION: 6`) is rejected;
    the 2.0 file is nested under `formalism` / `infrastructure` / `numerics`.
    Before/after YAML snippet. Runtime: `pyphi.config.numerics.precision` (or a
    routed top-level write `pyphi.config.precision = 6`).

  - **`## Saving and loading results`** [both] — `pyphi.jsonify` is gone;
    use `pyphi.save(result, "result.json")` / `pyphi.load("result.json")` (or
    `result.save(...)` / `.load(...)`), format inferred from the extension
    (`.json`, `.mpk`, transparent `.gz`). State plainly that this is a format
    break with no converter: **results saved in the old jsonify format must be
    recomputed.**

  - **`## Changed defaults`** [both] — the default formalism is now IIT 4.0
    (2023), where 1.x defaulted to IIT 3.0; this changes computed values unless
    IIT 3.0 is explicitly requested. (List only defaults that silently change
    results; do not pad.)

- [ ] **Step 3: Wire into the Migration toctree.** Edit
  `docs/migration/index.md` so its toctree lists `migration-2.0` before
  `from-substrate-modeler`:

  ````markdown
  # Migration

  ```{toctree}
  :maxdepth: 1

  migration-2.0
  from-substrate-modeler
  ```
  ````

- [ ] **Step 4: Build.** `uv run --all-extras --group docs sphinx-build -W
  --keep-going -b html docs docs/_build/html` → exit 0. Confirm the page built:
  `ls docs/_build/html/migration/migration-2.0.html`.

- [ ] **Step 5: Commit.**

  ```bash
  git add docs/migration/migration-2.0.md docs/migration/index.md
  git commit -m "Add the 2.0 migration guide"
  ```

---

## Task 2: Roadmap and ship-criterion

- [ ] **Step 1: Update the ROADMAP.** In the Documentation-overhaul dashboard
  row and the overhaul section's migration bullet, mark the migration guide
  landed; note the sole remaining sub-project is tutorials/how-tos. In the ship
  criteria, mark criterion #5 (`migration-2.0.md` ships) satisfied.
- [ ] **Step 2: Commit** the ROADMAP change.

---

## Self-review checklist

- Spec structure (orientation, rename table, five per-topic subsections) →
  Task 1 Step 2 one-to-one. ✓
- Spec audience tags → the table's *Affects* column and per-section tags. ✓
- Spec "static, verified 2.0 snippets" → Global Constraints + Task 1 Steps 1–2. ✓
- Spec "no converter; recompute" → the Saving-and-loading section. ✓
- Spec success (ships, toctree, `-W`, ROADMAP) → Task 1 Steps 3–4, Task 2. ✓
- Placeholder scan: the before/after snippet *content* is specified by the
  verified old→new pairs; exact prose is written at execution, but every API
  call is pinned in "Verified facts". No TBDs.
