# Migration guide (2.0) — design

The fourth sub-project of the documentation overhaul
(`docs/superpowers/specs/2026-07-07-documentation-overhaul-design.md` §6).
Writes `docs/migration/migration-2.0.md`, a single page documenting every API
change a pre-2.0 user hits when moving to PyPhi 2.0 (IIT 4.0). Discharges
ship-criterion #5.

## Audience

Two kinds of reader, served by one topic-organized page rather than two
era-specific walkthroughs (which would duplicate the many shared changes):

- **1.x** — users of released PyPhi 1.x (the PyPhi-paper / IIT 3.0 era): the
  large jump, including `Network`/`Subsystem` and the `compute.*` entry points.
- **4.0-branch** — users of the IIT 4.0 feature branch: the incremental changes,
  mainly the recent renames, the configuration format, and serialization.

Each topic is tagged with the audience it affects — **[1.x]**, **[4.0-branch]**,
or **[both]** — so readers scan for what applies to them.

## Structure

`docs/migration/migration-2.0.md`, in this order:

1. **Orientation.** 2.0 is a breaking release implementing IIT 4.0, with IIT 3.0
   retained through configuration. There are no deprecation shims: changes are
   hard breaks, so code written against pre-2.0 PyPhi must be updated. A short
   "who this is for" tying the audience tags to the two eras.
2. **Rename table.** The mechanical renames at a glance, as a scannable
   old → new table, each row audience-tagged: `Network → Substrate`,
   `Subsystem → System`, `cause_tpm → cause_marginal`,
   `effect_tpm → effect_marginal` (and the `proper_*` variants), and the notable
   module moves. Renames that need no more than the table stop here.
3. **Per-topic prose.** The changes that need more than a rename, each a short
   subsection with a before/after snippet:
   - **Entry point** — `compute.big_phi` / `compute.ces` (and related
     `compute.*`) give way to `pyphi.analyze(substrate, state)` and its result
     objects (`Analysis`, `CauseEffectStructure`) [1.x].
   - **Formalism selection** — choosing IIT 3.0 vs 4.0 (2023) vs 4.0 (2026) via
     `analyze(..., formalism=…)` or the layered config, replacing the old
     single `IIT_VERSION` toggle [both]. Cross-reference the theory section's
     formalism-versions page.
   - **Configuration** — the layered nested YAML format (top-level `formalism` /
     `infrastructure` / `numerics`); legacy flat YAML is rejected with a rename
     map; runtime access (`pyphi.config…`) [both].
   - **Serialization** — the jsonify → msgspec break: `pyphi.jsonify` is gone,
     `pyphi.save` / `pyphi.load` (and `.save()` / `.load()` on results) replace
     it, and old JSON is handled by the migration tool [both].
   - **Changed defaults** — any default that changed in a way that would
     silently alter results, so a reader knows to check it [both].

## Examples

Static before/after snippets. A migration guide's "before" is the old API, which
does not exist under 2.0 and cannot execute; the page is therefore a **static
reference, not an executed notebook** — the one place in the documentation
overhaul where non-executable is correct. Every "after" (2.0) snippet is
verified against the real 2.0 API while writing, so the guide never documents a
call that does not exist.

## Accuracy

Written from the actual change history, not from memory:

- The ROADMAP's records of landed changes (the renames, config layering,
  serialization break, changed defaults), `changelog.d/`, and git history are
  the source for *what changed*.
- Every 2.0 API call shown is verified against the current code (imported and
  run, or read in the source) before it appears in the guide.
- A rename or behavior change a real pre-2.0 user would hit must appear; the
  guide does not need to catalog internal-only changes.

## Success criteria

- `docs/migration/migration-2.0.md` exists, is wired into the Migration toctree,
  and builds clean under `-W`.
- The rename table covers the public renames; each per-topic subsection is
  present with a correct before/after.
- Every "after" snippet is a real, verified 2.0 call.
- Ship-criterion #5 (`migration-2.0.md` ships) is discharged; the ROADMAP row is
  updated.
