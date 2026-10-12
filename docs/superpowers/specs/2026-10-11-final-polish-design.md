# PyPhi 2.0.0 final polish — design

Status: DRAFT for review (2026-10-11). Not to be committed until approved.

## Purpose

A six-angle review of `main` after 2.0.0rc3 (public API, 1.x porting, a
researcher using the docs, the MCP server, docs drift, error messages and
theory prose) reported 86 findings; independent verifiers confirmed 75 and
partly confirmed 11. This document records the design decisions needed to
act on them, and the scope of the work. The verified findings, with their
reproductions, are in the review's `results.json` (scratchpad); finding IDs
below (`api-2`, `res-1`, …) refer to it.

After 2.0.0 ships, every public name, signature and result shape is a
compatibility promise. This pass therefore settles the API questions now,
in one release, rather than through deprecations later.

## Compatibility policy for this pass

- Names that exist only in the 2.0 release candidates are renamed outright,
  with no aliases (they were never in a final release).
- Names from 1.x that are removed or renamed raise an `AttributeError` that
  names the replacement, as `pyphi.Network` already does, and get a row in
  the migration guide.
- Files saved by rc1–rc3 must still load. Where a serialized field is
  renamed, the decoder accepts the old field name.
- No computed φ value changes, except where a finding is a correctness bug
  in the computation itself (decision D8). Every value change gets a test
  that states the old and new value and why.

## Decisions

### D1. The candidate system is called `subset` everywhere (findings api-5, api-2, port-1)

Every public entry point that takes "which units form the candidate system"
names that argument `subset`:

| Today | After |
| --- | --- |
| `analyze(..., subset=)` | unchanged |
| `Substrate.sia(state, indices=None, **kwargs)` | `Substrate.sia(state, *, subset=None, ...)` |
| `Substrate.ces / phi_structure(state, indices=None, **kwargs)` | same as `sia` |
| `System(substrate, state, node_indices=None, ...)` | `System(substrate, state, *, subset=None, ...)` |
| `System.from_substrate(substrate, state, nodes=None, ...)` | removed; it only called `System(...)` |
| `actual.events(..., nodes, ...)`, `actual.true_events(..., indices=None, ...)` | `subset` |

The resolved units stay readable as `system.node_indices`, matching
`substrate.node_indices`; `subset` is the input spelling only.

`subset` is keyword-only on the methods, as it already is on `analyze`, so
a state can never be read as a subset.

### D2. No silently ignored keyword arguments (findings api-2, port-1)

`Substrate.sia/ces/phi_structure/complexes/maximal_complex/irreducible_sias/all_sias`
and `System.sia/ces/distinctions/mic/mie/phi` replace `**kwargs` with
explicit keyword parameters, or validate `**kwargs` against the formalism's
accepted names and raise `TypeError` naming the unknown argument. A misspelt
option, `subset=` on a method that does not take it, or `formalism=` on a
method that does not take it must raise.

`formalism=` is accepted on `Substrate.sia/ces/phi_structure/complexes/
maximal_complex` and `System.sia/ces`, with the same meaning and
resolution (`presets.canonical`) as on `analyze`.

### D3. Shadowed modules: rename the implementation modules (findings api-3, api-4)

`pyphi.analyze`, `pyphi.sweep` and `pyphi.optimize` are both functions and
module names. Python lets the function win as an attribute, so
`pyphi.sweep.SweepResult` fails, `import pyphi.sweep as m` returns the
function, and Sphinx's autosummary documents the function and loses the
module page (which is why `Analysis`, `SweepResult` and
`OptimizationResult` are missing from the API reference).

The standard fix is to give the implementation module a different name from
the function it exports, and keep the function at the top level. Numpy,
pandas and requests all do this (`requests.get` lives in `requests.api`).
Modules are named for the thing they contain:

| Function (top level, unchanged) | Module today | Module after |
| --- | --- | --- |
| `pyphi.analyze`, `pyphi.Analysis` | `pyphi/analyze.py` | `pyphi/analysis.py` |
| `pyphi.sweep`, `pyphi.SweepResult` | `pyphi/sweep.py` | `pyphi/sweeps.py` |
| `pyphi.optimize`, `pyphi.OptimizationResult` | `pyphi/optimize.py` | `pyphi/optimization.py` |

This matches `pyphi.landscape` (module) / `pyphi.landscape_section`
(function). Saved files are unaffected: serialization identifies types by
msgspec tags, not module paths. The `_SHADOWED_BY_FUNCTION` special case in
`pyphi/__init__.py` is deleted.

Considered and rejected: private modules (`pyphi/_analyze.py`). They hide
the classes' home from the API reference and from `help()`, and nothing
else in PyPhi uses that pattern.

### D4. One vocabulary for the analysis entry points (findings api-6, api-7, api-18, api-12, api-19)

The functions that run analyses (`analyze`, `sweep`, `optimize`,
`landscape_section`, `perturb`, `phi_posterior`, `estimate_analysis`,
`Substrate.complexes`, `macro.complexes`) use these arguments, with these
meanings, wherever they apply:

| Argument | Meaning | Position |
| --- | --- | --- |
| `substrate` / `builder` / `posterior` | what is analyzed | positional |
| `state` | the current state | positional |
| `subset` | the candidate system (D1); `subsets=` on `sweep`, which takes an axis | keyword-only |
| `formalism` | version of IIT, resolved by `presets.canonical`; `formalisms=` on `sweep` | keyword-only |
| `compute` | what to compute: `None` (the full analysis: SIA and Φ-structure), `"sia"`, `"ces"`, `"distinctions"`, or a callable | keyword-only |
| `parallel` | `bool \| None`: run this call in parallel; `None` defers to `config.infrastructure.parallel` | keyword-only |
| `progress` | `bool \| None`: show progress bars; `None` defers to config | keyword-only |
| `seed` | seed for an isolated random number generator; only on functions that draw random numbers | keyword-only |

Consequences:

- `compute` accepts the same five values everywhere it appears, except
  that `estimate_analysis` rejects a callable, whose cost it cannot count.
  `sweep` gains `"distinctions"` and `None`. Default: `None` on `analyze` and
  `estimate_analysis` (unchanged); **`"sia"` on `sweep`**, kept because a
  sweep multiplies cost by the number of cells, and stated in the
  docstring and how-to. (Open for review: default `sweep` to `None` too.)
- `parallel_kwargs: dict` leaves the public signatures of `analyze`,
  `Substrate.complexes` and `macro.complexes`; they take `parallel: bool |
  None` like `sweep`. Fine-grained tuning stays in `config.infrastructure`
  (the per-level parallel options), used through `config.override`.
- `seed` is removed from `sweep`, where it only labelled provenance; the
  provenance writers (`save_json(seed=...)`) already record a seed.
- `optimize(direction="maximize")` becomes `goal="maximize" | "minimize"`,
  because `direction` means cause/effect everywhere else in PyPhi.
- `analyze(grains=False)` is accepted and means no grain search.
- Optional configuration arguments after the core ones are keyword-only:
  `System(substrate, state, *, subset, partition, external_indices,
  background_conditioning, background_state)`,
  `Transition(substrate, before_state, after_state, cause_indices,
  effect_indices, *, partition, noise_background)`,
  `macro.complexes(substrate, micro_history, *, bounds, parallel, prune,
  progress)`, `System.distinctions(*, congruent)`.

### D5. Specification, not "intrinsic information", per direction (findings res-1, res-2, res-3, theory-1, drift-7, errors api-1)

Mayner, Marshall & Tononi (2026, Sec. 2.3) rename the per-direction
selectivity × informativeness quantity "intrinsic specification" and keep
"intrinsic information" ii(s) for min(differentiation, specification). The
API follows the paper:

- `StateSpecification.intrinsic_information` → `intrinsic_specification`;
  `runner_up_intrinsic_information` → `runner_up_intrinsic_specification`.
  The same rename on the RIA/MICE objects and the system-level state
  specification wherever the attribute holds the per-direction quantity.
  The serialized field is renamed; the decoder still reads the old name.
- `sia.intrinsic_information` stays: it is ii(s), and it equals the term in
  φ_s = min(φ_c, φ_e, ii(s)).
- Card labels, `explain()` labels (`"Specified-cause-state margin (ii)"`)
  and docs follow the same split.
- Under the 2023 version, where the per-direction quantity was called
  intrinsic information, PyPhi still uses the 2026 name: one name per
  quantity. The Analysis card omits the `ii(s)` row for results computed
  under the 2023 version, which does not use ii(s) in φ_s.
- `MICE.purview_state` (the *current* state of the purview units) is
  renamed `current_state`. The specified state stays
  `mice.specified_state.state`.
- **Bug fix (res-1):** purview labels in distinction cards, the distinction
  table, `Distinctions` rows and `plot_ces` endpoints are written in the
  specified state, as the docs and the paper's figures do, not in the
  current state. Relation cards already do this.
- Theory pages, glossary and `phi-structure.md` are corrected: the
  specified state maximizes intrinsic specification (2026 Eq. 12); ii(s)
  appears only as the min term.

### D6. Explicit `__all__` (findings api-11, api-16)

`pyphi.__all__` becomes an explicit list matching the curated API page
(docs/reference/api.md): the public functions and classes plus the public
submodules. Internal modules (`deferred`, `registry`, `protocols`, `utils`,
`node`, `combinatorics`, `serializable`, `data_structures`, `constants`,
`log`, `cache`, and helpers such as `TransitionSystem`) stay importable but
are not in `__all__`, so `from pyphi import *` no longer rebinds names such
as `log` and `cache`. `pyphi.models.__all__` drops underscore names and
registries. `pyphi.actual` gets an `__all__` listing its tutorial surface.

### D7. Remaining API decisions (findings api-15, api-17, port-3, port-8, port-9, drift-12, api-13, res-9)

- **Aliases removed:** `models.Concept`, `System.null_concept`, the twelve
  `convert` short names (`b2l`, `l2b`, `l2s`, `b2s`, `s2l`, `s2b`, `b2l_sbs`,
  `l2b_sbs`, `to_md`, `to_2d`, `sbn2sbs`, `sbs2sbn`). Each raises an
  `AttributeError` naming the replacement and gets a migration-guide row.
  `System.apply_cut(partition)` becomes `System.apply_partition(partition)`
  and the card row "Cut" becomes "Partition".
- **Example names:** the IIT 3.0 (2014) examples get the paper prefix used
  for the 2023 ones: `fig1a_substrate` → `iit3_2014_fig1a_substrate`, and
  so on for all fifteen `fig*` names. The 1.x names (`fig1a()`,
  `fig4()`, …, `actual_causation()`, `prevention()`) raise an
  `AttributeError` naming the 2.0 name.
- **`analysis.ces` has one type:** always `CauseEffectStructure`. Under
  IIT 3.0 its `distinctions` are the resolved distinctions and its
  `relations` are empty. `len()`, iteration and `.phis` behave the same
  under every version.
- **Collections return tuples:** `Distinctions.phis`, `.mechanisms` and the
  purview properties return tuples instead of single-use generators.
- **`Relation.purview`** is always a `frozenset`.
- **Equality:** `LandscapeSection.__eq__` compares DataFrames with
  `.equals`, as `SweepResult` does; `LandscapeSection` and `Perturbation`
  are explicitly unhashable (`__hash__ = None`).
- **`AnalyticalRelations.to_pandas()`** returns the closed-form summary
  (degree spectrum and Σφ_r) instead of always raising.

### D8. Correctness bugs (findings api-1, mcp-1/err-3, res-4, theory-2, drift-3)

- **Grain search** drops candidates with φ_s ≤ 0 (by `numerics.is_positive`)
  before the exclusion cascade, in both the plain and the pruned path, so
  `analyze(..., grains=True).complexes` matches `Substrate.complexes` on the
  micro grain.
- **Config writes are validated on every path.** `config[...] = ...`,
  attribute writes and layer replacement run the same constraint check as
  `config.override` and YAML loading. Setting `iit.version` alone raises
  `ConfigurationError`, whichever way it is written, and the "ambiguous"
  message recommends a preset, not `config["iit.version"]`.
- **`explain()`** computes the runner-up gap on values without the
  intrinsic-information requirement (as `partition_margin` already does),
  and partitions that differ only in direction or severed connections print
  differently. `read-result.md` and `system-integration.md` say what the
  MIP is actually chosen on.
- **Intrinsic-unit criteria (Marshall et al. 2026, Eq. 17):** the competitor
  set for a unit with update grain τ_J contains only systems whose units
  have update grain ≤ τ_J. The "one verdict per decomposition" shortcut is
  keyed on (decomposition, τ_J). Behavior with the default
  `max_update_grain=1` is unchanged. Theory pages and the `criteria.py`
  docstring are corrected.
- **Complex definition in the docs** states the recursive rule (accept the
  φ_s-maximal candidate, exclude everything overlapping it, repeat on the
  rest) everywhere: `system-integration.md`, the glossary, the
  `Substrate.complexes` docstring, and the MCP interpretation guide.

### D9. Sweep results are long-format (findings api-8, api-10, res-6, api-9, err-9, api-14, api-20)

`SweepResult.df` always has one column per axis (`substrate`, `formalism`,
`subset`, `state`) and a default `RangeIndex`, whatever the number of values
on each axis. `compute="ces"` and `compute=None` rows include `big_phi`
and `sum_phi_d` with the column names `Analysis.to_pandas()` uses.
`Analysis.to_pandas()` returns a one-row DataFrame with the same columns.
Results saved by rc1–rc3 are normalized to this shape on load.

Axis arguments are validated before any computation: numpy integer states
and 2-D arrays are accepted, a single subset tuple is accepted like a single
state, empty axes and unknown strings raise `ValueError`. `SweepResult`,
`LandscapeSection` and `Perturbation` get compact cards instead of reprs
that print every SIA.

## Remaining findings (no design decision needed)

- **Error messages and first-run polish:** err-1, err-2, err-4, err-5, err-6,
  err-7/mcp-10 (a "Why φ_s = 0" row for systems that are not strongly
  connected, and the state recorded on null SIAs), err-8, err-10/mcp-15,
  err-11 and port-4 (frozen layers say how to set an option; near-match
  suggestions for typos), res-8, port-7 (`FactoredTPM.__array__`), res-5
  (`ComplexesResult` card), drift-6 (no `TqdmWarning` on import in
  Jupyter without ipywidgets), polish-1 (POT `SyntaxWarning`), drift-11
  (migration guide notes that `from pyphi import Network` gives a plain
  ImportError), port-11.
- **MCP server:** mcp-2 (refuse large runs without enumerating), mcp-3
  (reject unknown scope keys; pass `max_purview_order_by_mechanism_order`),
  mcp-4 (`inspect` with no path returns the summary and the inspectable
  paths), mcp-5, mcp-6 (`estimate_cost` takes `subset`), mcp-9, mcp-13,
  mcp-15, and content fixes mcp-7, mcp-8, mcp-11, mcp-12, mcp-14.
- **Docs errors:** api-3, api-20, port-2, port-3, port-5 (a Macro section in
  the migration guide; 1.x macro values do not reproduce), port-6, port-10,
  port-12, port-13/drift-1, drift-2, drift-4, drift-5, drift-8, drift-9
  (Colab install cells), drift-10, res-7, res-10, polish-2, and every
  migration-guide row the renames above require.

## Non-goals

- New features.
- Renaming `System.node_indices`, `Substrate.node_indices`, or the
  formalism version strings.
- Repository hygiene (tracked PDFs, internal planning files).
- Release mechanics (Zenodo, the version switcher, release-candidate
  wording); those follow this pass.

## Execution strategy and cost

Work happens in one worktree (`.claude/worktrees/final-polish`, branch
`claude/2-0-final-polish`), merged to `main` when every gate passes. Push
and tag only with the user's consent.

The session that coordinates the work keeps its own context small: each
task in the plan is carried out by one implementer subagent that starts
fresh, reads the plan, the spec and the code it touches, implements with
tests, runs the task's checks, commits, and returns a short report. The
coordinator reads the report and the diff summary, runs the fast suite
between tasks, and stops to ask the user only if a task hits a decision the
spec does not settle.

Cost controls:

- No per-task review subagent. One whole-branch review (one Opus agent)
  runs at the end, together with the slow suite, the docs build and the
  packaging check.
- Implementers that change computation or public API run on Opus.
  Documentation-only tasks run on Sonnet.
- Tasks run one at a time. Several tasks touch the same files (the API
  renames reach most of the package), so running them in parallel would
  produce merge conflicts.
- The coordinator relies on automatic context compaction; the plan file,
  the spec and the git log carry everything needed to resume.
