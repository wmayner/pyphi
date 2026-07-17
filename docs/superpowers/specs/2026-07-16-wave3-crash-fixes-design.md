# Wave 3: Crashes on Documented Usage — Design

**Date:** 2026-07-16
**Source:** Whole-library review (`REVIEW-2026-07-13.md`), Wave 3 of the
recommended fix waves. All findings re-verified live at `654a846a` before
this design was written.

## Goal

Fix every Wave 3 finding: crashes reached through documented, intended API
usage. Four independent clusters, one branch, per-cluster TDD commits.

## Scope

**In scope (9 findings, 4 clusters):**

1. `sia(directions=[single direction])` AttributeError under the default
   config (`shortcircuit_sia=True`).
2. `landscape_section`/`perturb` and `optimize()` AttributeErrors under
   `formalism="IIT_3_0"`.
3. Visualize/MCP crashes: `plot_system` k-ary KeyError; `plot_tpm` binary
   2ⁿ assumption (crash on k-ary, silent mislabeling of non-square
   arrays); MCP `_state_by_state` binary-only reshape; MCP
   `build_substrate` documented `alphabet: list[int]` always crashing;
   `project_ces` IndexError on an empty CES; `project_ces` bare
   AssertionError on IIT 3.0 CESes.
4. `settle()` false NonConvergenceError when settling in exactly
   `max_steps` steps.

**Out of scope:**

- `build_triggered_tpm` k-ary handling — already fixed on main
  (`03e18202` added `_validate_binary_substrate` at both matching entry
  points); the full k-ary generalization is an open ROADMAP follow-up.
- Real k-ary support in the matching/triggered-TPM path (feature work).
- The bare `assert self.specified_state is not None` in
  `pyphi/models/ria.py` — an internal IIT 4.0 invariant, unreachable from
  `plot_ces` after the cluster 3 fix.
- Rendering an empty figure for an empty CES. **Rejected alternative:**
  it would need per-view empty-input handling (the lattice, hypergraph,
  and scatter views each fail separately on empty input), and a silently
  empty plot can hide reducibility from the user. A clear error at the
  single choke point was chosen instead.

## Cluster 1 — `sia(directions=...)` shortcircuit guard

**Site:** `pyphi/formalism/iit4/__init__.py`, `_has_no_cause_or_effect`.

`system_intrinsic_information` builds a `SystemStateSpecification` with
`None` for any direction not requested, but `_has_no_cause_or_effect`
iterates `Direction.both()` unconditionally and dereferences
`.intrinsic_information` on the absent direction's `None` spec.

**Fix:** skip a direction whose specification is `None`. A direction that
was not requested cannot shortcircuit the analysis; the rest of the
pipeline already guards `None` specs. No behavior change for the
two-direction path.

## Cluster 2 — cross-formalism guards in `landscape` and `optimize`

**Sites:** `pyphi/landscape.py` `_eval_point`; `pyphi/optimize.py`
`_eval_one` and `_objective_value`.

Both drivers read IIT 4.0-only SIA attributes (`system_state`,
`state_margins`, `signed_phi`, `signed_normalized_phi`,
`partition_margin`, `effectively_tied`) unconditionally, so any
documented non-4.0 formalism preset crashes them. The sibling driver
`pyphi/sweep.py` `_row_sia` already establishes the intended
cross-formalism contract: `getattr(..., default)` guards, with
`None`/`NaN` carried in the IIT 4.0-only columns.

**Fix:**

- `_eval_point` and `_eval_one` adopt the `_row_sia` guard pattern.
  Rows produced under a non-4.0 formalism carry `None`/`NaN` in the
  IIT 4.0-only columns; `phi` and `partition` are populated for every
  formalism. The selection-identity/regime logic in `landscape` is
  unaffected (`None` states are hashable and comparable).
- `_objective_value` raises a clear `ValueError` naming the objective
  when the SIA lacks the requested attribute (e.g.
  `objective="signed_normalized_phi"` under `IIT_3_0`), instead of an
  AttributeError from deep inside a scipy run. The default
  `objective="phi"` exists on every formalism's SIA. Callable objectives
  are unaffected.

## Cluster 3 — visualize/MCP

Six fixes, one commit each or grouped where files coincide.

### 3a. `plot_system` k-ary node colors

**Site:** `pyphi/visualize/connectivity.py` (`NODE_COLORS`,
`_system_graph`).

The four-entry `(in_system, state)` color dict crashes on any state ≥ 2.

**Fix:** a color helper replaces the raw dict lookup. Binary states keep
the exact current colors (lightgrey/darkgrey out-of-system,
lightblue/darkblue in-system). For a unit with alphabet size k > 2, the
color interpolates within the same hue family (grey out-of-system, blue
in-system) by state intensity `state / (k - 1)`, so state 0 is the light
end and state k−1 the dark end. Per-unit alphabet sizes come from
`system.substrate.tpm.alphabet_sizes`.

### 3b. `plot_tpm` state labels

**Site:** `pyphi/visualize/connectivity.py`, `plot_tpm`.

Tick labels derive from `all_states_str(int(np.log2(shape)))`, hard-coding
2ⁿ binary axes: a 3×3 k-ary state-by-state TPM crashes with a
tick/label-count mismatch, and a non-square state-by-node array is
silently labeled with wrong bit strings.

**Fix:** `plot_tpm` gains an optional `states=` parameter (a sequence of
state label strings; an axis is labeled with it when the axis length
equals `len(states)`). When
absent, bit-string labels are used only when the matrix is square with a
power-of-two size — a genuine binary state-by-state TPM — and plain
integer state indices otherwise. A square 2ᵏ-sized k-ary TPM remains
ambiguous from the bare array alone; callers that know the state space
(the MCP path, 3c) pass `states=` explicitly.

### 3c. MCP `_state_by_state` k-ary generalization

**Site:** `pyphi/mcp/server.py`, `_state_by_state` and the
`plot(kind="tpm")` branch.

The current implementation slices `[..., 1]` as a binary ON-probability
and reshapes to 2ⁿ rows.

**Fix:** generalize to explicit-alphabet form. The joint array has axes
`(*alphabet_sizes, node, state)`; the state-by-state entry for
(current s, next t) is the product over nodes i of
`joint[s..., i, t_i]`. States enumerate in little-endian mixed-radix
order via `utils.all_states`, which already accepts per-node alphabet
sizes. The result is an S×S matrix with S = ∏ alphabet sizes; the binary
case reproduces the current output exactly. `plot(kind="tpm")` passes the
substrate's state labels to `plot_tpm(states=...)` so k-ary axes are
labeled correctly.

### 3d. MCP `build_substrate` alphabet translation

**Site:** `pyphi/mcp/server.py`, `build_substrate`.

The tool's documented `alphabet: list[int]` is passed straight to
`Substrate(alphabet=...)`, which accepts only a single int — the
documented form has never worked.

**Fix:** translate the list to a per-node state space:
`state_space=tuple(tuple(range(k)) for k in alphabet)` (verified accepted
by `Substrate`). The tool's schema and docstring stay as they are.

### 3e. `project_ces` empty-CES guard

**Site:** `pyphi/visualize/projection/__init__.py`, `project_ces`.

`distinctions[0].node_labels` raises a bare IndexError on a CES with zero
distinctions — exactly what a reducible system yields.

**Fix:** raise
`ValueError("cannot project an empty cause-effect structure (no
distinctions: the system is reducible)")` before any indexing. All
`plot_ces` views route through `project_ces`, so one guard covers them
all.

### 3f. `project_ces` IIT 3.0 support

**Site:** `pyphi/visualize/projection/__init__.py`, the purview-union
inclusion order in `project_ces`.

`d.purview_union` chains into `purview_units`, which requires a
specified state that IIT 3.0 RIAs never carry, so every `plot_ces` view
dies with a message-less AssertionError on an IIT 3.0 CES.

**Fix:** the projection uses only purview *indices*, so build the unions
directly: `frozenset(d.cause_purview) | frozenset(d.effect_purview)`.
Verified by live probe: with this change the lattice, hypergraph, and
scatter views all render an IIT 3.0 CES end-to-end.

## Cluster 4 — `settle()` off-by-one

**Site:** `pyphi/dynamics.py`, the `max_steps` guard in `settle`.

The guard `len(trajectory) > max_steps` fires after appending a state
that may itself be the fixed point, one iteration before the fixed point
would be confirmed. The docstring defines settling time as
`len(result) - 1`, so a trajectory settling in exactly `max_steps` steps
must return.

**Fix:** change the condition to `len(trajectory) - 1 > max_steps`. At
that point the best case (the just-appended state is the fixed point) has
settling time `len(trajectory) - 1`; raising only when that exceeds
`max_steps` permits exact-`max_steps` settles and still raises one step
beyond, at the cost of at most one extra step computation.

## Testing

TDD per finding: a failing crash-repro test first, then the fix.

- **Cluster 1:** `directions=[Direction.EFFECT]` and
  `directions=[Direction.CAUSE]` return SIAs under the default config;
  a control asserts the two-direction result is unchanged.
- **Cluster 2:** `landscape_section(..., formalism="IIT_3_0")` returns a
  `LandscapeSection` whose IIT 4.0-only columns are `None`/`NaN` and
  whose `phi` column is populated; `_eval_one` under `IIT_3_0` with
  `objective="phi"` returns a finite objective (tested directly, not
  through a full differential-evolution run); `objective`s missing from
  the SIA raise the new `ValueError` with the objective named.
- **Cluster 3:** k-ary `plot_system`/`plot_tpm`/`_state_by_state`/
  `build_substrate` repros from the review become passing tests; the
  binary `_state_by_state` output is pinned unchanged; empty-CES and
  IIT 3.0 `plot_ces` get one test per view family. These are the first
  k-ary, empty-CES, and IIT 3.0 cases in the visualize suite, closing
  part of the review's coverage-gap finding.
- **Cluster 4:** the exact-`max_steps` settle returns with the documented
  settling time; `max_steps` one below the settling time still raises.
- All tests asserting φ values or running IIT 3.0 pin their formalism
  with the complete preset-sourced context managers from
  `test/conftest.py` (`IIT_3_CONFIG`), never a partial pin.
- Completion gate: full pathless `uv run pytest` in the worktree, and
  again in the main tree after merge.

## Process

- Branch `fix/wave3-documented-usage-crashes`, worktree
  `.claude/worktrees/wave3-crash-fixes`, merged to `main` with
  `--no-ff`.
- Changelog fragments per cluster in `changelog.d/` (`.fix.md`).
- The review file's status block and the project memory are updated in
  the main tree after merge.
