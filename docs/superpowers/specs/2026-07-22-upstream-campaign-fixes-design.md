# Upstream fixes from the color-qualia binding-control campaign — design

**Source request:** `color-qualia/docs/2026-07-22-pyphi-upstream-fixes-prompt.md`
(sibling repo). The color-qualia project, pinned to pyphi `caaa7cd4` (current
`main` at time of writing), ran the first real scoped-CES campaign on a 21-unit
substrate and hit four problems. This spec covers all four, split across three
implementation sessions.

**Global constraints** (from the source request, confirmed):

- Items 1, 2, and 4 must be behavior-preserving for computed results; item 3
  adds capability. One documented exception: item 4's running-mean rewrite is
  bit-identical only up to 128 stacked states (numpy switches to pairwise
  summation above that), ulp-level beyond; tolerance-based tie handling absorbs
  this.
- Every fix gets a test that fails before and passes after.
- On-disk format changes are safe (the campaign version guard makes `collect`
  refuse outputs from a different pyphi version) but must be called out
  explicitly in the commit message and in the session's final summary, so
  color-qualia knows to re-pin and re-prepare rather than resume existing
  campaign directories.
- Line numbers below are against `caaa7cd4`; re-locate with grep if `main` has
  moved.

**Session protocol** (applies to every session):

- Work in a worktree under `.claude/worktrees/`; merge to `main` when green.
- Run the full suite with `uv run pytest` (no path argument — the unpathed run
  is what sweeps doctests) at least once before merging, plus the slow lane
  (`uv run pytest -m slow --slow`) redirected to a file whose summary line is
  read, not inferred from exit codes.
- Update the ROADMAP rows for the items landed (Session A creates the rows for
  all four items; B and C update theirs) and add changelog fragments.
- Sessions merge in order A → B → C. B and A both touch
  `pyphi/core/repertoire_algebra.py` (different functions); A and C both touch
  `pyphi/campaign/__init__.py`. Sequential merges keep conflicts trivial.

---

## Session A — production blockers (items 1 and 2)

### Item 1: order cap for `Substrate.potential_purviews`

**Problem.** `potential_purviews` (`pyphi/substrate.py:412`, inner `compute()`
at `:437`) runs `utils.powerset(self._node_indices)` — 2^N candidates — through
`irreducible_purviews` per (direction, mechanism). At N=21 that is ~15 s per
call; campaign planning for a 1,372-mechanism scope would take ~11 hours, and
mechanism-payload shards pay it again at run time.

**Change.**

- Add `max_order: int | None = None` to `Substrate.potential_purviews`,
  implemented by passing `max_size=max_order` to `utils.powerset` (the
  parameter already exists).
- Include the cap in the `_PURVIEW_CACHE` key — extend the per-fingerprint key
  from `(direction, mechanism)` to include `max_order` — so bounded and
  unbounded results never alias. A truncated list served to an unbounded caller
  is silent wrongness; this is the reason the downstream monkeypatch kept its
  own cache.
- Thread the scope's purview cap through the scoped consumers, all of which
  already filter the enumeration through `scope.purview_axis(...).select`
  immediately afterward: `pyphi/cost.py:460` and `:558` (`mechanism_workloads`,
  `estimate_analysis`), `pyphi/campaign/shards.py:210` (`plan_ces_shards`),
  `pyphi/campaign/runner.py:133,138` (`_run_ces_shard`),
  `pyphi/campaign/__init__.py:1218` (collect's split-mechanism reassembly). An
  `AxisScope` exposes `max_order` directly; when it is `None`, or the axis uses
  `explicit`/`within` from which a maximum order can be derived, derive it;
  otherwise fall back to unbounded.
- Also fix `pyphi/core/repertoire_algebra.py:736` and
  `pyphi/formalism/queries.py:285,434`, which call the unbounded version even
  when an explicit purview list is passed and then intersect — pass the known
  purviews (or their maximum order) down instead of enumerating everything and
  intersecting.

**Correctness argument.** `irreducible_purviews` is a pure per-purview filter
(list comprehension at `substrate.py:667`), so restricting the candidate set
commutes with it: the bounded result equals the unbounded result filtered to
the cap.

**Tests.**

- The commutation property, asserted exactly, on a small substrate: dense and
  sparse connectivity, both directions, assorted mechanisms. (A reference test
  exists in color-qualia's `tests/test_patches.py`.)
- Cache non-aliasing: after a bounded call, an unbounded call for the same
  (direction, mechanism) returns the full result, and vice versa.
- The `repertoire_algebra.py:736` / `queries.py` sites no longer enumerate the
  full powerset when explicit purviews are passed (observable via a small-N
  behavioral test or call accounting; results unchanged).

**Acceptance.** Planning the color-qualia binding scope (1,372 explicit
mechanisms, purview cap 2, 21-unit dense substrate) completes in well under a
minute with counts identical to before; no unbounded caller can ever receive a
bounded cache entry. This makes color-qualia's `patches.py` monkeypatch
deletable.

### Item 2: substrate serialization is larger than the raw array

**Problem.** The 21-unit substrate's state-by-node TPM is a 352 MB float64
array; `serialize.save` to `.json.gz` produces 641 MB — ~1.8× the raw binary.
Campaigns store one copy per campaign directory and condor transfers it to
every job.

**Diagnosis to verify first.** The source request prescribes "route ndarrays
through a bytes path", but that path already exists: `pyphi/serialize/arrays.py`
encodes arrays as npy-format bytes, and the substrate schema already routes TPM
factors through it (`convert.py:932`, `schema.py:378`). The working hypothesis
for the observed size: the factored form of a dense 21-unit binary substrate
stores each node's full conditional distribution, ~2× the state-by-node array
(~704 MB raw); msgspec base64-encodes bytes fields in JSON (+33%, ~939 MB); and
gzip cannot compress well across the base64 boundary, landing near the measured
641 MB.

**Step 1 — measurement repro, before any fix.** Build synthetic dense
substrates at a few sizes (large enough to measure scaling; sigmoid
probabilities so the data is irrational and incompressible-ish) and record: raw
state-by-node bytes, summed factored-form bytes, `.json.gz` size, `.msgpack.gz`
size. If the hypothesis is refuted, re-diagnose before fixing — do not follow
this spec's fix blindly.

**Fix (assuming hypothesis confirmed).** Save campaign substrate payloads as
`.msgpack.gz` instead of `.json.gz` — msgspec writes `bytes` fields raw in
msgpack, eliminating base64 entirely. `serialize.save`/`load` already support
`format="msgpack"`; the change is at the campaign save/load/glob sites
(`pyphi/campaign/__init__.py:435,711,1300–1303`). This is an on-disk format
change: call it out in the commit message and final summary.

**Deferred follow-up (separate commit, only if still needed after
measurement).** Trimming the redundant factor axis for binary units (store one
probability slice, reconstruct the complement) — requires encode-time
elementwise verification that reconstruction is bit-identical, with fallback to
storing both slices. Not attempted unless msgpack alone leaves the file
unacceptably large.

**Tests.**

- Bit-identical round-trip: a substrate with irrational probabilities
  (sigmoids) round-trips through save/load with exact array equality and exact
  dtype preservation (float64 stays float64).
- Size regression: at a fixed synthetic size, the saved file is smaller than
  the equivalent `.json.gz` and within a fixed factor of the raw factored
  bytes — the factor set from the measurement repro's numbers, not guessed.

**Acceptance.** The on-disk substrate approaches (or beats) the raw array size;
the 1.8×-raw pathology is gone.

---

## Session B — item 4: `system_intrinsic_information` is ~4^N

**Problem.** Two compounding issues make the specified-state computation
infeasible beyond ~14 units (measured: n=8 → 1.8 s, n=10 → 37 s, ~4^N):

1. `intrinsic_information` (`pyphi/core/repertoire_algebra.py:617`)
   materializes `list(all_states(...))` (2^N Python tuples) and builds a
   2^N-entry dict, even on the composite-measure path where `dist` is already a
   full numpy array over all states.
2. `unconstrained_forward_effect_repertoire` (`repertoire_algebra.py:479`)
   computes `np.stack([...])` over all 2^N mechanism states — hundreds of TB at
   N=21; this OOM-hung an 18 GB machine.

**Changes.**

- **Pin-first tests, before touching the implementation:** on small systems,
  pin the winner (first tied state in enumeration order — C-order over the
  purview alphabet) and the tolerance-based tie family (`numerics.eq` against
  the raw maximum) against the current implementation, including a case with
  exact ties. Verify that `all_states` enumeration order matches C-order
  unraveling of the array — the vectorized rewrite depends on this equivalence.
- **Vectorize the composite path** of `intrinsic_information`: winner via
  argmax on the flattened `dist` (first-in-C-order on ties), tie family via a
  `numerics.eq` mask, materializing state tuples only for the (small) tie
  family. The non-composite path, which evaluates a per-state distance
  function, is out of scope. The explicit `states=` argument's semantics are
  preserved (when a caller passes a restricted state list, the current
  dict-based path remains for that case).
- **Running mean** in `unconstrained_forward_effect_repertoire`: accumulate
  `+=` into a single repertoire and divide by the count. Memory drops to one
  repertoire; the 2^N time remains. Bit-identity caveat: numpy's
  `stack(...).mean(axis=0)` uses pairwise summation above 128 elements, so
  results for mechanisms over 7 binary units may differ in final ulps;
  tolerance-based downstream comparisons absorb this. Document the caveat where
  the tests assert equality.
- **Size guard:** an explicit check that raises a clear "infeasible at this
  size" error with the estimated cost, instead of silently grinding, when the
  mechanism-state enumeration is too large. Threshold: a named module-level
  constant with the estimate in the error message.
- **Collect-time trap:** `_assemble_without_sia`
  (`pyphi/campaign/__init__.py:916`) silently computes
  `system_intrinsic_information` when `resolution_state` is None — for a big
  system, collect hangs after all shards succeeded. Warn (or raise with
  guidance) when the system size makes this infeasible, before starting the
  computation.
- No closed forms for particular unit types — explicitly a follow-up, not
  attempted here.

**Rider — context-manager lifetime audit** (small): one bug in the downstream
campaign came from holding a monkeypatch via an unreferenced
`contextmanager(...).__enter__()`, which the garbage collector finalized,
silently reverting the patch mid-run. Audit pyphi's own long-lived context
managers (`config.override` and any similar): if any internal code or
documented pattern encourages `.__enter__()` without a held reference, fix or
document it. Where a capability is meant to be enabled for a process's
lifetime, prefer explicit `install()`/`uninstall()` functions over a context
manager the caller must keep alive.

**Acceptance.** Winner and tie family identical to the current implementation
on the pinned small-system cases; memory for the unconstrained forward effect
repertoire is one repertoire regardless of N; oversized requests fail fast with
a cost estimate; collect never silently hangs on an infeasible
specified-state computation.

---

## Session C — item 3: per-cell resolution states in multi-cell `prepare_ces`

**Problem.** `prepare_ces` (`pyphi/campaign/__init__.py:515`) sweeps substrates
× states under one scope, and the shard plan is state-independent — cells
differing only by state share one planning pass. But `resolution_state` raises
for multi-cell campaigns, and without one, each cell plans SIA shards —
intractable at 21 units. Consequence: nine states on one substrate today means
nine single-cell campaigns, each re-planning the identical shard ladder.

**API.** `resolution_state` accepts, in addition to the current single-cell
`SystemStateSpecification`:

- a mapping keyed by the full `(label, formalism, subset, state)` cell tuples
  `_enumerate_cells` produces;
- as a convenience, a mapping keyed by state alone — valid only when the other
  three axes are singletons (ambiguous otherwise: error at prepare time);
- a callable `cell -> SystemStateSpecification`.

All forms are normalized internally to one callable; downstream code sees a
single shape. `sia` remains single-cell only (out of scope here).

**Validation at prepare time.** Each resolved value must be a
`SystemStateSpecification` (or duck-type: indexable by `Direction.CAUSE` /
`Direction.EFFECT` yielding state specifications). A wrong type — e.g. a plain
state tuple — currently fails only at collect time, deep inside
`resolve_congruence`, with an opaque error; instead raise at prepare time with
a message pointing at `system_intrinsic_information` as the way to produce one.
Every enumerated cell must resolve (missing key → clear error naming the cell).

**Serialization.** Replace the single `resolution_state.json.gz`
(`campaign/__init__.py:610`) with one file holding per-cell records (the specs
are small; one file, cell-keyed). On-disk format change: call out in commit
message and final summary; the version guard covers safety.

**Behavior.** SIA shards are suppressed for every cell exactly as the
single-cell path does today. At collect, `_collect_ces` → `_merge_cell`
resolves each cell's congruence against that cell's own state. `collect`'s
override parameters (which carry the same single-cell-only restriction) accept
the same shapes, normalized the same way.

**Tests.**

- Prepare-time validation: wrong type, missing cell, ambiguous state-keyed
  mapping — each fails at prepare with the specified messages.
- Equivalence (the acceptance criterion, on a small substrate, run locally):
  one campaign spanning several states on one substrate plans once, produces
  the same shard set a matching group of single-cell campaigns would, and
  collects structures identical to the single-cell campaigns' — each
  congruence-resolved against its own state.

---

## Session kickoff prompts

Paste one line into a fresh session in the pyphi repo:

- **A:** `Read docs/superpowers/specs/2026-07-22-upstream-campaign-fixes-design.md and execute Session A (items 1 and 2), following its session protocol.`
- **B:** `Read docs/superpowers/specs/2026-07-22-upstream-campaign-fixes-design.md and execute Session B (item 4 plus the context-manager rider), following its session protocol.`
- **C:** `Read docs/superpowers/specs/2026-07-22-upstream-campaign-fixes-design.md and execute Session C (item 3), following its session protocol.`
