# Uncertainty Pipeline (Minimal) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the §8 "minimal first build" of
`docs/superpowers/specs/2026-07-07-tpm-from-data-and-uncertainty-design.md`
(on branch `worktree-tpm-uncertainty-exploration`): a data-to-substrate
estimation path with epistemic uncertainty carried honestly. Concretely:
`estimate_substrate(data, *, regime, prior)` → a `SubstratePosterior`
(independent Beta posteriors per TPM cell, Jeffreys default) with a
first-class `CoverageReport`; `.sample()` → an ordinary `Substrate` so the
whole existing compute stack is reused unchanged; `phi_posterior(...)` → a
`PhiPosterior` mixture object (`p_positive`, conditional quantiles, raw
samples, complex-identity categorical) that **refuses bare-float coercion**;
plus `edge_probability` replacing the exact-equality connectivity oracle for
estimated TPMs, and committed reproducible tests for the spec's three inline
demonstrations (twin-substrate non-identifiability, `infer_cm` saturation,
ε-boundary sensitivity).

**Architecture:** One new module, `pyphi/estimate.py`. The estimand is the
per-unit factorization the library already stores (each TPM cell
`P(unit_i = ON | current state)` gets an independent conjugate
Beta(counts + prior) posterior), so estimation is pure counting and sampling
is one `rng.beta` call plus the existing
`convert.to_multidimensional` → `Substrate(tpm=...)` construction. Nothing
downstream of `Substrate` changes: `phi_posterior` is a thin Monte Carlo
driver over `analyze(...)` and `maximal_complex(...)`. Uncertainty semantics
are enforced at the type level: `PhiPosterior.__float__` raises, pointing at
the named summaries; there is no posterior-mean substrate and no point
estimate anywhere on the surface. The regime ("perturbational" |
"observational") is a required constructor-time caller assertion, stamped
into a new structured `estimator` slot on `Provenance`. Binary units only;
k-ary data raises with a clear message (the Dirichlet generalization is
mechanical but out of scope).

**Tech Stack:** Python 3.13, numpy, msgspec, pytest. No new dependencies.

## Global Constraints

- Run everything with `uv run` (e.g. `uv run pytest`, `uv run python`).
- Work in a git worktree under `.claude/worktrees/` (confirm branch name with
  the user at execution start; base on the current working branch).
- Float comparisons in tests use `pytest.approx` (default tolerance) — never
  `==` on Φ or probability values.
- Every user-facing change gets a changelog fragment in `changelog.d/`
  (`<name>.<type>.md`), committed with the task.
- Docstrings describe final state only — no migration narrative, no planning
  artifacts (no task numbers, no "N12"/"Wave 7", no design-alternative
  discussion).
- Do not use `git checkout -- <path>` for cleanup; other sessions may have
  unrelated working-tree changes — stage only files this plan touches.
- Never pass `--no-verify` to git. If pre-commit hooks fail, fix the failure.
- The final verification (Task 6) must run `uv run pytest` **with no path
  argument** at least once (bare paths skip the doctest sweep).
- Reproducibility house rules apply to the library code itself: every
  randomized public entry point takes an explicit seed (or Generator), uses
  an isolated `np.random.default_rng` instance (never module-level seeding),
  and the seed is stored on the result object and in provenance.

## Background for implementers (read once)

**Semantics that are settled — encode, do not redesign.** Three decisions
were made upstream of this plan and the implementation must match them
exactly:

1. **No bare-float coercion on uncertain results.** `float(phi_posterior)`
   raises `TypeError` with a message naming the honest summaries
   (`p_positive`, `quantiles`, `conditional_quantiles`). Rationale (spec §5):
   the posterior over Φ is generically a mixture of a point mass at zero
   (reducible samples — the system does not exist as one entity) and a
   continuous density; its mean describes no possible system.
2. **Perturbational summaries are free; observational access is gated by the
   constructor-time assertion.** The caller must pass
   `regime="perturbational"` or `regime="observational"` — there is no
   default and the library cannot infer it from data. The assertion is
   stamped into provenance so every downstream result records which causal
   claim it rests on. The coverage report is attached in both regimes; in
   the observational regime it is the headline (unvisited rows are
   *unidentified*, not merely unsampled — spec §3.4).
3. **Jeffreys `Beta(½, ½)` is the default prior, not Laplace** — settled by
   the spec's paired prior experiment (§3.5): Laplace has the worst boundary
   bias at every sample size; Jeffreys roughly halves it.

**Explicit non-builds** (spec §8): no distribution-valued Φ threaded through
existing result types/display/schemas; no "distribution over Φ-structures"
summary object (label-switching is an open research problem); no GLM/Ising
fitting (counts model only); no ensemble axis on `sweep`.

**The construction primitive.** A state-by-node `P(on)` matrix (shape
`(2**n, n)`, rows in **little-endian** state order: row index
`sum(bit_i << i)`) becomes a substrate via
`convert.to_multidimensional(pon)` (`pyphi/convert.py:176`) →
`Substrate(tpm=multidim, node_labels=...)` (`pyphi/substrate.py:88`). The
spec validated the round trip: feeding `basic_substrate`'s exact ground-truth
probabilities through this path recovers Φ = 0.415037 to all printed digits.
Because `basic_substrate` (OR, AND, XOR) is *asymmetric*, a ground-truth
round-trip test doubles as an endianness/axis-order guard — symmetric
fixtures cannot catch a reversed axis.

**Existing seams used, verified at these locations:**

- `pyphi.analyze(substrate, state, *, subset=None, formalism=None, compute=None)`
  (`pyphi/analyze.py:80`); `compute="sia"` returns the raw SIA.
- `pyphi.substrate.maximal_complex(substrate, state)` (`substrate.py:1057`)
  returns a `Complex` with `.node_indices` (tuple) and `.phi`; when no
  irreducible candidate exists it returns a falsy null `Complex` with empty
  `node_indices` — record `()` for those samples.
- `Provenance` (`pyphi/provenance.py:60`) is a frozen dataclass
  (`pyphi_version` … `wall_time`, `seed`, `note`) with
  `Provenance.capture(*, wall_time=None, seed=None)`;
  `ProvenanceSchema` (`pyphi/serialize/schema.py:208`) mirrors it
  field-for-field with trailing defaults, so appending a defaulted field is
  backward compatible (msgspec decodes missing fields to defaults).
- Serialization registry: one frozen `msgspec.Struct` per type in
  `pyphi/serialize/schema.py`; one `_register_<type>()` in
  `pyphi/serialize/convert.py` populating `_ENCODERS`/`_DECODERS`. numpy
  arrays already round-trip via the bytes treatment used for repertoires —
  find the existing array encode/decode helpers in `convert.py` (used by the
  repertoire/TPM schemas) and reuse them verbatim.
- `utils.eq` / `utils.is_positive` (`pyphi/utils.py:134,142`) are the
  precision-respecting comparators; there is **no** `utils.is_zero` in this
  tree.
- `dynamics.simulate(tpm, initial_state, timesteps, clamp=None, rng=None)`
  (`pyphi/dynamics.py:40`) generates observational trajectories; pass an
  explicit `np.random.default_rng(seed)` — never leave `rng=None` in tests.
- `FactoredTPM.infer_cm()` (`pyphi/core/tpm/factored.py:314`) marks an edge
  when a factor varies along an input axis by more than `10^-precision`.
  On any continuously-estimated TPM every factor varies along every axis by
  sampling noise, so it saturates to all-ones (spec §5.4, verified). The
  `Substrate` constructor infers its connectivity this way, so estimated
  substrates are fully connected — that is honest (dependence is not
  excluded by the data) and is exactly why `edge_probability` exists as the
  graded replacement.

**Fixture facts** (from the spec's seeded demos, reproducible):
`basic_substrate` true Φ = 0.415037 at state `(1, 0, 0)`; free-running from
that state it falls onto a short orbit visiting only **3 of 8** states, and
a twin substrate altered on the 5 unvisited rows produces identical
observational data with Φ ≈ 0.327. `grid3_substrate` true Φ = 0.024666 at
`(0, 0, 0)` (near the reducibility boundary); its true cm has
`cm[0, 2] = cm[2, 0] = 0`; at the true TPM candidates `{0}` and `{2}` tie
exactly in system φ, so under a posterior the complex identity is genuinely
categorical (demo split ≈ 62/31/7 across `{0}`/`{2}`/`{1}`).
`xor_substrate` true Φ = 1.5 at `(0, 0, 0)`.

**Data model for `estimate_substrate`.** `data` is either

- a pair `(current, next)` of integer arrays, each shape `(T, n)` — one row
  per transition; the natural form for perturbational trials, where each
  `do(current) → next` observation is independent; or
- a single integer array of shape `(T, n)` — a trajectory; transitions are
  consecutive row pairs `(data[:-1], data[1:])`; the natural form for
  observational recordings.

Both forms are accepted under both regimes (the counts are identical);
`regime` remains an independent semantic assertion about how the data was
produced. Values must be 0/1; anything else raises `ValueError` naming the
binary-only limitation.

---

### Task 1: Counting estimator, `SubstratePosterior`, `CoverageReport`

**Files:**
- Create: `pyphi/estimate.py`
- Test: `test/test_estimate.py` (create)
- Create: `changelog.d/estimate-substrate.feature.md`

**Interfaces:**
- Produces:
  - `estimate_substrate(data, *, regime, prior=0.5, node_labels=None, model="counts") -> SubstratePosterior`.
  - `SubstratePosterior` — frozen dataclass (`eq=False`): `alpha_on`,
    `alpha_off` (float arrays, shape `(2**n, n)`), `regime: str`,
    `prior: float`, `coverage: CoverageReport`, `node_labels`,
    `provenance: Provenance`. Methods: `sample(*, seed=None, rng=None) ->
    Substrate` (exactly one of seed/rng required), `n_units`, `n_states`
    properties.
  - `CoverageReport` — frozen dataclass (`eq=False`): `counts` (int array,
    shape `(2**n,)` — transitions observed per current state), `n_units`.
    Properties: `n_states`, `uncovered_states` (tuple of little-endian state
    tuples with zero observations), `fraction_covered`, `is_complete`.
    `to_pandas()` returns a long-format DataFrame (`state`, `count`).
- Consumed by Tasks 2–6.

- [ ] **Step 1: Write the failing tests**

Create `test/test_estimate.py`:

```python
"""Tests for substrate estimation from data and the epistemic-uncertainty layer."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi.estimate import CoverageReport
from pyphi.estimate import SubstratePosterior
from pyphi.estimate import estimate_substrate


@pytest.fixture(autouse=True)
def _quiet():
    with pyphi.config.override(progress_bars=False):
        yield


def _ground_truth_pon(substrate):
    """State-by-node P(unit = ON | current state), little-endian row order."""
    import itertools

    ft = substrate.factored_tpm
    n = ft.n_nodes
    pon = np.zeros((2**n, n))
    for state in itertools.product((0, 1), repeat=n):
        row = sum(bit << i for i, bit in enumerate(state))
        for i in range(n):
            factor = ft.factor(i)
            idx = tuple(
                state[j] if factor.shape[j] > 1 else 0 for j in range(n)
            ) + (1,)
            pon[row, i] = factor[idx]
    return pon


def _exhaustive_transitions(substrate, repeats=1):
    """Deterministic (current, next) pairs covering every state ``repeats`` times."""
    import itertools

    pon = _ground_truth_pon(substrate)
    n = pon.shape[1]
    current = []
    for state in itertools.product((0, 1), repeat=n):
        row = sum(bit << i for i, bit in enumerate(state))
        for _ in range(repeats):
            current.append((state, tuple((pon[row] > 0.5).astype(int))))
    cur, nxt = zip(*current)
    return np.array(cur), np.array(nxt)


def test_regime_is_required():
    data = (np.zeros((2, 3), dtype=int), np.zeros((2, 3), dtype=int))
    with pytest.raises(TypeError):
        estimate_substrate(data)  # pyright: ignore[reportCallIssue]
    with pytest.raises(ValueError, match="regime"):
        estimate_substrate(data, regime="empirical")


def test_binary_only():
    data = (np.full((2, 3), 2), np.zeros((2, 3), dtype=int))
    with pytest.raises(ValueError, match="binary"):
        estimate_substrate(data, regime="perturbational")


def test_counts_model_only():
    data = (np.zeros((2, 3), dtype=int), np.zeros((2, 3), dtype=int))
    with pytest.raises(NotImplementedError):
        estimate_substrate(data, regime="perturbational", model="glm")


def test_posterior_mean_recovers_asymmetric_ground_truth():
    """With exact deterministic transitions and a vanishing prior, the
    posterior concentrates on the true asymmetric TPM (this is also the
    endianness/axis-order guard: OR/AND/XOR has no symmetry to hide a
    reversed axis)."""
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate, repeats=50)
    posterior = estimate_substrate(data, regime="perturbational", prior=1e-6)
    mean = posterior.alpha_on / (posterior.alpha_on + posterior.alpha_off)
    np.testing.assert_allclose(mean, _ground_truth_pon(substrate), atol=1e-4)


def test_sample_returns_working_substrate_and_phi_converges():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate, repeats=200)
    posterior = estimate_substrate(data, regime="perturbational", prior=0.05)
    sampled = posterior.sample(seed=7)
    assert isinstance(sampled, type(substrate))
    sia = pyphi.analyze(sampled, (1, 0, 0), compute="sia")
    assert float(sia.phi) == pytest.approx(0.415037, abs=0.05)


def test_sample_seed_discipline():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate)
    posterior = estimate_substrate(data, regime="perturbational")
    with pytest.raises(ValueError, match="seed"):
        posterior.sample()
    a = posterior.sample(seed=3)
    b = posterior.sample(seed=3)
    np.testing.assert_array_equal(
        np.asarray(a.factored_tpm.factor(0)), np.asarray(b.factored_tpm.factor(0))
    )
    rng = np.random.default_rng(3)
    c = posterior.sample(rng=rng)
    np.testing.assert_array_equal(
        np.asarray(a.factored_tpm.factor(0)), np.asarray(c.factored_tpm.factor(0))
    )


def test_trajectory_form_equals_pair_form():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1], [1, 0, 0]])
    from_traj = estimate_substrate(traj, regime="observational")
    from_pairs = estimate_substrate(
        (traj[:-1], traj[1:]), regime="observational"
    )
    np.testing.assert_array_equal(from_traj.alpha_on, from_pairs.alpha_on)
    np.testing.assert_array_equal(
        from_traj.coverage.counts, from_pairs.coverage.counts
    )


def test_jeffreys_is_default_prior():
    data = (np.zeros((1, 2), dtype=int), np.zeros((1, 2), dtype=int))
    posterior = estimate_substrate(data, regime="perturbational")
    assert posterior.prior == pytest.approx(0.5)
    # An unvisited row sits at the bare prior.
    assert posterior.alpha_on[3, 0] == pytest.approx(0.5)
    assert posterior.alpha_off[3, 0] == pytest.approx(0.5)


def test_coverage_report():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]])
    posterior = estimate_substrate(traj, regime="observational")
    report = posterior.coverage
    assert isinstance(report, CoverageReport)
    assert report.n_states == 8
    assert report.counts[0] == 2  # (0,0,0) observed as a current state twice
    assert report.counts[1] == 1  # (1,0,0) once (the final row has no successor)
    assert not report.is_complete
    assert report.fraction_covered == pytest.approx(2 / 8)
    assert (0, 1, 0) in report.uncovered_states
    assert (0, 0, 0) not in report.uncovered_states
    df = report.to_pandas()
    assert set(df.columns) >= {"state", "count"}
    assert len(df) == 8


def test_full_coverage_report_is_complete():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate)
    posterior = estimate_substrate(data, regime="perturbational")
    assert posterior.coverage.is_complete
    assert posterior.coverage.uncovered_states == ()
    assert posterior.coverage.fraction_covered == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_estimate.py -v`
Expected: FAIL at import (`ModuleNotFoundError: No module named
'pyphi.estimate'`).

- [ ] **Step 3: Implement `pyphi/estimate.py`**

Module structure (docstrings must state the interventional semantics of the
TPM, the two regimes, and the identifiability caveat — this module is the
honest path and the docstrings carry that honesty):

```python
"""Estimate substrates from observed transitions, with epistemic uncertainty.

The TPM that IIT requires is interventional — ``p(next | do(current))``
under a uniform perturbation of all current states — so what data can
legitimately provide depends on how the data was produced. The caller must
assert the ``regime``:

- ``"perturbational"``: each transition is an independent trial in which the
  current state was set by intervention. When the trials cover the state
  space, the estimand is identified and estimation is a counting problem.
- ``"observational"``: transitions come from a passively recorded
  trajectory. Treating them as interventional assumes the recorded dynamics
  are the causal dynamics (no unobserved driver, correct units and grain,
  stationarity) — assumptions about the world that the data cannot verify.
  States the trajectory never visits are *unidentified*, not merely
  unsampled: substrates that differ only on unvisited rows produce
  identical data and can have materially different Φ. The
  :class:`CoverageReport` records exactly which rows the data constrained.

Estimation is per-unit counting with a conjugate symmetric Beta prior on
every cell of the state-by-node TPM (default Jeffreys, ``a = 1/2``). The
result is a :class:`SubstratePosterior` — a distribution over substrates,
never a single point estimate: Φ of a posterior-mean TPM conflates
epistemic uncertainty with genuine indeterminism and suppresses Φ where the
data is merely uninformative.
"""
```

Implementation notes:

- `_row_index(state) = sum(bit << i for i, bit in enumerate(state))`
  (little-endian, matching `convert.to_multidimensional` row order).
- `estimate_substrate`: normalize `data` to `(current, next)` arrays;
  validate shapes match, values are 0/1 (`ValueError` mentioning "binary"),
  `regime in ("perturbational", "observational")` (`ValueError` mentioning
  "regime"), `model == "counts"` (`NotImplementedError`), `prior > 0`.
  Accumulate `counts_on`, `counts_off` (shape `(2**n, n)`) and per-row
  `row_counts` (shape `(2**n,)`); `alpha_on = counts_on + prior`,
  `alpha_off = counts_off + prior`. Build `CoverageReport(counts=row_counts,
  n_units=n)` and `Provenance.capture()` (the structured estimator record is
  Task 2).
- `SubstratePosterior.sample(*, seed=None, rng=None)`: raise `ValueError`
  ("provide a seed or rng") unless exactly one is given;
  `rng = np.random.default_rng(seed)` when seed is given;
  `p_on = rng.beta(self.alpha_on, self.alpha_off)`;
  `convert.to_multidimensional(p_on)` → `Substrate(tpm=..., node_labels=...)`.
- `CoverageReport.uncovered_states`: decode zero-count row indices to
  little-endian state tuples. `to_pandas`: long format, one row per state,
  columns `state` (tuple) and `count`.
- Both dataclasses `frozen=True, eq=False` (ndarray fields).
- Use module-level imports; `Substrate` may need a deferred import if a
  cycle appears (match the codebase's existing deferred-import pattern).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_estimate.py -v`
Expected: all PASS. The convergence test computes a handful of small SIAs
(~seconds).

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/estimate-substrate.feature.md <<'EOF'
Added `pyphi.estimate`: `estimate_substrate(data, *, regime, prior)` builds
a `SubstratePosterior` (independent Beta posteriors over every TPM cell,
Jeffreys prior by default) from perturbational transition pairs or an
observational trajectory, with a first-class `CoverageReport` recording
which states the data constrained. `SubstratePosterior.sample()` draws an
ordinary `Substrate`, so all existing computations apply to posterior
samples unchanged. The data regime is a required caller assertion.
EOF
git add pyphi/estimate.py test/test_estimate.py changelog.d/estimate-substrate.feature.md
git commit -m "Add substrate estimation from data with Beta posteriors and coverage reporting"
```

---

### Task 2: Structured estimator record on `Provenance`

**Files:**
- Modify: `pyphi/provenance.py` (`Provenance` dataclass ~line 60, `capture`
  ~line 76, display rows ~line 121)
- Modify: `pyphi/serialize/schema.py` (`ProvenanceSchema` ~line 208)
- Modify: `pyphi/serialize/convert.py` (the provenance register function)
- Modify: `pyphi/estimate.py` (stamp the record)
- Test: `test/test_estimate.py` (extend); `test/serialize` (extend the
  existing provenance round-trip test file)

**Interfaces:**
- Produces:
  - `Provenance.estimator: dict[str, Any] | None = None` (trailing defaulted
    field; JSON-able scalars only) and a matching `capture(*, ...,
    estimator=None)` kwarg.
  - `estimate_substrate` stamps
    `{"regime", "model", "prior", "n_transitions", "n_states_observed",
    "n_states_total", "uncovered_state_count"}`.
  - `ProvenanceSchema.estimator: dict | None = None` appended after `note`
    (old serialized results decode with `estimator=None`).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_estimate.py`:

```python
def test_provenance_records_estimator():
    substrate = examples.basic_substrate()
    data = _exhaustive_transitions(substrate, repeats=2)
    posterior = estimate_substrate(data, regime="perturbational", prior=0.5)
    record = posterior.provenance.estimator
    assert record is not None
    assert record["regime"] == "perturbational"
    assert record["model"] == "counts"
    assert record["prior"] == pytest.approx(0.5)
    assert record["n_transitions"] == 16
    assert record["n_states_observed"] == 8
    assert record["n_states_total"] == 8
    assert record["uncovered_state_count"] == 0


def test_provenance_records_observational_assertion():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    posterior = estimate_substrate(traj, regime="observational")
    record = posterior.provenance.estimator
    assert record["regime"] == "observational"
    assert record["uncovered_state_count"] == 6
```

Extend the provenance serialization tests (find the existing file under
`test/serialize/` that round-trips `Provenance` and match its style): a
round trip preserves a non-`None` `estimator` dict, and decoding a payload
without the field yields `estimator is None`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_estimate.py -k provenance test/serialize -k provenance -v`
Expected: FAIL (`Provenance` has no `estimator`).

- [ ] **Step 3: Implement**

- Add `estimator: dict[str, Any] | None = None` after `note` on
  `Provenance`; thread through `capture`. Docstring: "Structured record of
  how an estimated input was produced (data regime, estimator model, prior,
  sample counts). ``None`` for results computed from an exactly specified
  substrate." Add a display row (after the `note` row): when present, render
  a compact `regime=…, model=…, n_transitions=…` summary.
- Append `estimator: dict | None = None` to `ProvenanceSchema`; extend the
  encoder/decoder in `convert.py` (field pass-through).
- In `estimate_substrate`, build the dict from the coverage report and pass
  `Provenance.capture(estimator=...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_estimate.py test/serialize -q`
Expected: all PASS, including all pre-existing provenance tests.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/provenance-estimator.feature.md <<'EOF'
`Provenance` gained a structured `estimator` slot recording how an
estimated substrate was produced (data regime, estimator model, prior,
sample counts); populated by `pyphi.estimate.estimate_substrate` and
carried through serialization.
EOF
git add pyphi/provenance.py pyphi/serialize/schema.py pyphi/serialize/convert.py pyphi/estimate.py test/test_estimate.py test/serialize changelog.d/provenance-estimator.feature.md
git commit -m "Record estimator metadata in a structured Provenance slot"
```

---

### Task 3: `phi_posterior` and the `PhiPosterior` mixture object

**Files:**
- Modify: `pyphi/estimate.py`
- Test: `test/test_estimate.py` (extend)
- Create: `changelog.d/phi-posterior.feature.md`

**Interfaces:**
- Produces:
  - `phi_posterior(posterior, state, *, n_samples, seed, subset=None) ->
    PhiPosterior` — Monte Carlo over `posterior.sample(rng=...)`; per draw,
    the SIA of the candidate system (`subset`, default the whole substrate)
    via `analyze(sample, state, subset=subset, compute="sia")`, and the
    complex identity via `maximal_complex(sample, state)`. `seed` is
    required (keyword-only, no default).
  - `PhiPosterior` — frozen dataclass (`eq=False`): `samples` (float array,
    shape `(n_samples,)` — the raw per-draw Φ values), `complex_samples`
    (tuple of unit-index tuples, one per draw; `()` when no complex),
    `state`, `subset`, `seed`, `regime`, `coverage`, `provenance`.
    - `p_positive: float` — fraction of samples with
      `utils.is_positive(phi)`.
    - `quantiles(qs) -> np.ndarray` — over all samples (the mixture).
    - `conditional_quantiles(qs) -> np.ndarray | None` — over the positive
      samples only; `None` when no sample is positive.
    - `complex_identity: dict[tuple[int, ...], float]` — categorical over
      the selected complex's units.
    - `__float__` raises `TypeError` naming the summaries (and the
      uncovered-row count when coverage is partial).

- [ ] **Step 1: Write the failing tests**

Append to `test/test_estimate.py`:

```python
from pyphi.estimate import PhiPosterior
from pyphi.estimate import phi_posterior


@pytest.fixture(scope="module")
def grid3_posterior():
    """A seeded posterior over grid3 from sparse perturbational data."""
    substrate = examples.grid3_substrate()
    pon = _ground_truth_pon(substrate)
    rng = np.random.default_rng(20260708)
    current, nxt = [], []
    for row in range(8):
        state = tuple((row >> i) & 1 for i in range(3))
        for _ in range(5):
            current.append(state)
            nxt.append(tuple(rng.random(3) < pon[row]))
    data = (np.array(current), np.array(nxt, dtype=int))
    return estimate_substrate(data, regime="perturbational")


@pytest.fixture(scope="module")
def grid3_phi_posterior(grid3_posterior):
    with pyphi.config.override(progress_bars=False):
        return phi_posterior(
            grid3_posterior, (0, 0, 0), n_samples=40, seed=99
        )


def test_phi_posterior_seed_required(grid3_posterior):
    with pytest.raises(TypeError):
        phi_posterior(grid3_posterior, (0, 0, 0), n_samples=2)  # pyright: ignore[reportCallIssue]


def test_phi_posterior_is_reproducible(grid3_posterior):
    a = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=5, seed=42)
    b = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=5, seed=42)
    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.complex_samples == b.complex_samples
    assert a.seed == 42


def test_phi_posterior_is_a_mixture(grid3_phi_posterior):
    pp = grid3_phi_posterior
    assert pp.samples.shape == (40,)
    assert 0.0 < pp.p_positive < 1.0  # both mixture components present
    lo, hi = pp.quantiles([0.025, 0.975])
    assert lo == pytest.approx(0.0)
    assert hi > 0.0
    cond = pp.conditional_quantiles([0.5])
    assert cond is not None and cond[0] > 0.0


def test_phi_posterior_refuses_float_coercion(grid3_phi_posterior):
    with pytest.raises(TypeError, match="p_positive"):
        float(grid3_phi_posterior)


def test_complex_identity_is_categorical(grid3_phi_posterior):
    identity = grid3_phi_posterior.complex_identity
    assert sum(identity.values()) == pytest.approx(1.0)
    # grid3's exact {0} vs {2} tie at the true TPM is broken randomly by
    # the data, so more than one identity appears.
    assert len(identity) > 1
    assert grid3_phi_posterior.complex_samples[0] in identity


def test_phi_posterior_carries_regime_and_coverage(grid3_phi_posterior):
    assert grid3_phi_posterior.regime == "perturbational"
    assert grid3_phi_posterior.coverage.is_complete


def test_observational_result_reports_partial_coverage():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]] * 3)
    posterior = estimate_substrate(traj, regime="observational")
    pp = phi_posterior(posterior, (1, 0, 0), n_samples=3, seed=1)
    assert pp.regime == "observational"
    assert not pp.coverage.is_complete
    with pytest.raises(TypeError, match="unconstrained|uncovered"):
        float(pp)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_estimate.py -k "phi_posterior or complex_identity or observational_result" -v`
Expected: FAIL at import (`PhiPosterior` does not exist).

- [ ] **Step 3: Implement**

- `phi_posterior`: one `rng = np.random.default_rng(seed)` drives all draws
  (each `posterior.sample(rng=rng)` advances the same stream). Per draw:
  `sia = analyze(sample, state, subset=subset, compute="sia")` →
  `float(sia.phi)`; `complex_ = maximal_complex(sample, state)` →
  `tuple(complex_.node_indices)`. Progress: respect
  `config.infrastructure.progress_bars` via the same tqdm helper other
  drivers use (find the existing pattern; a plain loop is acceptable if no
  shared helper fits). Provenance: `Provenance.capture(seed=seed,
  estimator=posterior.provenance.estimator)` so the estimator record
  travels to the result.
- `PhiPosterior.__float__`:

```python
    def __float__(self) -> float:
        detail = ""
        if not self.coverage.is_complete:
            n = len(self.coverage.uncovered_states)
            detail = (
                f" {n} of {self.coverage.n_states} states were never"
                " observed, so the underlying TPM is unconstrained there"
                " (uncovered states are unidentified, not merely unsampled)."
            )
        raise TypeError(
            "A PhiPosterior is a distribution over Φ (generically a mixture"
            " of a point mass at zero and a continuous density) and cannot"
            " be summarized by one float. Use .p_positive, .quantiles(qs),"
            " .conditional_quantiles(qs), or .samples." + detail
        )
```

- `p_positive` uses `utils.is_positive` per sample (precision-respecting,
  never `> 0`). `conditional_quantiles` filters with the same predicate.
- `complex_identity`: `collections.Counter` over `complex_samples`,
  normalized.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_estimate.py -v`
Expected: all PASS. The module-scoped 40-draw fixture computes ~80 small
SIAs/complex searches (n = 3; the spec's 300-draw demo ran in well under a
minute, so this is tens of seconds at most). If it exceeds ~60 s, reduce to
25 draws — the assertions are qualitative.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/phi-posterior.feature.md <<'EOF'
Added `pyphi.estimate.phi_posterior`: Monte Carlo propagation of a
`SubstratePosterior` through the SIA, returning a `PhiPosterior` that
reports the mixture honestly — `p_positive` (the probability the system is
integrated at all), unconditional and conditional quantiles, the raw Φ
samples, and the complex-identity categorical (which unit set is maximal,
per sample). A `PhiPosterior` cannot be coerced to a bare float; the error
names the honest summaries and, when state coverage is partial, the
unconstrained states.
EOF
git add pyphi/estimate.py test/test_estimate.py changelog.d/phi-posterior.feature.md
git commit -m "Add Monte Carlo phi posterior with mixture summaries and complex-identity categorical"
```

---

### Task 4: `edge_probability` and the three committed reproducibility checks

**Files:**
- Modify: `pyphi/estimate.py`
- Test: `test/test_estimate.py` (extend)
- Create: `changelog.d/edge-probability.feature.md`

**Interfaces:**
- Produces: `SubstratePosterior.edge_probability(*, n_samples, seed,
  threshold) -> np.ndarray` — an `(n, n)` matrix; entry `(a, b)` is the
  fraction of posterior samples in which unit `b`'s estimated conditional
  varies by more than `threshold` along input axis `a` (maximum absolute
  difference in `P(b = ON)` between current states differing only in unit
  `a`). `threshold` is keyword-only with **no default** — any default would
  smuggle in a modeling choice, exactly what the `regime` assertion exists
  to avoid.

- [ ] **Step 1: Write the failing tests**

Append to `test/test_estimate.py`. These commit the spec's three inline
demonstrations as reproducible tests (each was previously an unreproducible
inline check):

```python
def test_infer_cm_saturates_on_estimated_tpm(grid3_posterior):
    """The exact-equality connectivity oracle reports every edge present on
    any continuously-estimated TPM: sampling noise exceeds 10^-precision on
    every axis. This documents the defect edge_probability replaces."""
    sample = grid3_posterior.sample(seed=0)
    inferred = sample.factored_tpm.infer_cm()
    assert inferred.all()  # including the truly absent edges 0->2 and 2->0


def test_edge_probability_discriminates_where_infer_cm_cannot():
    # grid3's weakest true edges (1->0, 1->2) vary the target's P(ON) by only
    # ~0.072, so the discrimination threshold must sit below that signal and
    # above the estimation noise floor. At 0.05 with enough data per state the
    # separation is exact and seed-robust (present edges fire 1.0, absent 0.0);
    # a threshold above 0.072 would only let the weak edges fire via noise.
    substrate = examples.grid3_substrate()
    pon = _ground_truth_pon(substrate)
    rng = np.random.default_rng(20260708)
    current, nxt = [], []
    for row in range(8):
        state = tuple((row >> i) & 1 for i in range(3))
        for _ in range(2000):
            current.append(state)
            nxt.append(tuple(rng.random(3) < pon[row]))
    posterior = estimate_substrate(
        (np.array(current), np.array(nxt, dtype=int)), regime="perturbational"
    )
    prob = posterior.edge_probability(n_samples=200, seed=5, threshold=0.05)
    true_cm = substrate.cm
    # Truly absent edges get low probability; present edges get high.
    assert prob[0, 2] < 0.1 and prob[2, 0] < 0.1
    assert (prob[true_cm.astype(bool)] > 0.9).all()


def test_observational_twin_nonidentifiability():
    """Two substrates identical on the observed orbit but different on the
    unvisited rows produce identical observational data yet materially
    different Φ. Deterministic: basic_substrate's free-running dynamics are
    deterministic, so no sampling is involved."""
    from pyphi import Substrate
    from pyphi import convert
    from pyphi import dynamics

    substrate = examples.basic_substrate()
    pon = _ground_truth_pon(substrate)
    rng = np.random.default_rng(1)  # unused by deterministic dynamics
    # Simulate from the state-by-node multidimensional form (the form
    # dynamics.simulate documents); both substrates go through the same path.
    traj = np.array(
        dynamics.simulate(
            convert.to_multidimensional(pon), (1, 0, 0), timesteps=50, rng=rng
        )
    )
    visited = {sum(bit << i for i, bit in enumerate(row)) for row in map(tuple, traj)}
    assert len(visited) == 3  # the orbit covers 3 of 8 states

    twin_pon = pon.copy()
    for row in set(range(8)) - visited:
        twin_pon[row] = 1.0 - twin_pon[row]  # arbitrary but deterministic
    twin = Substrate(tpm=convert.to_multidimensional(twin_pon))

    twin_traj = np.array(
        dynamics.simulate(
            convert.to_multidimensional(twin_pon), (1, 0, 0), timesteps=50, rng=rng
        )
    )
    np.testing.assert_array_equal(traj, twin_traj)  # identical data...

    phi_true = float(pyphi.analyze(substrate, (1, 0, 0), compute="sia").phi)
    phi_twin = float(pyphi.analyze(twin, (1, 0, 0), compute="sia").phi)
    assert phi_true == pytest.approx(0.415037)
    assert abs(phi_true - phi_twin) > 0.05  # ...materially different Φ


def test_epsilon_boundary_sensitivity():
    """Pushing a deterministic TPM off the 0/1 boundary by epsilon lowers Φ
    monotonically: indeterminism (real or estimated) shrinks selectivity."""
    from pyphi import Substrate
    from pyphi import convert

    pon = _ground_truth_pon(examples.basic_substrate())
    phis = []
    for eps in (0.0, 0.001, 0.02):
        smoothed = np.clip(pon, eps, 1.0 - eps)
        substrate = Substrate(tpm=convert.to_multidimensional(smoothed))
        phis.append(float(pyphi.analyze(substrate, (1, 0, 0), compute="sia").phi))
    assert phis[0] == pytest.approx(0.415037)
    assert phis[0] > phis[1] > phis[2]
    assert phis[1] == pytest.approx(0.413, abs=0.005)
    assert phis[2] == pytest.approx(0.374, abs=0.01)
```

Note on the twin test: the demo established the 3-state orbit and the
Φ ≈ 0.327 twin. The twin construction here (inverting unvisited rows) is a
specific deterministic choice; if the resulting Φ gap is smaller than 0.05,
loosen only after checking against the demo's construction
(`demoA_raw_seed1.npz` in the exploration worktree). If
`dynamics.simulate`'s exact signature or the joint-TPM accessor differs,
adjust to the actual API (`pyphi/dynamics.py:40`) — the test's substance is
trajectory equality plus Φ difference, not the call form. Verify the orbit
size actually is 3 for the chosen initial state; if the free run needs a
different start to reproduce it, use the demo's.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/test_estimate.py -k "edge_probability or twin or epsilon or saturates" -v`
Expected: `edge_probability` tests FAIL (`AttributeError`); the twin,
epsilon, and saturation tests may PASS already (they exercise existing
machinery) — that is fine; they are being *committed*, not newly enabled.
Confirm they pass for the right reason (run them and inspect the values).

- [ ] **Step 3: Implement `edge_probability`**

Per sample: draw `p_on` (shape `(2**n, n)`) with the shared rng; for each
input unit `a`, pair rows differing only in bit `a` (`row` vs
`row | (1 << a)` for rows with bit `a` clear) and compute
`max |p_on[row_pair_diff]|` per target unit; edge `(a, b)` fires when that
maximum exceeds `threshold`. Accumulate firing fractions across samples.
Vectorize with numpy (reshape to expose axis `a`); no per-row Python loops
over samples.

Docstring: define the statistic precisely, state that the exact-equality
`infer_cm` saturates on estimated TPMs (which is why this graded oracle
exists), and note that `threshold` is a modeling choice the caller must own.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/test_estimate.py -v`
Expected: all PASS.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/edge-probability.feature.md <<'EOF'
Added `SubstratePosterior.edge_probability`: a graded connectivity oracle
for estimated substrates (the fraction of posterior samples in which a
unit's conditional varies beyond a caller-chosen threshold along each input
axis). The exact-equality `infer_cm` saturates to fully-connected on any
continuously-estimated TPM; this is the replacement for that regime.
EOF
git add pyphi/estimate.py test/test_estimate.py changelog.d/edge-probability.feature.md
git commit -m "Add graded edge_probability oracle and commit the identifiability demonstrations as tests"
```

---

### Task 5: Serialization of the posterior objects

**Files:**
- Modify: `pyphi/serialize/schema.py`
- Modify: `pyphi/serialize/convert.py`
- Modify: `pyphi/estimate.py` (inherit `Serializable`, matching how result
  types opt in — see `pyphi/models/complex.py`)
- Test: `test/serialize/test_serialize_estimate.py` (create)

**Interfaces:**
- Produces: `CoverageReportSchema`, `SubstratePosteriorSchema`,
  `PhiPosteriorSchema` (tagged frozen Structs) + registered
  encoder/decoder pairs. numpy arrays use the existing bytes treatment
  (reuse the array encode/decode helpers the repertoire/TPM schemas use in
  `convert.py` — do not invent a second array encoding).

- [ ] **Step 1: Write the failing tests**

Create `test/serialize/test_serialize_estimate.py` (mirror
`test/serialize/test_serialize_sia.py`'s `FORMATS`/`round_trip` helpers):

```python
"""Round-trip serialization of estimation-layer objects."""

import numpy as np
import pytest

import pyphi
from pyphi import examples
from pyphi import serialize
from pyphi.estimate import estimate_substrate
from pyphi.estimate import phi_posterior

FORMATS = ["json", "msgpack"]


def round_trip(obj, fmt):
    return serialize.loads(serialize.dumps(obj, format=fmt), format=fmt)


@pytest.fixture(scope="module")
def posterior():
    traj = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 0]] * 5)
    return estimate_substrate(traj, regime="observational")


@pytest.mark.parametrize("fmt", FORMATS)
def test_substrate_posterior_round_trip(posterior, fmt):
    restored = round_trip(posterior, fmt)
    np.testing.assert_array_equal(restored.alpha_on, posterior.alpha_on)
    np.testing.assert_array_equal(restored.alpha_off, posterior.alpha_off)
    assert restored.regime == posterior.regime
    assert restored.prior == pytest.approx(posterior.prior)
    np.testing.assert_array_equal(
        restored.coverage.counts, posterior.coverage.counts
    )
    assert restored.provenance.estimator == posterior.provenance.estimator
    # A restored posterior is fully functional.
    a = restored.sample(seed=11)
    b = posterior.sample(seed=11)
    np.testing.assert_array_equal(
        np.asarray(a.factored_tpm.factor(0)), np.asarray(b.factored_tpm.factor(0))
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_phi_posterior_round_trip(posterior, fmt):
    with pyphi.config.override(progress_bars=False):
        pp = phi_posterior(posterior, (1, 0, 0), n_samples=4, seed=3)
    restored = round_trip(pp, fmt)
    np.testing.assert_array_equal(restored.samples, pp.samples)
    assert restored.complex_samples == pp.complex_samples
    assert restored.seed == pp.seed
    assert restored.regime == pp.regime
    assert restored.p_positive == pytest.approx(pp.p_positive)
    with pytest.raises(TypeError):
        float(restored)  # coercion semantics survive the round trip
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest test/serialize/test_serialize_estimate.py -v`
Expected: FAIL (`No serializer registered for SubstratePosterior`).

- [ ] **Step 3: Implement**

- Schemas: `CoverageReportSchema(counts=<array-bytes fields>, n_units: int)`;
  `SubstratePosteriorSchema(alpha_on, alpha_off, regime, prior, coverage,
  node_labels, provenance)`; `PhiPosteriorSchema(samples, complex_samples,
  state, subset, seed, regime, coverage, provenance)`. Match the existing
  array-field shape used by repertoire/TPM schemas exactly (dtype + shape +
  bytes, whatever the current helpers emit).
- One `_register_<type>()` each in `convert.py`, added to
  `_ensure_registered()` alongside the existing ones. Reuse the existing
  `NodeLabels` and `Provenance` encoders via the registry (nested encode),
  matching how composite types like the SIA do it.
- Follow the codebase's `Serializable` opt-in for the three estimate types.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest test/serialize -q && uv run pytest test/test_estimate.py -q`
Expected: all PASS, including the whole pre-existing serialize suite.

- [ ] **Step 5: Changelog fragment and commit**

```bash
cat > changelog.d/estimate-serialization.feature.md <<'EOF'
`SubstratePosterior`, `CoverageReport`, and `PhiPosterior` round-trip
through `pyphi.serialize` (JSON and msgpack), including the raw posterior
parameters and per-draw Φ samples, so estimation results can be stored and
re-analyzed without recomputation.
EOF
git add pyphi/serialize/schema.py pyphi/serialize/convert.py pyphi/estimate.py test/serialize/test_serialize_estimate.py changelog.d/estimate-serialization.feature.md
git commit -m "Serialize the estimation-layer posterior objects"
```

---

### Task 6: Acceptance test, exports, roadmap close-out, full verification

**Files:**
- Modify: `pyphi/__init__.py` (exports)
- Test: `test/test_estimate.py` (extend)
- Modify: `ROADMAP.md` (Status Dashboard row "Uncertainty pipeline
  (minimal)", line ~65)

**Interfaces:**
- Produces: `pyphi.estimate_substrate`, `pyphi.phi_posterior` lifted to the
  package root (match the `from .analyze import analyze as analyze` +
  `__all__` pattern at `pyphi/__init__.py:92,153`); the types stay
  addressable as `pyphi.estimate.SubstratePosterior` etc.

- [ ] **Step 1: Exports**

Add the two functions to `pyphi/__init__.py` following the existing
import-and-`__all__` pattern. Test (append to `test/test_estimate.py`):

```python
def test_top_level_exports():
    assert pyphi.estimate_substrate is estimate_substrate
    assert pyphi.phi_posterior is phi_posterior
```

- [ ] **Step 2: Acceptance test — the spec's grid3 demonstration, end to end**

Append to `test/test_estimate.py` (mark `slow` — this is the one
larger-sample run):

```python
@pytest.mark.slow
def test_grid3_mixture_acceptance(grid3_posterior):
    """From five perturbational samples per state, the Φ posterior over
    grid3 at (0,0,0) is a genuine mixture: substantial mass on
    reducibility, a conditional density that brackets the true Φ, and a
    contested complex identity concentrated on the symmetric pair."""
    with pyphi.config.override(progress_bars=False):
        pp = phi_posterior(grid3_posterior, (0, 0, 0), n_samples=150, seed=2026)
    # The reference run (300 draws) gave P(phi > 0) = 0.20; a different
    # counting stream shifts this, so assert the band, not the point.
    assert 0.05 < pp.p_positive < 0.5
    lo, hi = pp.conditional_quantiles([0.025, 0.975])
    assert lo < 0.024666 < hi  # brackets the true phi
    identity = pp.complex_identity
    assert identity[(0,)] + identity.get((2,), 0.0) > 0.5
```

Run once in isolation
(`uv run pytest test/test_estimate.py -k acceptance -v`), record the
observed `p_positive` and conditional interval as a comment in the test for
future readers, and confirm the runtime is acceptable for the `slow` lane
(~150 draws × (SIA + complex search) at n = 3; the spec's 300-draw demo ran
in well under a minute).

- [ ] **Step 3: Update the ROADMAP dashboard row**

Change the "Uncertainty pipeline (minimal)" row (`ROADMAP.md` line ~65)
from `⬜ open` to `✅ landed`, following the format of neighboring landed
rows and describing what shipped: `pyphi.estimate` with
`estimate_substrate` (counts model, Jeffreys default, required `regime`
assertion stamped into a structured `Provenance.estimator` slot),
`SubstratePosterior.sample()` reusing the whole compute stack,
`phi_posterior` returning the mixture (`p_positive` + conditional quantiles
+ raw samples + complex-identity categorical, bare-float coercion refused),
first-class `CoverageReport`, `edge_probability` replacing the saturating
`infer_cm` for estimated TPMs, and the three identifiability
demonstrations committed as reproducible tests. Note the explicit
non-builds (distribution-valued Φ through result types, GLM/Ising, sweep
ensemble axis) as deliberately out of scope.

(The source spec lives on the `worktree-tpm-uncertainty-exploration`
branch, not in this worktree; do not attempt to edit or commit it.)

- [ ] **Step 4: Full test suite (no path argument — includes the doctest sweep)**

Run: `uv run pytest -q` (background it and run the fast lanes in the
foreground per the project's parallel-testing convention if wall time is a
concern; the bare invocation must complete green at least once).
Expected: all pass. Pay particular attention to:

- `test/serialize` — the provenance schema gained a trailing field; every
  pre-existing payload must still decode.
- doctests in `pyphi/estimate.py` and `pyphi/provenance.py` — any examples
  written in docstrings run under the sweep.

If an unrelated-looking test fails, diagnose before touching anything —
other sessions may have concurrent working-tree changes; only fix failures
traceable to this plan's commits.

- [ ] **Step 5: Pre-commit hooks over the changed files**

Run: `uv run pre-commit run --files $(git diff --name-only $(git merge-base HEAD <base-branch>) | tr '\n' ' ')` — substitute the branch this worktree was created from.
Expected: all hooks pass (ruff, pyright, file checks).

- [ ] **Step 6: Commit the close-out**

```bash
git add pyphi/__init__.py test/test_estimate.py ROADMAP.md
git commit -m "Export the estimation surface; add the grid3 mixture acceptance test"
```

---

## Self-review notes

- **Spec §8 coverage:** item 1 (`estimate_substrate`, required regime,
  Jeffreys default, counts only) — Task 1; item 2 (`.sample()` → ordinary
  `Substrate`) — Task 1; item 3 (`phi_posterior` with `p_positive`,
  conditional quantiles, samples, complex-identity categorical, ~thin
  driver, not a sweep axis) — Task 3; item 4 (coverage report first-class,
  refusal of a lone scalar under partial coverage) — Tasks 1 and 3 (the
  refusal is structural: no float coercion exists at all, and the partial-
  coverage case names the unconstrained states in the error).
- **Settled semantics:** no-bare-float (Task 3 `__float__`, retested after
  serialization round-trip in Task 5); constructor-time regime assertion
  (Task 1, no default) stamped into provenance (Task 2); Jeffreys default
  (Task 1). None of these is relitigated anywhere in the plan.
- **Non-builds honored:** no changes to any existing result type, display
  path, or comparison operator; no `DistanceResult.aux` smuggling; no sweep
  axis; counts model only (`model="glm"` raises).
- **Wave-0 confirmations committed:** the spec's three inline checks are now
  reproducible tests (Task 4): twin non-identifiability (fully
  deterministic), `infer_cm` saturation (documents the defect), ε-boundary
  (with the spec's measured values as regression anchors).
- **Endianness guard:** the ground-truth round-trip test uses the
  *asymmetric* OR/AND/XOR substrate, so a reversed axis or wrong row order
  fails loudly (symmetric fixtures cannot catch this class of bug).
- **Seeding discipline:** every randomized entry point (`sample`,
  `phi_posterior`, `edge_probability`) requires an explicit seed or
  Generator, uses an isolated `default_rng`, and the seed is stored on the
  result and in provenance; raw per-draw Φ samples and complex identities
  are retained on the object and serialized (Task 5), so analyses can be
  redone without recomputation.
- **Known caveats, documented not fixed:** estimated substrates are fully
  connected via the constructor's `infer_cm` (honest — dependence is not
  excluded; `edge_probability` is the graded view); within-step dependence
  between units cannot be represented by the factored form and is silently
  absorbed (spec §3.3/§9 — noted in the module docstring); the acceptance
  test asserts bands, not the demo's exact 0.20, because the counting
  stream differs from `demo.py`'s.
- **Riskiest assumptions:** (1) `Substrate(tpm=convert.to_multidimensional(pon))`
  constructor form — validated by the spec's round trip on the sibling
  branch, but verify the kwarg name on this branch in Task 1; (2) the twin
  test's orbit-size-3 claim depends on the initial state — the test asserts
  it explicitly, and the demo's raw NPZ is available for cross-checking;
  (3) the numpy-array bytes helpers in `serialize/convert.py` — names must
  be read from the file, not assumed.
