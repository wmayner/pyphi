# P12a — Factored TPM Foundation

**Status:** Draft (awaiting review)
**Date:** 2026-05-22
**Branch baseline:** `2.0` at `114eca20`
**Related:** ROADMAP item 6 (P12 — Non-binary units); PR #105 (Implicit TPMs — design reference, not extension target); legacy PR #42 / #48 (multi-valued elements on the dead `nonbinary` branch — archaeology only)

---

## 1. Background

PyPhi today stores a substrate's transition probability matrix as a single joint conditional distribution `P(s_{t+1} | s_t)` of shape `(a_1, …, a_N, N)` (or `(a_1, …, a_N, N, a_i)` for k-ary in the legacy convention) — an `ExplicitTPM` instance. Every internal consumer materializes and operates on that joint.

This representation has two costs that scale poorly:

1. **Memory** grows as `∏(alphabet_sizes)`. For binary at N=10 that's 1024 rows; for k=3 at N=10 it's 59,049 rows; for k=4 at N=10 it's >1M rows. Multi-valued IIT analysis becomes infeasible quickly.
2. **Math obscurity.** The joint conditional is mathematically equivalent to a product of per-node conditional marginals under IIT's standing assumption that nodes update independently given the joint past. The joint form hides that factorization; downstream code that conditions / marginalizes does so by joint-shape ndarray gymnastics rather than per-node operations.

P12a introduces a `FactoredTPM` representation as the canonical substrate storage and cuts every internal hot path over to consume it directly. Multi-valued substrates (P12b, separate spec) drop in cleanly on top of this foundation because the math is already alphabet-generic. The joint form survives as a derived view for boundary use (serialization, legacy fixture comparison, user-facing display).

---

## 2. Scope

### 2.1 In scope

- **`FactoredTPM` class** in `pyphi/core/tpm/factored.py`. N per-node factors with a swappable storage backend (default ndarray; xarray as opt-in via `pyphi[xarray]` extra). Implements the `TPM` Protocol.
- **`Substrate` canonical storage cutover.** The substrate's TPM field holds `FactoredTPM`. Existing `Substrate(tpm=joint_array, …)` keeps working — it auto-converts. New `marginals=[…]` keyword and `Substrate.from_factored(…)` factory accepted.
- **`Unit.alphabet_size: int = 2`** field added. P12a defaults to 2 everywhere; P12b will populate from user input.
- **Hot-path cutover.** Marginalization (`cause_tpm`, `effect_tpm`), repertoire reconstruction (`cause_repertoire`, `effect_repertoire`, `unconstrained_repertoire`), partition iteration, MICE evaluation, and validation all consume `FactoredTPM` directly. The joint becomes a derivation, not a state.
- **`Substrate.joint_tpm()`** method returns the joint on demand, no cache.
- **Storage-backend benchmark** in `benchmarks/factored_tpm_backend.py`. Measures xarray vs. ndarray on the existing perf-budget fixtures. The default backend is set from the measured result before the project lands.
- **Internals alphabet-generic.** Property tests (Hypothesis) exercise k ∈ {2, 3, 4, 5} math on synthetic substrates. User surface stays binary.
- **`ExplicitTPM` → `JointTPM` rename.** The legacy joint storage class is renamed for symmetry with `FactoredTPM`. Mechanical search-replace across pyphi/ and tests/; one focused commit.

### 2.2 Out of scope (P12b's deliverable, separate spec)

- User-facing multi-valued substrate construction.
- k>2 golden fixtures.
- Examples, tutorials, docs for multi-valued.
- Multi-valued IIT analysis surface (anything in `compute/`, `new_big_phi/`, formalism paths that needs a non-binary public API).

### 2.3 Explicit non-goals

- No changes to formalism dispatch (`config.formalism.iit.version`).
- No changes to partition schemes.
- No new metric or distance measures.
- No relations / phi-fold work.
- No benchmark-suite rewrite (ROADMAP item P11.8 Tier 2 — separate project).
- No back-compat shim for the `ExplicitTPM` → `JointTPM` rename (2.0 unpushed; clean rename).
- No `pyphi.config` flag for "preserve old joint-storage behavior". Cutover is unconditional.

### 2.4 Success criteria

- All current binary golden fixtures pass byte-identically (numerical equivalence at `config.numerics.precision` required; byte-identical NPZ outputs preferred).
- All current binary tests pass with current API; no `DeprecationWarning` on existing scripts.
- Perf-budget gate from P11.8 Tier 1 stays green (no >4× regression on the 5 hot-path fixtures).
- New property tests pass for k ∈ {3, 4, 5} synthetic substrates exercising marginalization, conditioning, and repertoire reconstruction.
- Pyright clean against the existing 0 errors / 1 baseline-warning state.
- Ruff clean across all modified files.
- Benchmark report committed under `benchmarks/results/` showing xarray-vs-ndarray on the hot path; the chosen default backend is justified by the report.

---

## 3. Architecture

### 3.1 The shape of the change

Today, a substrate's source of truth is a joint conditional TPM stored as one ndarray. After P12a, the source of truth is a `FactoredTPM`: a sequence of N per-node factors, where factor `i` has shape `(a_1, …, a_N, a_i)` (the first N dims are the joint input state at `t`; the last is node `i`'s state at `t+1`). Inputs that aren't connected to node `i` get singleton dims (size 1) — those singletons are semantically load-bearing and never squeezed away. The joint factors as

```
P(s_{t+1} | s_t) = ∏_{i=1..N} P(s_{i,t+1} | s_t)
```

under IIT's conditional-independence assumption. Each factor is a (conditional) marginal of the joint conditional; collectively the factors are the joint in factored form.

### 3.2 Objects (dependency order)

1. **`FactoredTPM`** (`pyphi/core/tpm/factored.py`, new) — the canonical storage. Holds an internal `_StorageBackend` and an N-tuple of factor arrays. Implements the `TPM` Protocol. Knows `alphabet_sizes: tuple[int, ...]` per node. Operations: indexing, conditioning, marginalizing, factor extraction, joint reconstruction (slow path).

2. **`_StorageBackend`** (`pyphi/core/tpm/_factored_backends.py`, internal). Thin abstraction with two implementations: `_NdarrayBackend` (default) and `_XarrayBackend` (opt-in, decided by benchmark). Not part of the public API; `FactoredTPM` is the public surface.

3. **`Substrate`** (`pyphi/substrate.py`, modified). The TPM field changes type from `JointTPM` to `FactoredTPM`. Existing `Substrate(tpm=joint_array, …)` constructor auto-converts to `FactoredTPM` via `FactoredTPM.from_joint(...)`. New `marginals=[per_node_factors]` keyword accepted alongside `tpm=joint` (mutually exclusive). Factory `Substrate.from_factored(factored, ...)` for explicit construction.

4. **`Unit`** (`pyphi/core/unit.py`, extended). Adds `alphabet_size: int = 2`. The default makes P12a a no-op for binary; P12b populates from user input.

5. **`TPM` Protocol** (`pyphi/core/tpm/base.py`, extended). Gains `alphabet_sizes` property. Loses `squeeze` (had no coherent meaning on the factored representation — see §5.3).

6. **`pyphi/core/tpm/marginalization.py`** (modified). `cause_tpm` and `effect_tpm` accept `TPM` Protocol, dispatch on whether the argument is a `FactoredTPM` (fast path: marginalize per factor) or a `JointTPM` (legacy: existing code).

7. **`JointTPM`** (`pyphi/tpm.py` and `pyphi/core/tpm/joint.py`, renamed from `ExplicitTPM`). Stays in the codebase as the boundary value type — output of `FactoredTPM.to_joint()`, format consumed by legacy fixture comparison and serialization. Not the canonical substrate storage.

### 3.3 Information flow at use

Hot path (steady state — no joint materialized):

```
User: Substrate(tpm=joint_array, cm=...)
  ↓
constructor: FactoredTPM.from_joint(joint) → FactoredTPM
  ↓
Substrate stores FactoredTPM
  ↓
Hot path consumer (e.g., effect_tpm): reads factored.factor(i) directly
  ↓
No joint materialized in steady state
```

Legacy path (boundary use):

```
Substrate.joint_tpm()
  ↓
factored_tpm.to_joint() → np.ndarray, allocated fresh each call
  ↓
Caller uses it; GC reclaims when caller drops it
```

### 3.4 Where the existing P7 scaffold fits

`pyphi/core/tpm/{base.py, explicit.py, marginalization.py}` already exist on `2.0` with anticipatory markers (`# P12 lifts that assumption`, `# P12 adds alphabet_size`). P12a removes those markers (they violate the no-planning-artifacts-in-code constraint), lifts the assumption they anticipated, adds `factored.py` next to them, and renames `explicit.py` → `joint.py`.

---

## 4. Components & file layout

### 4.1 Files

**New:**

```
pyphi/core/tpm/factored.py              # FactoredTPM
pyphi/core/tpm/_factored_backends.py    # _NdarrayBackend + _XarrayBackend
benchmarks/factored_tpm_backend.py      # xarray-vs-ndarray micro-benchmark
benchmarks/results/factored-tpm-backend-2026-05-22.md   # decision artifact
benchmarks/results/factored-tpm-backend-2026-05-22.json # raw timings
test/test_factored_tpm.py               # unit tests
test/test_factored_tpm_kary.py          # Hypothesis property tests
test/test_marginalization.py            # cause_tpm/effect_tpm against FactoredTPM
changelog.d/factored-tpm.feature.md
changelog.d/rename-explicit-tpm-to-joint-tpm.change.md
```

**Modified:**

```
pyphi/core/tpm/base.py                  # add alphabet_sizes; drop squeeze
pyphi/core/tpm/__init__.py              # export FactoredTPM, JointTPM
pyphi/core/tpm/marginalization.py       # Protocol dispatch; FactoredTPM fast path
pyphi/core/unit.py                      # add alphabet_size: int = 2
pyphi/substrate.py                      # canonical storage → FactoredTPM
pyphi/repertoire.py                     # consume FactoredTPM directly
pyphi/subsystem.py                      # joint reads → factored reads
pyphi/validate.py                       # factored_tpm validator; rename tpm → joint_tpm
pyphi/__init__.py                       # re-export FactoredTPM, JointTPM
pyphi/tpm.py                            # ExplicitTPM → JointTPM rename
pyproject.toml                          # add `xarray` to optional `[xarray]` extra
```

**Renamed (git mv):**

```
pyphi/core/tpm/explicit.py → pyphi/core/tpm/joint.py
```

### 4.2 `FactoredTPM` — public surface

```python
class FactoredTPM:
    """Per-node-factored conditional TPM.

    Represents the joint conditional ``P(s_{t+1} | s_t)`` as a product of
    N per-node conditional marginals ``P(s_{i,t+1} | s_t)``. The joint is
    the product of the factors under conditional independence (IIT's
    standing assumption that nodes update independently given the joint
    past).

    Factor ``i`` has shape ``(a_1, …, a_N, a_i)`` where ``a_j`` is the
    alphabet size of node ``j``. Input dims for non-input nodes are size 1
    and are semantically load-bearing (they encode the connectivity
    structure); they are never squeezed away.
    """

    # construction
    def __init__(
        self,
        factors: Sequence[ArrayLike],
        alphabet_sizes: Sequence[int] | None = None,
        backend: Literal["ndarray", "xarray"] | None = None,
    ) -> None: ...

    @classmethod
    def from_joint(
        cls,
        joint: ArrayLike,
        /,
        alphabet_sizes: Sequence[int] | None = None,
    ) -> "FactoredTPM": ...

    # TPM Protocol
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def alphabet_sizes(self) -> tuple[int, ...]: ...
    def condition(self, fixed: Mapping[int, int]) -> "FactoredTPM": ...
    def to_array(self) -> NDArray[np.float64]: ...  # alias for to_joint()

    # factor surface
    @property
    def factors(self) -> tuple[NDArray, ...]: ...
    def factor(self, i: int) -> NDArray: ...
    def condition_factor(self, i: int, fixed: Mapping[int, int]) -> NDArray: ...

    # slow path
    def to_joint(self) -> NDArray[np.float64]: ...

    # value-type machinery
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def __reduce__(self) -> ...: ...
```

### 4.3 `_StorageBackend` — internal

```python
class _StorageBackend(Protocol):
    """Internal storage abstraction. Not part of the public API."""

    def get_factor(self, i: int) -> NDArray[np.float64]: ...
    def n_factors(self) -> int: ...
    def alphabet_sizes(self) -> tuple[int, ...]: ...
    def select(self, i: int, fixed: Mapping[int, int]) -> NDArray: ...


class _NdarrayBackend:
    """Tuple of ndarrays. Positional indexing.
    Name-based lookup goes through FactoredTPM's node-label mapping."""

class _XarrayBackend:
    """Tuple of xr.DataArray with named input dims.
    Loaded lazily so xarray is an optional dep at import time.
    Raises ImportError at instantiation if xarray is not installed."""
```

### 4.4 `TPM` Protocol — final shape

```python
@runtime_checkable
class TPM(Protocol):
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def n_nodes(self) -> int: ...
    @property
    def alphabet_sizes(self) -> tuple[int, ...]: ...
    def condition(self, fixed: Mapping[int, int]) -> "TPM": ...
    def to_array(self) -> NDArray[np.float64]: ...
```

`squeeze` is removed from the Protocol; it lives only on `JointTPM` as a numpy-cleanup affordance.

### 4.5 `Unit` — extended

```python
@dataclass(frozen=True, slots=True)
class Unit:
    index: int
    label: str
    alphabet_size: int = 2
```

### 4.6 `Substrate` constructor signature

```python
def __init__(
    self,
    *,
    tpm: ArrayLike | None = None,           # joint form (auto-converts)
    marginals: Sequence[ArrayLike] | None = None,  # factored form (direct)
    cm: ArrayLike | None = None,
    node_labels: Sequence[str] | None = None,
    state_space: ... | None = None,
    ...
) -> None: ...
```

Mutually exclusive: passing both `tpm=` and `marginals=` raises `ValueError("pass tpm= or marginals=, not both")`.

---

## 5. Validation, error handling, edge cases

### 5.1 Construction-time validation

`validate.factored_tpm(factored, *, strict=True)`:

- **Shape consistency.** Every factor's input dims match either `alphabet_sizes[j]` for an input node `j` or `1` for a non-input. Mismatch raises `pyphi.exceptions.InvalidTPM` with the offending factor index and shape.
- **Probability axiom.** Each factor sums to 1 along its output (last) dim within `config.numerics.precision`. Tolerance `10**(-config.numerics.precision)`; floor `1e-15`. Failure raises `InvalidTPM` with the worst-deviation location.
- **Alphabet validity.** `alphabet_sizes[i] >= 2` for every node. Single-state nodes are invalid substrates.
- **CM compatibility.** If `cm` is supplied at substrate construction, the substrate constructor (not `validate.factored_tpm`) checks the cm-inferred-from-factors matches. Mismatch raises `pyphi.exceptions.InvalidNetwork` listing disagreeing edges. Validation runs once at construction; downstream code may assume consistency.

`validate.factored_tpm` runs from `Substrate.__init__` after the FactoredTPM is constructed. The existing `validate.tpm` is renamed to `validate.joint_tpm` and retained for boundary use.

### 5.2 Run-time guards

When `Substrate.__init__` accepts `tpm=joint_array`: joint passes `validate.joint_tpm` (existing path), then `FactoredTPM.from_joint(...)` converts. If conversion produces a factored form that fails `validate.factored_tpm` (which should not happen for a valid joint), assert internally and surface as `pyphi.exceptions.InternalConversionError` — saves debugging time later if the conversion ever drifts.

When `Substrate.__init__` accepts `marginals=...`: no joint conversion happens; only `validate.factored_tpm` runs.

### 5.3 Numerical edge cases

- **Deterministic factors** (entries exactly 0 or 1). Allowed. Conditioning on a state assigned probability 0 in some factor produces a degenerate form; `validate.factored_tpm` rejects it on the sums-to-1 check. Documented as a known limitation; raised on construction.
- **Unreachable states.** Handled at the repertoire layer (existing logic in `compute/`), not at the TPM layer.
- **Joint → factored → joint drift.** Property test enforces agreement within `precision` over ≥100 random TPMs.
- **Singleton dims for non-input nodes.** Carry semantic identity. Operations use `keepdims=True` / explicit `None` indexing consistently. `FactoredTPM` has no `squeeze` method; there is no path that silently drops these dims.

### 5.4 Exception surface

- `pyphi.exceptions.InvalidTPM` — shape / probability errors (existing).
- `pyphi.exceptions.InvalidNetwork` — CM mismatch (existing).
- `ValueError` — mutually-exclusive constructor args (stdlib).
- `pyphi.exceptions.InternalConversionError` — joint→factored conversion produced an invalid factored TPM (new; internal-only; logged with offending shape and a hint to file a bug).
- `ImportError` — `_XarrayBackend` requested but xarray not installed (stdlib; raised at backend instantiation, not at FactoredTPM import).

### 5.5 Logging

- `DEBUG`: factor shapes and alphabet sizes at FactoredTPM construction.
- `INFO`: storage backend selection on first FactoredTPM construction per process (module-level flag prevents spam).
- No `WARNING` in P12a. xarray-unavailable conditions raise `ImportError`; they do not warn-and-fallback.

---

## 6. Data flow & key algorithms

### 6.1 Construction from joint

`FactoredTPM.from_joint(joint, alphabet_sizes)`:

For binary inputs with the legacy pyphi convention `(2, …, 2, N)`, factor `i` is built by expanding the output along an explicit alphabet dim: `joint[..., i:i+1]` reshaped to `(2, …, 2, 2)` with the trailing alphabet dim made explicit. For k-ary inputs with the convention `(a_1, …, a_N, N, a_i)` (existing pyphi convention), factor `i` is `joint[..., i, :]` with shape `(a_1, …, a_N, a_i)`.

Singleton input dims are detected by checking which input dims are uniform along node `i`'s output (`np.all(factor == factor[:, 0:1, …])` along each input axis). Collapsing those gives the inferred-CM behavior PR #105 implemented. We keep the inference but expose it as `infer_cm_from_factors(factored) -> NDArray[bool]` so callers can pass `cm=None` (derived) or `cm=…` (validated against inferred).

### 6.2 Conditioning

```python
def _condition_one_factor(factor_i, fixed):
    idx = [slice(None)] * factor_i.ndim
    for j, state_j in fixed.items():
        idx[j] = state_j
    out = factor_i[tuple(idx)]
    # restore rank: collapsed dims become singletons
    for j in sorted(fixed):
        out = np.expand_dims(out, axis=j)
    return out
```

`FactoredTPM.condition(fixed)` maps `_condition_one_factor` across factors. O(N) per call; no joint materialization.

### 6.3 Effect repertoire

```python
def effect_repertoire(mechanism, purview, factored: FactoredTPM, state) -> NDArray:
    # 1. condition each factor on the mechanism's current state
    conditioned = factored.condition({i: state[i] for i in mechanism})
    # 2. for each node in purview, marginalize non-input background uniformly
    purview_factors = [
        _marginalize_inputs(conditioned.factor(i), background_indices)
        for i in purview
    ]
    # 3. outer-product over purview nodes (independent given inputs)
    return _outer_product(purview_factors)
```

`_marginalize_inputs` sums each input background dim with uniform weight `1 / a_j`. The outer-product step is `O(∏ purview alphabet sizes)` — same as today for binary, naturally k-aware. The biggest behavioral change: today's `effect_repertoire` materializes the joint TPM and then conditions; the new version operates entirely on factors.

### 6.4 Cause repertoire

Mirrors §6.3 with the dual operation. Today's `cause_tpm` (`backward_tpm`) inverts the joint conditional via Bayes; the factored form inverts each factor and then product-combines. Mathematically identical for binary. The factorization-survives-Bayes property is verified by Hypothesis (`cause_from_factored(...) ≈ cause_from_joint(joint(factored), ...)` for random k-ary TPMs).

### 6.5 Joint reconstruction (slow path)

```python
def to_joint(self) -> NDArray:
    shape = self.alphabet_sizes + (self.n_nodes,)
    out = np.empty(shape)
    for i, factor in enumerate(self.factors):
        out[..., i, :] = np.broadcast_to(
            factor, shape[:-1] + (factor.shape[-1],)
        )
    return out
```

Used only at boundaries: serialization, legacy fixture comparison, `Substrate.joint_tpm()`. For multi-valued this is exactly the cost the factored form is designed to avoid; calls to `to_joint()` are flagged as slow-path in docstrings.

---

## 7. Testing strategy

### 7.1 Unit tests (`test/test_factored_tpm.py`)

- `from_joint` round-trip for binary networks of size 2, 3, 4, 5 (deterministic + probabilistic).
- `condition` on partial states; equality with conditioning the joint.
- `factor(i)` and `condition_factor(i, fixed)` correctness.
- `to_joint` round-trip stability.
- `__eq__` content-based across backend types.
- `__repr__` smoke test.
- Pickling round-trip.
- Each `InvalidTPM` raise path covered.

### 7.2 K-ary property tests (`test/test_factored_tpm_kary.py`)

Hypothesis-driven. The only tests exercising k>2 math in P12a (user surface is binary).

Strategies:
- `alphabets(n_nodes)` — `tuple[int, ...]` of length `n_nodes` with entries in `{2, 3, 4, 5}`.
- `factored_tpms(alphabets, n_inputs_per_node)` — valid FactoredTPMs.

Properties:
- `from_joint(factored.to_joint()) == factored` (round-trip stability).
- `factored.condition(fixed).to_joint() ≈ condition_joint(factored.to_joint(), fixed)` (conditioning commutes with reconstruction).
- `cause_tpm(factored, state, indices).to_joint() ≈ cause_tpm(JointTPM(factored.to_joint()), state, indices).to_array()`.
- Effect-repertoire independence: for non-overlapping purviews `Z_1`, `Z_2`, `effect_repertoire(M, Z_1∪Z_2) == ⊗(effect_repertoire(M, Z_1), effect_repertoire(M, Z_2))`.

Hypothesis settings: `max_examples=50` per property in fast lane; `max_examples=500` in slow lane via `@pytest.mark.slow`.

### 7.3 Marginalization tests (`test/test_marginalization.py`)

`cause_tpm` and `effect_tpm` against `FactoredTPM`. Dispatch correctness; math agreement with `JointTPM` path for binary inputs.

### 7.4 Golden regression

Existing 17 fixtures + 1 canonical preset coverage guardrail must pass byte-identically. If any drift past `config.numerics.precision`, that's a P12a bug. No new fixtures in P12a.

### 7.5 Perf budget

The 5 hot-path fixtures (basic / xor / rule110 / grid3 / micro_s) must stay within `max(3.0, 4×median)` floors after cutover. Perf-budget gate is a hard stop — if any fixture regresses past its floor, P12a doesn't land. No "accept the regression" branch.

### 7.6 Storage-backend benchmark (`benchmarks/factored_tpm_backend.py`)

xarray vs. ndarray on:

- `condition` over random partial states
- `effect_repertoire` for fixed mechanism/purview pairs
- `from_joint` and `to_joint` round-trips
- Sizes: 4, 6, 8, 10 nodes; alphabet binary; one k=3 size for P12b preview

Reports median + p95 wall time per operation per backend per size. Output committed to `benchmarks/results/factored-tpm-backend-2026-05-22.md` (markdown table) plus raw timings in `.json` (raw data alongside aggregates).

The benchmark drives `_FACTORED_TPM_DEFAULT_BACKEND`. Decision rule: if xarray is within ≤2× of ndarray on every measured operation and size, set xarray as default. Otherwise ndarray.

### 7.7 Pyright + ruff

Pyright passes with the existing baseline (0 errors / 1 baseline warning). No new pyright errors introduced. Ruff clean across all modified files.

### 7.8 Not tested in P12a (P12b's domain)

- k>2 golden fixtures
- k>2 examples / tutorial notebooks
- k>2 perf budget fixtures
- xarray-backend perf budgets (only the binary benchmark drives default; xarray-as-default inherits perf budgets in P12b if it ships)

---

## 8. Migration & cutover plan

### 8.1 Commit sequence (10-12 commits)

Each commit independently passes the fast lane + pyright + ruff. Mid-sequence states have some overlap (e.g., legacy repertoire functions kept under `_legacy_*` for one commit before deletion), but no broken intermediate states.

1. **Extend the TPM Protocol and Unit.** `alphabet_sizes` added to Protocol; `squeeze` removed; `Unit.alphabet_size: int = 2` added; `ExplicitTPM` (still under that name) gains `alphabet_sizes` property returning `(2,) * n_nodes`.

2. **Add `FactoredTPM` skeleton with ndarray-only backend.** New `factored.py` + `_factored_backends.py`. Class compiles, validates, no consumers yet.

3. **Add `FactoredTPM.from_joint` and `to_joint` round-trip + binary property test.**

4. **Add xarray backend, optional dep wiring, lazy import.** `pyphi[xarray]` extra in `pyproject.toml`; `_XarrayBackend` raises `ImportError` if xarray unavailable. CI matrix: one job with xarray, one without.

5. **Add the storage-backend benchmark and run it.** Result file committed.

6. **Set `_FACTORED_TPM_DEFAULT_BACKEND` from the benchmark.**

7. **Rename `ExplicitTPM` → `JointTPM`.** Mechanical search-replace; `git mv pyphi/core/tpm/explicit.py pyphi/core/tpm/joint.py`; top-level re-exports updated; changelog fragment. ~30 importers/consumers touched.

8. **Add k-ary property tests** (`test_factored_tpm_kary.py`).

9. **Cut over `pyphi/repertoire.py` to consume FactoredTPM.** `cause_repertoire`, `effect_repertoire`, `unconstrained_repertoire` rewritten against factored form. Joint-form versions retained under `_legacy_*` for one commit's overlap.

10. **Cut over `pyphi/subsystem.py` and `pyphi/core/tpm/marginalization.py`.** `_compute_cause_repertoire`, `_compute_effect_repertoire`, internal MICE helpers updated. `marginalization.cause_tpm` / `effect_tpm` dispatch on Protocol.

11. **Substrate canonical storage cutover.** `pyphi/substrate.py` TPM field changes to `FactoredTPM`. Constructor auto-converts joint inputs. Add `marginals=` keyword and `joint_tpm()` method. Delete the `_legacy_*` repertoire functions from step 9. Update `pyphi/validate.py` with the `factored_tpm` validator. **Largest commit by diff.**

12. **Changelog + scaffold-marker cleanup.** `changelog.d/factored-tpm.feature.md`. Remove `# P12 lifts that assumption` / `# P12 adds alphabet_size` markers from `pyphi/core/tpm/base.py` and `pyphi/core/unit.py`.

### 8.2 What existing tests change

- **Goldens.** Zero changes expected. Drift > `precision` is a bug to fix.
- **Perf budget tests.** Zero changes expected. Any regression past floor is a hard stop.
- **Substrate construction via `Substrate(tpm=joint_array, ...)`.** Zero changes — auto-conversion preserves the surface.
- **Tests reading `substrate.tpm` and expecting joint shape.** ~20-30 sites in `test/test_subsystem.py`, `test/test_tpm.py`, `test/test_network.py`, possibly `test/test_macro_*.py`. Migration: `substrate.tpm` → `substrate.joint_tpm()` (if reading joint structure) or `substrate.factored_tpm.factors[i]` (if modernizing to read factored). Plan enumerates these via grep.
- **Mocked TPMs.** Construct via `FactoredTPM.from_joint(...)` preserving the test's joint-form input.

### 8.3 What user-facing scripts change

- Binary scripts using `pyphi.Substrate(tpm=joint, cm=...)`: zero changes.
- Scripts doing `substrate.tpm.<joint_specific_op>()`: need `.joint_tpm()`. Documented in the changelog.
- `docs/examples/*.ipynb` references to ExplicitTPM updated to JointTPM (mechanical). `substrate.tpm` reads expecting joint replaced with `substrate.joint_tpm()`.
- `IIT_4.0_demo.ipynb`: re-rendered after P12a; output cells regenerated to capture repr changes.

### 8.4 Explicit non-shims

- No `ExplicitTPM = JointTPM` alias.
- No `Substrate.tpm` ambiguous-type property.
- No `DeprecationWarning` on the joint-form constructor (the `tpm=` keyword stays as primary; auto-conversion is a coercion, not a deprecated path).
- No `pyphi.config.use_factored_tpm = False` flag.

The only soft-landing surface is the constructor auto-conversion — a useful coercion, not a compat shim.

### 8.5 Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Cause-repertoire factoring diverges from joint-form math for binary | Med-low | Critical (silent wrongness) | Hypothesis property test commit 3; goldens commit 11 |
| Hot path slower than joint for binary | Medium | High (blocks landing under perf gate) | Benchmark commit 5; perf gate hard-stop |
| xarray backend > 2× overhead | Medium | Low (ndarray default still ships) | Benchmark decides; xarray opt-in if not default |
| Substrate refactor breaks macro/actual paths | Medium | High | Commit 11 includes `test_macro_subsystem.py` + `test_actual.py -m "not slow"` smoke |
| Joint → factored → joint drift exceeds `precision` | Low | Critical | Hypothesis property tests across k ∈ {2,3,4,5} with adversarial examples |
| `Substrate.joint_tpm()` callers proliferate | Low | Medium | Code review for any new `.joint_tpm()` call site; comment in Substrate class docstring |
| xarray optional dep complicates CI matrix | Low | Low | One CI job with xarray, one without; ~10 min added to CI |

### 8.6 Out-of-scope migration items

- IIT 3.0 paths using joint TPM through legacy interfaces — covered by cutover, no special treatment.
- The 32-node numpy limit — unchanged; ROADMAP's "easier with named dims" claim is not a P12a deliverable.
- The `_backward_tpm` side effect in `System.__init__` — already handled by existing `core/tpm/marginalization.py` free functions; P12a makes them Protocol-aware.
- `MacroSubsystem`'s `SystemAttrs.apply()` mutation pattern — out of scope. Macro paths use `substrate.joint_tpm()` for their existing logic; the macro refactor is a separate project.

---

## 9. Open subdecisions deferred to P12b

These belong to the multi-valued spec, not this one. Recorded here so the P12a→P12b transition is friction-free.

- **User-facing constructor for non-binary alphabets.** Likely `Substrate(marginals=..., alphabet_sizes=..., ...)` or `Substrate.from_factored(..., alphabet_sizes=...)` extended. Decision: when P12b is brainstormed.
- **State-space labeling.** PR #105 added `state_space=...` for naming states (e.g., `("OFF", "ON")` per node). Whether P12b adopts that pattern, adopts xarray's coord vocabulary, or invents something else: P12b's call.
- **Per-node alphabets vs. uniform-alphabet shortcut.** Whether `Substrate(alphabet=3, ...)` is a sugar for `alphabet_sizes=(3,) * n_nodes`: P12b's call.
- **k>2 golden fixtures.** Which canonical examples (Albantakis et al. 2023 paper figures? specific test cases?) become k-ary goldens: P12b's call.
- **Multi-valued examples and tutorial notebook.** P12b ships at minimum one worked example demonstrating non-binary IIT analysis.

---

## 10. References

- ROADMAP item 6 (P12 — Non-binary units): the parent project framing.
- ROADMAP §"Target Architecture" lines 130-225: `core/tpm/` layout, xarray-vs-ndarray caveat, alphabet_size on Unit.
- ROADMAP §"P11.8 Tier 1" (`cad8a967`): the perf-budget gate this project must not regress.
- PR #105 ("Initial support for Implicit TPMs", Isaac David, OPEN since 2023-03-26): design reference for the factored representation, the `state_space` parameter, and the xarray-style indexing API. Not a rebase target — the branch is 21 months stale against 2.0 and conflicts heavily.
- PR #42 ("Multi-valued elements", merged 2020-11-12 into the dead `nonbinary` branch): archaeology for multi-valued semantics. Will inform P12b, not P12a.
- PR #48 ("Nonbinary", merged 2021-05-14): same.
- Albantakis et al. 2023, *PLoS Comp Bio* 19(10): e1011465 — IIT 4.0 paper; the conditional-independence assumption is from §Methods.
- Bayesian-network literature for "factored representation" terminology (e.g., Koller & Friedman, *Probabilistic Graphical Models*).
