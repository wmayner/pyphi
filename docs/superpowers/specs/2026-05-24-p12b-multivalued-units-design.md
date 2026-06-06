# P12b — Multi-Valued Units (k-ary IIT Analysis)

**Status:** Draft (awaiting review)
**Date:** 2026-05-24
**Branch:** `feature/p12b-factored-kary` (worktree at `../pyphi-p12b`)
**Branch baseline:** `2.0` at `15720651` (post-P12a + follow-ups)
**Related:**
- P12a design spec at `docs/superpowers/specs/2026-05-22-p12a-factored-tpm-design.md`
- P12a plan at `docs/superpowers/plans/2026-05-22-p12a-factored-tpm.md`
- ROADMAP item P12 (parent) and P12c (xarray state_space coord labels — separate follow-on)
- ROADMAP item P9.5 (math-fingerprint cache keys — separate follow-on surfaced during this brainstorm)

---

## 1. Background

P12a established the storage and protocol layers needed for multi-valued substrates:
`FactoredTPM` as canonical Substrate storage; the `TPM` Protocol gaining `alphabet_sizes`;
the SBN-conversion bridge for binary-only IIT analysis. P12a explicitly deferred the
multi-valued user-facing surface and the native k-ary math to P12b.

The 2021 multi-valued paper — Gomez JD, Mayner WGP, Beheler-Amass M, Tononi G,
Albantakis L, *Computing Integrated Information (Φ) in Discrete Dynamical Systems with
Multi-Valued Elements.* Entropy. 2021;23(1):6
(https://doi.org/10.3390/e23010006) — is the canonical theoretical reference.
The dead `nonbinary` branch (PR #42, #47, #48 from 2020–2021) is archaeology, not
extension target.

P12b lands:

1. **Native k-ary cause inversion math** in `pyphi/core/tpm/marginalization.py`. Binary
   keeps the SBN bridge to preserve byte-identical goldens; k>2 uses a new per-factor
   likelihood-product code path.
2. **Hot-path cutover** to FactoredTPM consumption. `System.cause_tpm` returns the new
   `CausePosterior` type; `System.effect_tpm` returns `FactoredTPM`. The `_inner`
   unwrap pattern from P12a is retired.
3. **A new type hierarchy** for joint-distribution storage: `JointDistribution` base
   class, `JointTPM` and `CausePosterior` as siblings (not parent/child).
4. **User-facing multi-valued constructor API** for Substrate, with flexible
   `state_space=` and an `alphabet=` shortcut. The `alphabet_sizes=` parameter
   (redundant per P12a brainstorm) is removed.
5. **State_space as canonical TPM-level metadata** on FactoredTPM; Substrate delegates.
6. **Substrate.joint_tpm() unification** — no more alphabet-branched shape.
7. **AC parallel cutover** — `TransitionSystem` mirrors System's changes; k>2 AC works.
8. **Declarative measure-alphabet support metadata** — measures declare which alphabets
   they support; EMD raises for k>2 with a Gomez 2021 citation.
9. **k>2 golden fixtures + end-to-end SIA and AC smoke tests.**

Macro (`pyphi/macro.py`) stays binary-only — multi-valued state grouping is its own
design problem, deferred to the existing P7b deferral.

---

## 2. Scope

### 2.1 In scope

- Native k-ary cause TPM math (per-factor likelihood product) in
  `pyphi/core/tpm/marginalization.py`.
- Full hot-path cutover: `System.cause_tpm` / `System.effect_tpm` return precise typed
  objects (`CausePosterior` / `FactoredTPM`); the `_inner` unwrap with `# type: ignore`
  goes away.
- Parallel `TransitionSystem` cutover in `pyphi/actual.py`.
- Downstream consumers (`pyphi/core/repertoire_algebra.py`, `pyphi/node.py`) updated to
  consume the new return types directly without materializing the joint.
- New type hierarchy:
  - `pyphi.JointDistribution` — multidimensional joint-probability-storage base class.
  - `pyphi.JointTPM(JointDistribution)` — forward conditional TPM (refactored from
    P12a's standalone class).
  - `pyphi.CausePosterior(JointDistribution)` — joint posterior over past states,
    returned by `cause_tpm`.
- User-facing `Substrate` constructor changes:
  - New `state_space=` keyword (flexible: flat-uniform or tuple-of-tuples-per-node).
  - New `alphabet=` shortcut for uniform-alphabet integer labels.
  - `alphabet_sizes=` parameter removed (derivable from state_space or factor shapes).
  - `Substrate(tpm=FactoredTPM(...))` raises (use `marginals=` or `from_factored`).
- `FactoredTPM` constructor signature change:
  - `state_space=` parameter; `alphabet_sizes=` parameter removed; alphabet sizes
    derived from state_space lengths.
  - State_space canonically stored on FactoredTPM; Substrate exposes via delegated
    `@property`.
- `Substrate.joint_tpm()` returns the explicit-alphabet shape for both binary and
  k-ary; legacy callsites (`convert.state_by_node2state_by_state`, `infer_cm`, etc.)
  migrated to consume the unified shape.
- Declarative `supports_alphabet` metadata on every registered measure. EMD-family
  measures declare binary-only support; intrinsic-difference family declares
  alphabet-generic. Operation-level dispatcher raises `NotImplementedError` with a
  Gomez 2021 citation for unsupported (measure, alphabet) combinations.
- k>2 golden fixtures: at minimum a synthetic small k=3 substrate; ideally the
  p53-Mdm2 network from Gomez 2021; plus a heterogeneous-alphabet fixture.
- End-to-end smoke tests for k>2 SIA (`pyphi.compute.sia`) and AC
  (`pyphi.compute.account`).
- Documentation: API docs for new types; brief mention in IIT 4.0 demo notebook;
  fix the pre-existing `docs/conventions.rst` doctest while in passing.
- Changelog fragment.

### 2.2 Out of scope (deferred)

See §9 for the complete deferred-items list with destinations. Highlights:

- Macro analysis with k>2 (P7b deferral; state-grouping is its own design problem).
- Native k-ary EMD (per Gomez 2021 §2.3; EMD raises for k>2).
- New multi-valued tutorial notebook (post-2.0).
- Math-fingerprint cache keys for cross-label cache reuse (ROADMAP P9.5).
- Paper-aligned `cause_tpm`/`effect_tpm` terminology rename (ROADMAP follow-on).
- Unify binary cause-TPM onto native path; retire `_legacy_backward_tpm` (ROADMAP
  follow-on).
- Consolidate `pyphi/tpm.py` into `pyphi/core/tpm/` (ROADMAP follow-on).
- xarray state_space coord labels (ROADMAP P12c).

### 2.3 Success criteria

- All P12a binary goldens stay **byte-identical** with their P12a outputs.
- New k>2 goldens pass with computed phi values that round-trip stably.
- End-to-end k>2 SIA smoke test produces a phi value via `pyphi.compute.sia(...)`.
- End-to-end k>2 AC smoke test produces an `Account` via `pyphi.compute.account(...)`.
- The `_inner` `# type: ignore[attr-defined]` pattern is gone from `pyphi/system.py` and
  `pyphi/actual.py`.
- `Substrate.joint_tpm()` returns the unified explicit-alphabet shape for both binary
  and k-ary.
- A user can write `Substrate(marginals=[...], state_space=...)` (any of the parsing
  forms in §4) and `pyphi.compute.sia(system)` returns a valid SIA.
- Pyright clean (0 errors / 5 baseline warnings).
- Ruff check + format: clean.
- Perf-budget gate (P11.8 Tier 1) stays green on the 5 binary fixtures.

---

## 3. Architecture

### 3.1 The shape of the change in one paragraph

P12a established storage (FactoredTPM is the canonical Substrate TPM; `TPM` Protocol
defined; SBN-bridge for binary cause inversion). P12b lands the **math layer** (native
k-ary cause inversion via per-factor likelihood product) and the **consumption layer**
(System, TransitionSystem, repertoire_algebra, Node consume FactoredTPM factors
directly, no joint materialization on the hot path). State_space joins the FactoredTPM
as TPM-level metadata. `Substrate.joint_tpm()` stops branching on alphabet. AC moves
parallel to System throughout. The legacy `JointTPM` class becomes a sibling of a new
`CausePosterior` class under a shared `JointDistribution` base.

### 3.2 Mathematical content of the new code path

The native k-ary cause TPM, conditioned on observing `state` over `node_indices` for a
substrate with per-node `factor_i` of shape `(a_1, …, a_N, a_i)`:

```
cause(s_t) ∝ Π_{i ∈ node_indices} factor_i(s_t)[state[i]]
```

Multiply across the observed nodes; normalize over s_t. The output is a joint posterior
over past states (`alphabet_sizes`-shaped distribution).

This is identical in formalism to the binary `_legacy_backward_tpm` math — only the
implementation differs (per-factor likelihood product on factors vs. joint conditioning
+ Bayes on the flattened joint). For binary, the two paths produce floating-point-
equivalent results within the property test's `atol=1e-10` bar.

**Cause posterior shape (deferred §5 decision):** the exact output shape contract
(`alphabet_sizes` vs `alphabet_sizes + (1,)`) is resolved at plan-time by an audit of
`_legacy_backward_tpm`'s actual output and `_single_node_cause_repertoire`'s actual
dependencies. The audit reveals one of two cases:

- Both paths can naturally produce shape `alphabet_sizes` — k-ary path uses this
  natively; binary path emits the same shape; no shim required.
- The SBN-bridge produces a trailing-singleton-axis shape that downstream depends on
  — `CausePosterior.__init__` becomes the canonicalization point, accepting either
  shape and storing the canonical form for downstream consumption.

Either way: no `[..., np.newaxis]` shim in the dispatcher code. The canonicalization
(if needed) lives inside `CausePosterior`.

### 3.3 Cause and effect return-type semantics

The cause and effect sides have **intentionally asymmetric** return types, reflecting
the underlying math:

| Side | Return type | Why |
|---|---|---|
| `System.cause_tpm` | `CausePosterior` (joint distribution over past states) | The cause posterior does not factor across past nodes in general (observing a future node couples its past inputs; cf. the AND-gate counterexample in §3.4). Natural representation: joint storage. |
| `System.effect_tpm` | `FactoredTPM` (conditioned forward factors) | The effect TPM is a forward conditional that retains the per-node factor structure under `condition()`. Native factored representation preserves the perf opportunity. |

Both are typed (subclasses of `JointDistribution` and the `TPM` Protocol respectively),
not raw ndarrays. The asymmetry surfaces in type hints but not in the storage layer's
abstract interface.

### 3.4 Why cause distributions don't factor across past nodes

For a substrate where node 0 at t+1 is the AND of nodes 0 and 1 at time t:

- If observed `s_{0,t+1} = 0` and we condition (M = {0}, μ = 0):
  - `factor_0(s_t)[0]` is 1 when `s_t ∈ {(0,0), (0,1), (1,0)}`; 0 when `s_t = (1,1)`.
  - cause distribution: uniform 1/3 on the three "not (1,1)" past states.
- Per-past-node marginals from this:
  - `P(s_{0,t}=0) = 2/3`, `P(s_{1,t}=0) = 2/3`.
- If they factored across past nodes: `P(s_{0,t}=0, s_{1,t}=0) = 2/3 × 2/3 = 4/9`.
- Actual joint: `P(s_{0,t}=0, s_{1,t}=0) = 1/3 ≠ 4/9`.

Observing the AND output couples the past inputs — they are not conditionally
independent in the posterior. This is why `CausePosterior` lives alongside `JointTPM`
as a sibling rather than being represented as a "FactoredTPM-for-the-past."

### 3.5 The type hierarchy

```
JointDistribution                    (multidimensional probability storage,
│                                    marginalize_out, normalize, array machinery,
│                                    ProxyMetaclass-driven arithmetic)
├── JointTPM                          forward conditional P(s_{t+1} | s_t)
│                                    (+ TPM-specific: condition_tpm, subtpm,
│                                    expand_tpm, infer_cm, infer_edge)
└── CausePosterior                    joint posterior P(s_t | s_{t+1,M} = μ)
                                     (no TPM-specific methods; semantically tagged
                                     for "posterior over past" use)
```

The user types `isinstance(x, JointDistribution)` for "any joint-stored probability
distribution"; `isinstance(x, JointTPM)` for "specifically a forward conditional";
`isinstance(x, CausePosterior)` for "specifically a cause-side posterior."

### 3.6 The asymmetry is theoretically honest

`JointTPM` IS-A multidimensional probability distribution. `CausePosterior` IS-A
multidimensional probability distribution. Neither IS-A the other. Subtyping
`CausePosterior` from `JointTPM` (the "Option A" considered during brainstorming) would
be a convenient Liskov-substitution shortcut, but the IS-A claim is semantically wrong:
a posterior is not "a kind of" conditional TPM. The sibling hierarchy reflects the
actual semantics.

### 3.7 Information flow at use (binary, post-P12b)

```
User: Substrate(tpm=joint_array)
  ↓
constructor: FactoredTPM.from_joint(joint) → FactoredTPM with state_space=((0,1),)*n
  ↓
substrate.factored_tpm (canonical storage)
  ↓
System(substrate, state, ...).cause_tpm
  ↓
marginalization.cause_tpm(factored, state, node_indices)
  ↓
[binary branch] _cause_tpm_factored_binary: SBN-bridge → _legacy_backward_tpm
  ↓
CausePosterior wrapping the cause-side joint posterior over past binary states
  ↓
repertoire_algebra._single_node_cause_repertoire reads from this via
inherited .marginalize_out()
```

Binary goldens reproduce byte-identically because the SBN-bridge → legacy code is
unchanged. Only the wrapping type changes from `JointTPM` (post-P12a) to
`CausePosterior` (post-P12b).

### 3.8 Information flow at use (k-ary)

```
User: Substrate(marginals=[f0, f1, f2], state_space=(("L","M","H"),)*3)
  ↓
constructor: FactoredTPM(factors=[...], state_space=...)
  ↓
substrate.factored_tpm (canonical storage; state_space-labeled)
  ↓
System(substrate, state, ...).cause_tpm
   (state may be either tuple-of-ints or tuple-of-labels; coerced to ints internally)
  ↓
marginalization.cause_tpm(factored, state, node_indices)
  ↓
[k>2 branch] _cause_tpm_factored_kary: per-factor likelihood product, normalize
  ↓
CausePosterior over past k-ary joint states
  ↓
downstream repertoire_algebra reads via .marginalize_out() exactly as for binary
```

### 3.9 State_space data flow

State_space lives canonically on `FactoredTPM`:

- **Construction:** accepted via `Substrate(..., state_space=...)` or
  `FactoredTPM(..., state_space=...)`. Substrate constructor passes through to
  FactoredTPM. Single authoritative storage: `FactoredTPM.state_space`.
- **Access:** `Substrate.state_space` is a `@property` delegating to
  `self._factored_tpm.state_space`.
- **Validation:** `FactoredTPM._validate` extended — state_space length per node must
  match the corresponding factor's last-dim shape; labels must be unique within each
  node's state set.
- **Display:** `__repr__` / `__str__` on Substrate and FactoredTPM include state_space
  when non-default (i.e., when not just integer 0..k-1).
- **Serialization:** `jsonify` carries state_space through round-trips; P12b adds
  round-trip tests.

---

## 4. User-facing API in detail

### 4.1 Substrate constructor signature (final)

```python
class Substrate:
    def __init__(
        self,
        tpm: NDArray[np.float64] | JointTPM | dict[str, Any] | None = None,
        cm: ArrayLike | None = None,
        node_labels: Sequence[str] | NodeLabels | None = None,
        purview_cache: cache.PurviewCache | None = None,
        *,
        marginals: Sequence[ArrayLike] | None = None,
        state_space: StateSpace | None = None,
        alphabet: int | None = None,
    ) -> None: ...
```

### 4.2 Canonical user-call patterns

```python
# Binary substrate — legacy joint form, default integer state labels
pyphi.Substrate(tpm=joint_array)                            # state_space = ((0,1),)*n

# Binary substrate with state_space labels
pyphi.Substrate(tpm=joint_array, state_space=("OFF","ON"))

# Multi-valued, uniform alphabet, integer labels (alphabet= shortcut)
pyphi.Substrate(marginals=[f0, f1, f2], alphabet=3)

# Multi-valued, uniform alphabet, semantic labels
pyphi.Substrate(marginals=[f0, f1, f2], state_space=("LOW","MID","HIGH"))

# Heterogeneous alphabets with per-node semantic labels
pyphi.Substrate(
    marginals=[f0_binary, f1_ternary, f2_quaternary],
    state_space=(
        ("OFF", "ON"),
        ("LOW", "MID", "HIGH"),
        ("S0", "S1", "S2", "S3"),
    ),
)

# Direct FactoredTPM input via factory
factored = FactoredTPM(factors=[f0, f1], state_space=("OFF","ON"))
pyphi.Substrate.from_factored(factored, cm=cm)
```

### 4.3 state_space parsing rule

Defined in `_normalize_state_space(raw, factors) -> tuple[tuple[Any, ...], ...]`:

- `None` (default) → integer labels 0..k-1 per node; k inferred from each factor's
  last-dim size. Heterogeneous alphabets supported (each factor gets its own integer
  range).
- Flat sequence of strings/ints (e.g. `("LOW","MID","HIGH")` or `(0,1,2)`) → uniform
  labels across all nodes. Alphabet size = `len(state_space)`. All factors must have
  matching last-dim.
- Sequence-of-sequences (e.g. `(("OFF","ON"), ("LOW","MID","HIGH"))`) → per-node
  labels. `state_space[i]` labels factor `i`'s last-dim entries.

Detection rule between flat and per-node: if every element of `state_space` is itself a
non-string sequence, treat as per-node; otherwise treat as uniform-flat.

Edge cases:

| Input | Parsed as | Outcome |
|---|---|---|
| `("OFF","ON")` | uniform binary, string labels | OK |
| `(0,1,2)` | uniform k=3, integer labels | OK |
| `(("OFF","ON"),)*3` | per-node: 3 binary nodes | OK |
| `()` | empty | `ValueError("state_space cannot be empty")` |
| `("OFF",)` | uniform alphabet=1 | Fails `alphabet >= 2` validation |
| `(("OFF","ON"),("LOW","MID"))` for a 3-node substrate | mismatched per-node length | `ValueError("state_space has 2 per-node entries; substrate has 3 nodes")` |

### 4.4 alphabet= shortcut

`alphabet=k` is sugar that:

- Asserts uniform alphabet size `k` across all nodes.
- Sets `state_space=tuple(range(k))` (uniform integer labels).
- Mutually exclusive with `state_space=` — passing both raises
  `ValueError("pass alphabet= or state_space=, not both")`.
- `alphabet < 2` fails the construction-time `alphabet >= 2` validation.

### 4.5 Mechanism-state as labels (when state_space is set)

`System` accepts state as either integer indices or state-space labels:

```python
sub = pyphi.Substrate(marginals=[...], state_space=(("L","M","H"),)*3)

pyphi.System(sub, state=(0, 1, 2))           # integer indices — always works
pyphi.System(sub, state=("L", "M", "H"))     # labels — looked up via state_space
```

Internal storage: integer indices (canonical form). The `_coerce_state_to_indices`
helper converts labels to indices using state_space lookups. Stored on System for
hashing and equality.

**Disambiguation rule** when integer labels are used (e.g.,
`state_space=(0,1,2)` AND `state=(0,1,2)`): check membership in `state_space[i]`
first; if all elements match, treat as labels. Otherwise treat as integer indices. This
privileges label-interpretation when labels are explicit.

### 4.6 Display

```python
>>> sub = pyphi.Substrate(marginals=[f0, f1], state_space=(("L","M","H"),)*2)
>>> sub
Substrate(
    state_space=(('L','M','H'), ('L','M','H')),
    cm=[[1 1]
        [1 1]],
    node_labels=NodeLabels(('A', 'B')),
)

>>> sub.factored_tpm
FactoredTPM(n_nodes=2, alphabet_sizes=(3, 3), state_space=(('L','M','H'),('L','M','H')))
```

Binary substrates with default integer state_space omit it from `__repr__` (preserves
the terser binary display):

```python
>>> sub_binary = pyphi.Substrate(tpm=joint_binary)
>>> sub_binary
Substrate(cm=[[1 1] [1 1]], node_labels=NodeLabels(('A', 'B')))
```

### 4.7 Public namespace additions

```python
# pyphi/__init__.py exports new in P12b
from .core.tpm import CausePosterior as CausePosterior
from .core.tpm import JointDistribution as JointDistribution
# (existing: FactoredTPM, JointTPM, TPM, Substrate, System, etc.)
```

---

## 5. Math implementation

### 5.1 Pre-flight audit (first plan task)

Before any code changes, one investigative task documents:

1. The actual output shape of `pyphi.tpm.backward_tpm` (referred to internally as
   `_legacy_backward_tpm`).
2. The actual shape dependencies of `pyphi.core.repertoire_algebra._single_node_cause_repertoire`
   on its `mechanism_node.cause_tpm` input.

Based on the audit, the plan settles the cause-output canonical shape: either pure
`alphabet_sizes` (no trailing axis) if downstream consumers are agnostic, or with a
trailing singleton if downstream depends on it. If a trailing singleton is needed, it
is canonicalized inside `CausePosterior.__init__`, not added ad-hoc in dispatcher code.

### 5.2 Cause inversion — the binary path (unchanged from P12a)

```python
def _cause_tpm_factored_binary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Binary cause TPM via SBN conversion + legacy backward_tpm.

    Preserves byte-identical goldens. The legacy backward_tpm produces the
    canonical binary cause output that all pre-P12b goldens were derived
    against.
    """
    n = factored.n_nodes
    sbn = np.stack([factored.factor(i)[..., 1] for i in range(n)], axis=-1)
    return CausePosterior(_legacy_backward_tpm(sbn, state, node_indices))
```

The output is wrapped in `CausePosterior` (replacing the `JointTPM` wrapping from
P12a). `CausePosterior` is a `JointDistribution` subclass, sibling to `JointTPM`.
Downstream code that called `.marginalize_out()` on `JointTPM` results keeps working
because both classes inherit `marginalize_out` from `JointDistribution`.

### 5.3 Cause inversion — native k-ary path

```python
def _cause_tpm_factored_kary(
    factored: FactoredTPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Native k-ary cause TPM via per-factor likelihood product.

    Computes P(s_t | s_{M, t+1} = state) over the joint past state space,
    where M = node_indices. The likelihood at each past joint state is the
    product of per-mechanism-node factor lookups:

        cause(s_t) ∝ Π_{i ∈ node_indices} factor_i(s_t)[state[i]]

    Normalized over s_t.
    """
    alphabet_sizes = factored.alphabet_sizes
    likelihood = np.ones(alphabet_sizes, dtype=np.float64)
    for i in node_indices:
        likelihood = likelihood * factored.factor(i)[..., state[i]]
    total = likelihood.sum()
    if total <= 0:
        raise exceptions.StateUnreachableBackwardsError(state)
    posterior = likelihood / total
    return CausePosterior(posterior)
```

Output shape: per §5.1 audit. `CausePosterior.__init__` canonicalizes if needed.

### 5.4 Cause-side dispatcher

```python
def cause_tpm(
    tpm: TPM,
    state: tuple[int, ...],
    node_indices: tuple[int, ...],
) -> CausePosterior:
    """Cause TPM dispatcher — IIT 4.0 Eq. 3."""
    if isinstance(tpm, FactoredTPM):
        if all(a == 2 for a in tpm.alphabet_sizes):
            return _cause_tpm_factored_binary(tpm, state, node_indices)
        return _cause_tpm_factored_kary(tpm, state, node_indices)
    if isinstance(tpm, JointTPM):
        return CausePosterior(_legacy_backward_tpm(tpm._inner, state, node_indices))
    arr = tpm.to_array()
    return CausePosterior(_legacy_backward_tpm(arr, state, node_indices))
```

### 5.5 Effect — alphabet-generic via condition

```python
def effect_tpm(tpm: TPM, background: Mapping[int, int]) -> TPM:
    """Effect TPM = tpm.condition(background). IIT 4.0 Eq. 4.

    FactoredTPM.condition (from P12a) is alphabet-generic. JointTPM.condition_tpm
    (legacy) is alphabet-generic. The P12a-era k>2 NotImplementedError guard comes
    out — effect_tpm Just Works for k-ary.
    """
    return tpm.condition(background)
```

Return type follows input type: `FactoredTPM → FactoredTPM`, `JointTPM → JointTPM`. In
the post-cutover canonical path, the FactoredTPM-in/FactoredTPM-out case dominates.

### 5.6 Downstream consumption — repertoire_algebra

After cutover, `_single_node_cause_repertoire` consumes `CausePosterior` directly:

```python
@_memoize
def _single_node_cause_repertoire(
    cs: Any, mechanism_node_index: int, purview_set: frozenset[int]
) -> NDArray[np.float64]:
    """Single-node cause repertoire from per-node cause posterior."""
    mechanism_node = cs._index2node[mechanism_node_index]
    posterior = mechanism_node.cause_tpm  # CausePosterior
    return posterior.marginalize_out(
        mechanism_node.inputs - purview_set
    ).tpm
```

`.marginalize_out(...).tpm` works identically for binary and k-ary because
`JointDistribution.marginalize_out` is alphabet-generic numpy summation.

`_single_node_effect_repertoire` consumes the FactoredTPM:

```python
@_memoize
def _single_node_effect_repertoire(
    cs: Any,
    condition: FrozenMap,
    purview_node_index: int,
    direction: Direction,
) -> NDArray[np.float64]:
    """Single-node effect repertoire from FactoredTPM factor."""
    purview_node = cs._index2node[purview_node_index]
    factored_tpm = (
        purview_node.cause_tpm if direction == Direction.CAUSE
        else purview_node.effect_tpm
    )
    factor = factored_tpm.condition_factor(purview_node_index, condition)
    nonmechanism_inputs = purview_node.inputs - set(condition)
    for axis in sorted(nonmechanism_inputs, reverse=True):
        factor = factor.sum(axis=axis)
    return factor.reshape(
        repertoire_shape(cs.substrate.node_indices, (purview_node_index,))
    )
```

### 5.7 Declarative measure-alphabet support

Each registered measure gains a `supports_alphabet` attribute — a callable
`(alphabet_sizes: tuple[int, ...]) -> bool`. Default for new registrations: `lambda _:
True` (alphabet-generic).

```python
# pyphi/measures/distribution.py

_binary_only = lambda a: all(x == 2 for x in a)
_any_alphabet = lambda a: True

@measures.register("EMD", supports_alphabet=_binary_only)
def emd(...): ...

@measures.register("AID", supports_alphabet=_any_alphabet)
def aid(...): ...

@measures.register("GENERALIZED_INTRINSIC_DIFFERENCE", supports_alphabet=_any_alphabet)
def gid(...): ...
```

At operation boundary (measure resolution before use):

```python
measure = measures.resolve(config.formalism.iit.mechanism_phi_measure)
if not measure.supports_alphabet(substrate.alphabet_sizes):
    raise NotImplementedError(
        f"Measure {measure.name!r} does not support alphabet sizes "
        f"{substrate.alphabet_sizes}. For multi-valued substrates, use an "
        f"alphabet-generic measure (AID, GID, INTRINSIC_INFORMATION, "
        f"GENERALIZED_INTRINSIC_DIFFERENCE). "
        f"See Gomez et al. 2021 §2.3 (https://doi.org/10.3390/e23010006) for "
        f"the theoretical rationale on EMD specifically."
    )
```

Single source of truth per measure; the dispatcher reads the metadata and raises. Adding
k-ary EMD in the future would be a one-line registry-entry change.

### 5.8 Plan-time implementation order

The plan orders tasks to minimize golden-breakage risk:

1. Audit `_legacy_backward_tpm` output + downstream dependencies; settle §5.1.
2. Extract `JointDistribution` base class; refactor `JointTPM` to subclass it.
3. Add `CausePosterior(JointDistribution)`.
4. Update `marginalization.cause_tpm` binary path to return `CausePosterior` instead of
   `JointTPM`. Verify goldens byte-identical.
5. Add native k-ary path + dispatcher.
6. Hot-path cutover: `System.cause_tpm` / `effect_tpm` return precise types; `_inner`
   unwrap retired.
7. Parallel cutover: `TransitionSystem`.
8. Downstream `core/repertoire_algebra` and `node.py`.
9. State_space plumbing.
10. `Substrate.joint_tpm()` unification.
11. Declarative measure metadata + operation-level guard.
12. k>2 goldens; end-to-end SIA + AC smoke tests.
13. Documentation; changelog.

---

## 6. Validation, error handling, edge cases

### 6.1 FactoredTPM construction validation (extended from P12a)

```python
def _validate(factored: FactoredTPM) -> None:
    a = factored.alphabet_sizes  # derived from state_space
    if len(factored._state_space) != factored.n_nodes:
        raise exceptions.InvalidTPM(
            f"state_space has {len(factored._state_space)} per-node entries; "
            f"factors imply {factored.n_nodes} nodes"
        )
    for i, labels in enumerate(factored._state_space):
        if len(labels) != a[i]:
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has {len(labels)} labels but factor[{i}] "
                f"has alphabet size {a[i]}"
            )
        if len(set(labels)) != len(labels):
            raise exceptions.InvalidTPM(
                f"state_space[{i}] has duplicate labels: {labels}"
            )
    # ... existing P12a invariants: alphabet >= 2, factor shape, sum-to-1
```

### 6.2 Substrate constructor argument validation

1. `tpm=` and `marginals=` are mutually exclusive → `ValueError`.
2. `alphabet=` and `state_space=` are mutually exclusive → `ValueError`.
3. One of `tpm=` or `marginals=` is required → `ValueError`.
4. `Substrate(tpm=FactoredTPM(...))` raises (use `marginals=` or `from_factored`).
5. `alphabet` must be `int >= 2` → `ValueError`.
6. State_space mismatch with marginals raises with explicit per-node detail.

### 6.3 State-coercion errors

```python
def _coerce_state_to_indices(state, state_space):
    if label_not_found:
        raise ValueError(
            f"state[{i}] = {state[i]!r} not in state_space[{i}] = {state_space[i]!r}"
        )
    if integer_out_of_range:
        raise ValueError(
            f"state[{i}] = {state[i]} out of range for alphabet size "
            f"{len(state_space[i])}"
        )
```

### 6.4 Cause-inversion edge cases

- **Unreachable past state.** `_cause_tpm_factored_kary`'s likelihood product sums to 0
  → raises `StateUnreachableBackwardsError(state)` matching the legacy pattern.
- **Single-state alphabet.** Caught upstream at FactoredTPM construction (`alphabet
  >= 2`).
- **All-zero factor.** Allowed; sum-to-1 still satisfied via other states. Some past
  states may have zero likelihood after inversion.
- **NaN in factors.** Caught at construction (sum-to-1 fails); never reaches
  marginalization.

### 6.5 EMD k>2 error path

Single declarative entry via the `supports_alphabet` metadata (§5.7); single error
message format including the measure name, alphabet sizes, the alphabet-generic
alternatives, and the Gomez 2021 citation.

### 6.6 `_inner` cleanup verification

Post-cutover, a grep-based regression test asserts the unwrap pattern is gone from
production code:

```python
def test_no_inner_unwrap_pattern_in_production():
    import subprocess
    result = subprocess.run(
        ["grep", "-rn", r"_inner if hasattr", "pyphi/"],
        capture_output=True, text=True,
    )
    offending = [
        line for line in result.stdout.splitlines()
        if not line.startswith("pyphi/tpm.py:")  # JointTPM class def is allowed
        and "# " not in line.split(":", 2)[-1]   # comments OK
    ]
    assert not offending, f"Found _inner unwrap patterns: {offending}"
```

---

## 7. Testing strategy

### 7.1 Existing binary goldens — byte-identical

All 23 P12a binary goldens must continue to pass byte-identically after every commit.
At each commit boundary:

```bash
uv run pytest test/test_golden_regression.py -v
# Expected: 23/23 pass; byte-identical against stored fixtures.
```

This is the load-bearing regression net for the type-hierarchy refactor, the hot-path
cutover, and the SBN-bridge math wrapping change.

### 7.2 New k>2 goldens

- `test/data/golden/v1/multivalued_k3_tiny.{json,npz}` — synthetic 2- or 3-node k=3
  substrate; computed phi pinned; canonical "smallest k>2 SIA" regression test.
- `test/data/golden/v1/multivalued_p53_mdm2.{json,npz}` — the 12-state p53-Mdm2 network
  from Gomez 2021 §3, **conditional on reproducibility**. Reproduction attempt is a
  P12b validation milestone. If phi values match the paper within `precision`, this
  becomes the second golden. Otherwise the discrepancy is recorded.
- `test/data/golden/v1/multivalued_2x3x3.{json,npz}` — heterogeneous-alphabet 3-node
  substrate exercising the per-node-different-alphabet code path.

### 7.3 K-ary property tests (extending P12a)

P12a added `test_factored_tpm_kary.py` for FactoredTPM math. P12b extends with
cause-side property tests in `test/test_marginalization_kary.py`:

- Posterior sums to 1 within precision.
- Posterior entries are non-negative.
- Binary equivalence: `_cause_tpm_factored_kary(binary_input)` agrees with
  `_cause_tpm_factored_binary(binary_input)` within `atol=1e-10`. This is the load-
  bearing math-correctness test.

### 7.4 End-to-end smoke tests

```python
# test/test_substrate_multivalued.py

def test_kary_sia_end_to_end():
    """Smoke test: construct a k=3 substrate; run SIA; receive a phi value."""
    # ... build small k=3 substrate ...
    sia = pyphi.compute.sia(system)
    assert sia.phi >= 0
    assert sia.partition is not None


def test_kary_account_end_to_end():
    """Smoke test: construct a k=3 substrate; compute an account."""
    # ... build small k=3 substrate ...
    account = pyphi.compute.account(transition)
    assert account is not None
    assert all(link.alpha >= 0 for link in account)
```

### 7.5 Type-hierarchy tests

```python
# test/test_joint_distribution.py

def test_jointtpm_isinstance_jointdistribution(): ...
def test_cause_posterior_isinstance_jointdistribution(): ...
def test_cause_posterior_not_isinstance_jointtpm(): ...
def test_joint_distribution_marginalize_out_shared(): ...
```

### 7.6 State_space tests

`test/test_substrate_state_space.py` covers:

- Uniform string labels parse correctly.
- Per-node heterogeneous labels parse correctly.
- Default integer labels inferred from factor shapes.
- `alphabet=` shortcut equivalent to integer `state_space=`.
- Mutual exclusion of `alphabet=` and `state_space=` raises.
- State-as-labels resolves to state-as-indices.
- Mismatch errors are clear and per-node.

### 7.7 Measure-alphabet support tests

`test/test_measure_alphabet_support.py`:

- EMD raises clearly when invoked with a k>2 substrate via config override.
- AID works for a k>2 substrate end-to-end.
- Every registered measure has a callable `supports_alphabet` attribute.

### 7.8 `_inner` cleanup verification

Per §6.6 — the grep-based regression test runs in the fast lane.

### 7.9 Doctest scope

Per the P12a CLAUDE.md addition: P12b verification MUST include `uv run pytest`
(no path argument) at every gate. Bare-path invocations skip `pyphi/` source
doctests. The plan's verification recipe enforces this.

### 7.10 Perf budget

P12b does not add new perf-budget fixtures. Existing 5 binary perf-budget tests must
stay within `max(3.0, 4×median)` floors. The hot-path cutover changes how return types
flow through downstream; perf-budget gate catches any regression.

### 7.11 Final acceptance gates

| Gate | Command | Expected |
|---|---|---|
| Full suite (incl. doctests) | `uv run pytest --tb=short -q` | 0 failures |
| Fast lane | `uv run pytest test/ -m "not slow" -q` | 0 failures |
| Slow lane | `uv run pytest test/ --slow -q` | 0 failures |
| Goldens (binary) | `uv run pytest test/test_golden_regression.py -v` | 23 + new k>2; byte-identical on binary |
| Perf budget | `uv run pytest test/test_perf_budget.py -v` | All within floor |
| Pyright | `uv run pyright pyphi` | 0 errors / 5 baseline warnings |
| Ruff | `uv run ruff check pyphi test` | clean |
| End-to-end SIA k>2 | invocation in test_substrate_multivalued.py | Returns valid SIA |
| End-to-end AC k>2 | invocation in test_substrate_multivalued.py | Returns valid Account |
| `_inner` grep clean | grep regression test | No production-code matches |

---

## 8. Migration & cutover plan

### 8.1 Pre-flight audit (first plan task)

Settle the cause-output canonical shape per §5.1 by reading
`pyphi.tpm.backward_tpm` and `pyphi.core.repertoire_algebra._single_node_cause_repertoire`.
Plan time, not implementation time. The decision unblocks the type-hierarchy commits.

### 8.2 Commit sequence (16-18 commits)

**Phase A — Type hierarchy:**

1. Extract `JointDistribution` base class; refactor `JointTPM` to subclass it. Move
   `ProxyMetaclass` to base. Verify: existing tests pass; goldens byte-identical.
2. Add `CausePosterior(JointDistribution)`. Export from `pyphi.__init__`.
3. Update `marginalization._cause_tpm_factored_binary` to return `CausePosterior`
   instead of `JointTPM`. Liskov keeps downstream working. Verify: goldens
   byte-identical.
4. Checkpoint task: verify goldens byte-identical post-type-hierarchy.

**Phase B — Native k-ary math:**

5. Settle §5.1 from the audit; implement `_cause_tpm_factored_kary`; dispatcher routes
   binary vs k-ary in `marginalization.cause_tpm`. Verify: k-ary property tests pass;
   goldens byte-identical.
6. Add k-ary property tests (`test/test_marginalization_kary.py`) including the
   binary-equivalence property.
7. Remove P12a's defensive k>2 `NotImplementedError` from
   `_effect_tpm_factored` (effect is alphabet-generic via `condition`).

**Phase C — Hot-path cutover:**

8. `System.cause_tpm` → `CausePosterior`; `System.effect_tpm` → `FactoredTPM`; remove
   `_inner` unwrap pattern. Type hints precise. Verify: goldens byte-identical;
   `_inner` grep clean for production code.
9. Parallel cutover for `TransitionSystem` in `pyphi/actual.py`. Remove
   `# type: ignore[attr-defined]` at `actual.py:261`.
10. Downstream `core/repertoire_algebra._single_node_*_repertoire` cutover. This is
    the most-load-bearing commit for binary-goldens preservation. Verify: full fast
    lane + goldens + perf-budget.

**Phase D — State_space and user-facing API:**

11. `FactoredTPM` constructor signature change: `state_space=` keyword; drop
    `alphabet_sizes=`. `_normalize_state_space` helper. `alphabet_sizes` becomes
    a derived property. Verify: FactoredTPM tests pass.
12. `Substrate` constructor adds `state_space=` and `alphabet=`. Drops
    `alphabet_sizes=`. State-coercion helper. `Substrate.state_space` property
    delegates. Verify: state_space tests pass.
13. `Substrate.joint_tpm()` alphabet-branch cleanup. Unify on the explicit-alphabet
    shape. Migrate legacy callsites (`convert.state_by_node2state_by_state`,
    `infer_cm`, etc.). Verify: callsite migrations don't break binary tests.

**Phase E — Measure surface:**

14. Declarative `supports_alphabet` metadata on measure registry. Verify: all
    measures have the attribute.
15. Operation-level guard in measure dispatch. Verify: EMD raises on k>2; AID works
    on k>2.

**Phase F — Goldens, end-to-end, docs:**

16. Add k>2 golden fixtures (synthetic + heterogeneous + p53-Mdm2 if reproducible).
17. End-to-end k>2 SIA + AC smoke tests.
18. Docs touch (API docs, IIT 4.0 demo, `docs/conventions.rst` doctest fix);
    changelog fragment; scaffold-marker sweep.

### 8.3 Known intermediate states

- **After commit 1** (JointDistribution extraction): `JointTPM` subclasses a new
  class. Internal API; users don't notice.
- **After commit 3** (CausePosterior return type): downstream consumers see
  `CausePosterior` via Liskov. No visible behavior change.
- **After commit 8** (System cutover): `System.cause_tpm` type changes. Any caller
  that explicitly type-checks against `JointTPM` would break. Grep audit before
  commit identifies any survivors.
- **After commits 11-12** (FactoredTPM / Substrate constructor changes):
  `Substrate(alphabet_sizes=...)` raises `TypeError("unexpected keyword argument")`.
  Documented in changelog as a 2.0 API break.

### 8.4 Risk mitigation

| Risk | Mitigation |
|---|---|
| Native k-ary math diverges from legacy backward_tpm on binary | Property test in commit 6 enforces binary-equivalence within 1e-10 |
| Hot-path cutover slows binary perf | Perf-budget gate at commit 10 (largest cutover) and again at final |
| `_inner` removal breaks unanticipated consumers | Commit 8/9 includes pre-commit grep for `_inner` access; investigate any survivors |
| Substrate constructor signature break catches users | Changelog documents the break; 2.0 is pre-release so no deprecation cycle needed |
| `_legacy_backward_tpm` output shape changes mid-implementation | Audit in §5.1 happens BEFORE any code change; decision locked at plan time |
| State_space construction validation rejects valid inputs | Property test in commit 11 covers labeled, integer, mixed-alphabet cases |
| Measure `supports_alphabet` callable missing or wrong | Test in commit 14 iterates all registered measures, asserts attribute presence + correctness |
| p53-Mdm2 doesn't reproduce within `precision` | Reproduction attempt is a P12b milestone; if it fails, only the synthetic golden ships |

### 8.5 Verification recipe at each commit

```bash
# Inner loop (per-commit fast feedback)
uv run pytest test/ -m "not slow" -x -q

# Pre-commit-boundary verification
uv run pytest --tb=short -q                       # full suite + doctests (per CLAUDE.md)
uv run pytest test/test_golden_regression.py -v    # byte-identical bar
uv run pytest test/test_perf_budget.py -v          # perf gate
uv run pyright pyphi                               # 0 errors / 5 baseline warnings
uv run ruff check pyphi test                       # clean
```

### 8.6 Branch & merge strategy

P12b lives on `feature/p12b-factored-kary` (worktree at `../pyphi-p12b`).

At completion:

1. Final code review (via `superpowers:finishing-a-development-branch`).
2. Merge back to `2.0` (the long-running release branch).
3. Goldens at merge match worktree's goldens.
4. Worktree disposed; branch deleted post-merge.

No push to origin until the user explicitly consents (per saved-memory
`feedback_ask_before_push`).

---

## 9. Out of scope / deferred

### 9.1 Macro analysis with k>2

MacroSystem (`pyphi/macro.py`) stays binary-only. K-ary state grouping is a design
problem (alphabet-respecting macro mapping, heterogeneous-to-uniform reductions,
alphabet-compressing coarse-graining). Belongs in the existing P7b deferral.

### 9.2 Native k-ary EMD

EMD raises `NotImplementedError` for k>2 with Gomez 2021 citation. Theory points
toward the intrinsic-difference family for multi-valued; EMD on Hamming distance
doesn't generalize cleanly. No ROADMAP entry — feature request if a specific
theoretical case emerges later.

### 9.3 New multi-valued tutorial notebook

Post-2.0 docs pass. P12b confines documentation work to: API docs for new types;
brief inline mention in the existing IIT 4.0 demo if natural; the pre-existing
`docs/conventions.rst` doctest fix.

### 9.4 Custom EMD ground metric

No user-specifiable ground metric for EMD. Theory says use AID/GID for k-ary; the
custom-ground-metric path would expand EMD's surface in a direction the theory
points away from. Future feature request if needed.

### 9.5 k>2 perf budget fixtures

The P11.8 Tier 1 perf budget keeps its 5 binary fixtures. k>2 perf characteristics
depend on the SIA-on-multi-valued workload which isn't well-understood until users
run real multi-valued analyses. Premature perf budgets would risk locking in
suboptimal floors. Pairs naturally with the "unify binary cause-TPM onto native path"
ROADMAP follow-on.

### 9.6 `Substrate.tpm` deprecation in favor of explicit `.factored_tpm` / `.joint_tpm()`

The redundancy is preserved. Breaking `.tpm` accessor would force every legacy script
to update; cost-benefit isn't obvious without surveying real usage patterns. No
ROADMAP entry — discretionary if/when the project decides to do it.

### 9.7 Math-fingerprint cache keys (cross-label cache reuse)

The repertoire cache keys on `id(System)`. Two Systems with label-distinct
math-identical substrates get distinct cache entries — wasted recomputation. P12b
does not fix this.

**ROADMAP P9.5** captures the deferred work.

### 9.8 Paper-aligned cause/effect terminology cleanup

`System.cause_tpm` returning `CausePosterior` is still named `cause_tpm` — a misnomer
("TPM" implies a conditional, not a posterior). P12b inherits the naming sloppiness
from existing pyphi conventions. Renaming is a P11.7-class focused-rename project
(captured in ROADMAP informal notes).

### 9.9 Unify binary cause-TPM onto native factored path; retire `_legacy_backward_tpm`

Binary keeps the SBN bridge indefinitely (until the follow-on project). The dual
implementation (binary via SBN, k>2 native) persists post-P12b. Captured in ROADMAP
informal notes.

### 9.10 Consolidate `pyphi/tpm.py` into `pyphi/core/tpm/`

After P12b, the module boundary is fuzzy (P12b adds new types to `core/tpm/` while
`tpm.py` retains the legacy `JointTPM`). Mechanical refactor; captured in ROADMAP
informal notes.

### 9.11 xarray state_space coord labels

The xarray backend (P12a opt-in) uses generic dim names. Adding state_space as
named coords on `xr.DataArray` would enable xarray-native indexing. **ROADMAP P12c**
captures the deferred work.

### 9.12 IIT 3.0 multi-valued path

P12b focuses on IIT 4.0 (the 2.0 default). 3.0 + multi-valued is a smaller surface
sharing infrastructure (formalism dispatch in `pyphi/formalism/`). The k-ary support
inherits via measure-registry metadata (AID is alphabet-generic), but P12b does not
add 3.0-specific k>2 goldens or smoke tests. Plan-time judgment call whether to add
an inline 3.0 k>2 test.

### 9.13 Removal / cleanup of `Substrate.factored_tpm` alias

`Substrate.factored_tpm` and `Substrate.tpm` both return the FactoredTPM (P12a).
Redundant. P12b keeps both — co-evolves with §9.6 if/when both are addressed.

---

## 10. References

- ROADMAP P12 (parent project), P12c (xarray state_space coords follow-on), P9.5
  (math-fingerprint cache-key follow-on).
- ROADMAP informal-notes section (paper-aligned naming cleanup, unify-binary-onto-native,
  consolidate-pyphi/tpm.py follow-ons).
- P12a design at `docs/superpowers/specs/2026-05-22-p12a-factored-tpm-design.md`.
- P12a plan at `docs/superpowers/plans/2026-05-22-p12a-factored-tpm.md`.
- Gomez JD, Mayner WGP, Beheler-Amass M, Tononi G, Albantakis L. *Computing Integrated
  Information (Φ) in Discrete Dynamical Systems with Multi-Valued Elements.* Entropy.
  2021;23(1):6. https://doi.org/10.3390/e23010006
- Barbosa LS, Marshall W, Albantakis L, Tononi G. *Mechanism Integrated Information.*
  Entropy. 2021;23(3):362. https://doi.org/10.3390/e23030362 (cited by Gomez 2021 for
  the AID measure family).
- Albantakis L et al. *Integrated information theory (IIT) 4.0.* PLOS Comp Bio. 2023;
  19(10):e1011465.
- PR #105 "Initial support for Implicit TPMs" (Isaac David, OPEN since 2023): the
  prior implementation reference for state_space metadata and per-node-factored TPMs.
  Archaeology for P12a/P12b; not a rebase target.
- PR #42 (Multi-valued elements, 2020), #47/#48 (Nonbinary, 2021): legacy archaeology
  on the dead `nonbinary` branch.
