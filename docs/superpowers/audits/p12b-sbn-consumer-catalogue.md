# P12b SBN cause-TPM producer/consumer catalogue

This audit catalogues every site in `pyphi/` and `test/` that produces or
consumes "SBN-form cause-TPM data" — multidimensional arrays whose trailing
axis is interpreted as per-output-node firing probability `P(node_i = 1 |
s_t)`. The audit's purpose is to enable a goldens-stable migration that
unifies the two cause-side mathematical paths currently in the codebase.

## Background: the two genuinely-different mathematical objects

The earlier `p12b-cause-shape-audit.md` concluded that both the binary
(SBN-bridge) path and the native k-ary path produce shape
`(*alphabet_sizes, n_observed_nodes)`. That conclusion is correct for the
binary path but wrong for the native k-ary path. The two paths compute
different mathematical objects:

**Legacy path — `_legacy_backward_tpm` (`pyphi/tpm.py:464-498`)**

Given an SBN-form forward TPM `forward[s_t, i] = P(node_i,t+1 = 1 | s_t)`
of shape `(*alphabet_sizes, n)`, the body computes:

```
pr_current_state = probability_of_current_state(forward, current_state)
  → shape (*alphabet_sizes, 1), value at s_t is
    ∏_i [forward[s_t, i]   if current_state[i] == 1
         else 1 - forward[s_t, i]]
  i.e. P(s_{t+1} = current_state | s_t)   (joint Bernoulli over nodes)

pr_only_background = sum over system_indices, keepdims
normalization      = sum over all axes

backward_tpm = (forward * pr_only_background / normalization)
                  .sum(background_indices, keepdims=True)
```

Shape `(*alphabet_sizes, n)`. The trailing axis is sized by `n` (number of
observed substrate nodes); after `remove_background=True` it is sized by
`len(system_indices)`. The value at `[s_t, ..., i]` is
`P(node_i,t+1 = 1 | s_t) · P(s_{t+1,M} = current_state | s_t) / normalization`
summed over background — i.e. **SBN-form**: per-node firing probability
weighted by the observed-mechanism-state likelihood. The trailing axis
slices sum to N (not 1).

**Native k-ary path — `_cause_tpm_factored_kary` (`pyphi/core/tpm/marginalization.py:62-82`)**

```
likelihood[s_t] = ∏_{i ∈ node_indices} factor_i(s_t)[state[i]]
posterior       = likelihood / likelihood.sum()
```

Shape `(*alphabet_sizes,)`. This is the proper joint posterior
`P(s_t | s_{t+1,M} = state)`. Sums to 1.

The unification decision (post-discovery) is to migrate every producer
and consumer onto the joint-posterior representation, treating any
end-to-end golden drift as evidence that the consumer change is wrong,
not that the goldens are wrong.

## Producers

### `pyphi/tpm.py::backward_tpm` (lines 464-498)

- **Returns**: `JointTPM` wrapping ndarray of shape `(*alphabet_sizes, n)`
  (after squeezing background axes to size 1) or `(*alphabet_sizes,
  len(system_indices))` when `remove_background=True`.
- **Semantics**: SBN-form (`forward * pr_current_state / normalization`).
  Trailing axis carries `P(node_i = 1 | s_t) ·
  P(observed-mechanism-state | s_t)`. Trailing axis is load-bearing for
  every binary-path consumer.
- **Direct callers**:
  - `pyphi/core/tpm/marginalization.py:29` (`JointTPM` branch of dispatcher)
  - `pyphi/core/tpm/marginalization.py:32` (Protocol fallback)
  - `pyphi/core/tpm/marginalization.py:59` (`_cause_tpm_factored_binary`)
  - `test/test_core_tpm.py:66` (parity test)

### `pyphi/core/tpm/marginalization.py::cause_tpm` (dispatcher, lines 18-32)

- **Returns**: `CausePosterior` (post-`e85fa73f`) wrapping the underlying
  ndarray.
- **Three wrap sites**:
  - Line 29: `CausePosterior(_legacy_backward_tpm(tpm._inner, ...))` —
    JointTPM branch, ndarray shape `(*alphabet_sizes, n)` (SBN-form).
  - Line 32: `CausePosterior(_legacy_backward_tpm(legacy._inner, ...))` —
    Protocol fallback, same SBN-form.
  - Lines 26-27: dispatch into `_cause_tpm_factored_binary` (SBN) or
    `_cause_tpm_factored_kary` (joint posterior).
- **Semantics asymmetry**: dispatcher hides the fact that the binary
  branch's CausePosterior carries SBN semantics while the k-ary
  branch's CausePosterior carries joint-posterior semantics. Both
  paths' wrapped arrays have different shapes AND different
  mathematical meaning.

### `pyphi/core/tpm/marginalization.py::_cause_tpm_factored_binary` (lines 45-59)

- **Returns**: `CausePosterior` wrapping SBN-form ndarray of shape
  `(*alphabet_sizes, n)`.
- **Method**: stacks `factored.factor(i)[..., 1]` into SBN form, then
  delegates to `_legacy_backward_tpm`.
- **Mathematical semantics**: identical to the legacy path's output.

### `pyphi/core/tpm/marginalization.py::_cause_tpm_factored_kary` (lines 62-82)

- **Returns**: `CausePosterior` wrapping joint posterior of shape
  `(*alphabet_sizes,)`.
- **Method**: per-factor likelihood product over mechanism indices,
  normalized over the full joint past-state space.
- **Mathematical semantics**: `P(s_t | s_{t+1,M} = state)`.
- **Not yet reachable** from any production consumer: binary substrates
  always route to `_cause_tpm_factored_binary`, and k>2 substrates raise
  in `_effect_tpm_factored` (line 94-99) before they can reach this path
  through `Substrate`.

### `pyphi/macro.py::rebuild_system_tpm` (lines 41-51) — secondary producer

- **Returns**: `JointTPM(..., validate=True)` wrapping array stacked from
  per-node `cause_tpm_on` / `effect_tpm_on` slices via
  `np.stack(..., axis=-1)`.
- **Semantics**: re-derives an SBN-form substrate-level TPM from per-node
  on-probability slices. The output is consumed as if it were a forward
  TPM — but for the cause direction this re-derives an SBN-form cause
  TPM (NOT a joint posterior). This is a load-bearing site for
  blackbox/coarse-grain pipelines.
- **Callers**:
  - `pyphi/macro.py:286` (`_squeeze`)
  - `pyphi/macro.py:308` (`_blackbox_partial_noise`)
  - `pyphi/macro.py:99` (`run_tpm`)
  - `test/test_macro.py:304` (parity test)
- **Status**: `MacroSystem.__init__` raises `NotImplementedError` (P7b
  rewrite pending), so macro paths are not currently reachable from
  end-to-end goldens. But the test suite exercises `rebuild_system_tpm`
  directly.

## Consumers

Each entry: file:line · category · operation · downstream · invariance
guess.

### `pyphi/system.py::System.cause_tpm` (lines 160-167)

- **Category C** (reference + unwrap).
- **Operation**: calls `_marginalize_cause(_typed_tpm, state,
  node_indices)`, then strips the typed wrapper via `typed._inner if
  hasattr(typed, "_inner") else typed`.
- **Feeds into**: every other site that reads `system.cause_tpm`,
  notably `Node.__init__` via `System.nodes`.
- **Invariance guess**: pure unwrap — if the wrapped ndarray's shape and
  values change, this site emits the new shape unchanged.

### `pyphi/system.py::System.proper_cause_tpm` (lines 186-189)

- **Category B** (SBN semantics).
- **Operation**: `np.asarray(self.cause_tpm.squeeze())[...,
  list(self.node_indices)]` — slices the trailing axis by substrate node
  index. Under SBN semantics, this selects per-mechanism-node firing
  probability columns. Under joint-posterior shape `(*alphabet_sizes,)`
  there is no trailing axis to index this way.
- **Feeds into**: `proper_cause_tpm` is part of the
  `SystemPublicInterface` Protocol (`protocols.py:175, 276, 370`) and
  is consumed by external API users. Not load-bearing inside the
  repertoire kernel, but a public-API shape contract.
- **Invariance guess**: possible drift — under unification this would
  need to either compute the substrate-restricted joint posterior or
  redefine its semantics.

### `pyphi/system.py::System.nodes` (lines 229-243)

- **Category C** (pass-through to generate_nodes).
- **Operation**: passes the unwrapped `cause_tpm` ndarray into
  `generate_nodes`, which constructs per-`Node` objects via
  `Node.__init__`.
- **Feeds into**: `Node.__init__` (the load-bearing SBN consumer).
- **Invariance guess**: shape-transparent — but `Node.__init__` is the
  load-bearing site, and the unwrap-then-pass pattern means the
  substrate-level shape change propagates downstream verbatim.

### `pyphi/node.py::Node.__init__` (lines 42-105)

- **Category B** (load-bearing SBN semantics).
- **Operation** (line 63): `cause_tpm_on = cause_tpm[..., self.index]`
  reads the SBN trailing axis at the node's substrate index to get a
  scalar slice `P(node_i = 1 | s_t)` of shape `(*alphabet_sizes,)` (after
  the index drop).
- **Operation** (line 72): `cause_tpm.tpm_indices()` reads
  `JointTPM.tpm_indices()` from `joint_distribution.py:245-248`, which
  uses `np.where(shape[:-1] == 2)` — binary-specific.
- **Operation** (line 80): `cause_tpm_off = 1 - cause_tpm_on` — this
  arithmetic assumes binary alphabet AND assumes `cause_tpm_on` is a
  probability (NOT a joint-posterior slice). Under SBN this gives
  `P(node_i = 0 | s_t)`. Under joint-posterior this gives a meaningless
  quantity.
- **Operation** (lines 87-88): stacks off/on along a new trailing axis,
  producing per-node SBN-shaped `(*alphabet_sizes, 2)`. This shape feeds
  every downstream per-node repertoire consumer.
- **Feeds into**: `node.cause_tpm` (per-node TPM) consumed by
  `_single_node_cause_repertoire` (`core/repertoire_algebra.py:130`)
  and by `_single_node_effect_repertoire` when
  `direction == CAUSE` (line 143).
- **Invariance guess**: definite drift if substrate-level cause TPM
  switches to joint-posterior shape. The `[..., self.index]` slice,
  the `1 - x` arithmetic, and the off/on stack ALL depend on SBN
  semantics. This is THE primary site requiring rewrite under
  unification.

### `pyphi/node.py::Node.cause_tpm_off` and `Node.cause_tpm_on` (lines 108-120)

- **Category A** (pure shape — but on the *per-node* TPM, not the
  substrate-level cause TPM).
- **Operation**: `self.cause_tpm[..., 0]` / `[..., 1]` on the per-node
  shape `(*alphabet_sizes, a_node)`.
- **Feeds into**: `macro.run_tpm` (line 85), `macro._blackbox_partial_noise`
  (line 298), `macro._squeeze` (line 286), tests.
- **Invariance guess**: per-node shape is downstream of the
  `Node.__init__` rewrite. If `Node.__init__` constructs a new per-node
  representation, these accessors generalize naturally (`a_node` instead
  of literal `2`).

### `pyphi/core/repertoire_algebra.py::_single_node_cause_repertoire` (lines 122-131)

- **Category B** (per-node trailing-axis indexing).
- **Operation** (line 130): `mechanism_node.cause_tpm[...,
  mechanism_node.state]` indexes the per-node trailing axis (the node's
  own alphabet axis) by scalar state, then `marginalize_out(non_purview
  inputs).tpm`.
- **Feeds into**: `_cause_repertoire_inner` (`core/repertoire_algebra.py:156-172`)
  → `cause_repertoire` → all phi computation. **This is the hot path
  for cause-side small-φ.**
- **Invariance guess**: this site reads the per-node cause TPM, not the
  substrate-level cause TPM. The invariance under unification depends
  on whether `Node.__init__` continues to produce shape `(*alphabet_sizes,
  a_node)` with the same semantic content (per-node conditional
  posterior over the node's alphabet given past state). Under SBN that
  semantic is encoded as `[P(node=0|s_t), P(node=1|s_t)]`, which is
  consistent with the joint-posterior interpretation IF the substrate
  level supplies the right input to `Node.__init__`. This is the
  load-bearing math invariance question.

### `pyphi/core/repertoire_algebra.py::_single_node_effect_repertoire` (lines 134-153)

- **Category C** (cause-direction usage uses `condition_tpm` only).
- **Operation** (line 143): `purview_node.cause_tpm.condition_tpm(condition)`.
  Shape-agnostic.
- **Feeds into**: effect-direction repertoire path when direction==CAUSE
  (used for some symmetric computations).
- **Invariance guess**: likely invariant — operates on a `JointTPM`'s
  `condition_tpm`, not on shape directly.

### `pyphi/macro.py:103` — `convert.state_by_node2state_by_state(system.cause_tpm.tpm)`

- **Category B** (SBN→SBS conversion requires SBN trailing axis).
- **Operation**: SBN→SBS converter requires `shape[-1] == n_nodes` with
  per-node firing probabilities.
- **Feeds into**: `run_tpm` body for time-iterating the noised TPM.
- **Invariance guess**: definite drift — the converter is mathematically
  only meaningful on SBN-form data. Under unification, this site
  either needs an alternative path (joint-posterior → SBS) or must be
  documented as binary+SBN-only.
- **Status**: dead code currently (MacroSystem raises in __init__).

### `pyphi/macro.py:153` — `system.cause_tpm = self.cause_tpm` (SystemAttrs.apply)

- **Category C** (assignment).
- **Operation**: assigns the substrate-level cause TPM container.
- **Feeds into**: the assigned-to object's `cause_tpm` attribute.
- **Invariance guess**: shape-transparent.

### `pyphi/macro.py:266` — `assert system.node_indices == system.cause_tpm.tpm_indices()`

- **Category B** (relies on `tpm_indices()`'s SBN-binary-trailing-axis
  semantics).
- **Operation**: asserts the cause TPM's `tpm_indices()` (computed via
  `np.where(shape[:-1] == 2)`) matches the system's `node_indices`.
- **Feeds into**: invariant check in `_squeeze`.
- **Invariance guess**: definite drift if `tpm_indices()` semantics
  change OR if the trailing axis is removed. Both are likely under
  unification.
- **Status**: dead code currently.

### `pyphi/macro.py:271` — `cause_tpm = remove_singleton_dimensions(system.cause_tpm)`

- **Category B** (`remove_singleton_dimensions` at `macro.py:55-70`
  uses `tpm.squeeze()[..., tpm.tpm_indices()]`, which assumes trailing
  axis is sized by output-node count and indexed by `tpm_indices()`).
- **Operation**: squeezes singleton dims, then re-slices the trailing
  axis by the post-squeeze tpm_indices.
- **Invariance guess**: definite drift — the function's body assumes
  SBN trailing axis. Status: dead code.

### `pyphi/macro.py:286` — `cause_tpm = rebuild_system_tpm(node.cause_tpm_on for node in nodes)`

- **Category B** (`rebuild_system_tpm` stacks per-node ON probabilities
  on a new trailing axis, producing SBN-form).
- **Operation**: rebuilds substrate-level cause TPM in SBN form from
  per-node on-probability slices.
- **Feeds into**: subsequent macro pipeline stages.
- **Invariance guess**: drift — under unification, "rebuild a cause TPM
  from per-node slices" needs a different mathematical operation
  (joint posterior cannot be reconstructed from per-node marginals
  alone because the past nodes are NOT conditionally independent under
  observation).
- **Status**: dead code.

### `pyphi/macro.py:341` — `system.cause_tpm.marginalize_out(blackbox.hidden_indices)`

- **Category C** (shape-agnostic via `JointTPM.marginalize_out`).
- **Operation**: marginalization over the joint distribution.
- **Invariance guess**: likely invariant — operates through the typed
  wrapper.

### `pyphi/macro.py:371` — `coarse_grain.macro_tpm(system.cause_tpm.tpm, check_independence=...)`

- **Category B** (consumes the SBN ndarray; `check_independence` only
  meaningful on forward TPMs).
- **Operation**: applies coarse-graining to the underlying ndarray.
- **Invariance guess**: drift; coarse-graining is defined for SBN
  forward TPMs and the cause-direction usage is suspect even pre-P12b.
- **Status**: dead code.

### `pyphi/actual.py::TransitionSystem.cause_tpm` (lines 244-246)

- **Category C** (pure pass-through to the underlying System).
- **Operation**: `return self._underlying_system.cause_tpm`.
- **Feeds into**: `TransitionSystem.nodes` (line 304), which calls
  `generate_nodes` — same downstream chain as System.
- **Invariance guess**: shape-transparent. The transitive consumer is
  `Node.__init__`.

### `pyphi/actual.py::TransitionSystem.proper_cause_tpm` (lines 263-265)

- **Category C** (pass-through).
- Same shape behavior as `System.proper_cause_tpm`.

### `pyphi/actual.py:303-310` — `generate_nodes(self.cause_tpm, ...)`

- **Category C** (same path as `System.nodes`).

### `pyphi/protocols.py:165, 175, 267, 276, 370`

- **Category C** (Protocol declarations, no math).
- The string `"cause_tpm"` appears in `PUBLIC_SYSTEM_ATTRS` and the
  Protocol class declares `cause_tpm: Any` / `proper_cause_tpm: Any`.
- **Invariance guess**: invariant — type is `Any`. The Protocol does
  not pin shape.

### `pyphi/core/tpm/joint_distribution.py::tpm_indices` (lines 245-248)

- **Category B** (binary-specific helper).
- **Operation**: `tuple(np.where(np.array(self.shape[:-1]) == 2)[0])`.
- **Feeds into**: every site that calls `tpm.tpm_indices()` — most
  notably `Node.__init__` line 72 (cause_non_inputs computation).
- **Invariance guess**: definite drift in k>2 case; even in binary the
  semantics conflate "node axis of size 2" with "axis is a node-input".
  Already flagged in the prior audit.

### Test sites — Category D (pin existing shape)

- `test/test_core_tpm.py:51-67` — `test_cause_tpm_parity`: asserts
  `cause_tpm(JointTPM(joint), state, indices)` matches
  `legacy_backward_tpm(...)` byte-identically. Will fail under
  unification if `cause_tpm` switches to joint-posterior shape.
- `test/test_marginalization_factored.py:14-26` —
  `test_cause_tpm_factored_dispatch_matches_joint`: asserts `cause_tpm`
  on JointTPM and FactoredTPM inputs agree within `atol=1e-10`. Will
  fail in k>2 only (binary still goes through the SBN bridge in both
  branches).
- `test/test_marginalization_factored.py:50-69` — return-type checks
  (`isinstance(result, CausePosterior)`). Shape-agnostic; invariant.
- `test/test_marginalization_kary.py:22-32` — k-ary smoke tests; assert
  sum-to-1 (passes under joint posterior, FAILS under any "wrap k-ary
  in SBN form" alternative; effectively pins the joint-posterior
  semantics for the k-ary path).
- `test/test_node.py:10-141` — `test_node_init_tpm`,
  `test_generate_nodes`, `test_generate_nodes_default_labels`: assert
  per-node `cause_tpm` shape `(*alphabet_sizes, 2)` byte-identically
  against hardcoded answers. Will fail if per-node shape changes.
- `test/test_system.py:131-133` — `test_apply_cut`: `np.array_equal(
  cut_s.cause_tpm.tpm, s.cause_tpm.tpm)`. Shape-transparent.
- `test/test_macro.py:301-306, 363, 380` — assert macro pipeline
  preserves substrate-level `cause_tpm` shape. Status: dead code (macro
  pipeline raises NotImplementedError).
- `test/test_tpm.py:115` — `assert s.cause_tpm == s.effect_tpm` (basic
  network has symmetric TPM). Shape-transparent.

## Producer-consumer graph

```
[FactoredTPM (substrate)]
    │
    ├──→ _cause_tpm_factored_binary ──→ stack to SBN ──→ _legacy_backward_tpm
    │                                                         │
    │                                                  CausePosterior
    │                                                  (SBN-form, shape *α, n)
    │
    └──→ _cause_tpm_factored_kary ──→ per-factor product ──→ CausePosterior
                                                              (joint posterior,
                                                              shape *α)
                                                              UNREACHABLE today

[JointTPM-only callers, Protocol fallback]
    └──→ _legacy_backward_tpm ──→ CausePosterior (SBN-form)

                              │
                              ▼
              System.cause_tpm  (unwrap typed wrapper)
                              │
                              ▼
              ┌───────────────┴────────────────────────────────────┐
              │                                                    │
              ▼                                                    ▼
        proper_cause_tpm                                  System.nodes
        (slice [..., node_indices])                            │
              │                                                ▼
              ▼                                       generate_nodes
        Public API surface,                                    │
        protocols.py                                           ▼
                                                       Node.__init__
                                                       (line 63: [..., self.index])
                                                       (line 72: cause_tpm.tpm_indices())
                                                       (line 80: 1 - cause_tpm_on)
                                                       (lines 87-88: stack off/on)
                                                              │
                                                              ▼
                                                       node.cause_tpm
                                                       shape (*α, a_node)
                                                              │
                                              ┌───────────────┴──────────────┐
                                              ▼                              ▼
                              _single_node_cause_repertoire        _single_node_effect_repertoire
                              (line 130: [..., state])             (line 143: condition_tpm)
                                              │                              │
                                              ▼                              ▼
                                      _cause_repertoire_inner       _effect_repertoire_inner
                                              │                              │
                                              ▼                              ▼
                                      cause_repertoire             effect_repertoire
                                              │                              │
                                              ▼                              ▼
                                            φ_cause                       φ_effect
                                                          │
                                                          ▼
                                                        small-φ / SIA / Φ

[Secondary dead-code branches under MacroSystem (raises NotImplementedError):
 rebuild_system_tpm, run_tpm, _squeeze, _blackbox_partial_noise,
 _coarsegrain_space — all consume/produce SBN-form substrate cause TPM.]
```

## Sites that warrant especially careful migration

1. **`pyphi/node.py::Node.__init__` (lines 63, 72, 80, 87-88).** Four
   distinct dependencies on SBN semantics in 25 lines:
   `cause_tpm[..., self.index]` (per-node slice from SBN trailing axis),
   `cause_tpm.tpm_indices()` (binary-axis-size heuristic),
   `1 - cause_tpm_on` (Bernoulli complement, assumes binary alphabet
   AND assumes the slice is a probability not a joint-posterior chunk),
   and the off/on stack. The per-node output shape `(*α, a_node)` feeds
   every downstream cause-side computation; any unification has to
   produce a semantically equivalent per-node representation here OR
   refactor the per-node downstream path. This is the single
   highest-risk site.

2. **`pyphi/core/repertoire_algebra.py::_single_node_cause_repertoire`
   (line 130) + `_cause_repertoire_inner` (lines 156-172).** The hot
   path for cause-side small-φ. Indexes the per-node trailing axis by
   `mechanism_node.state`, then multiplies single-node cause repertoires
   into a joint cause repertoire. The math invariance question that
   Phase 1's algebraic derivation has to settle: under what conditions
   does `∏_i (single-node-cause-repertoire derived from SBN)` equal the
   marginalized joint-posterior? If they're equal up to normalization,
   unification is goldens-stable. If not, this is where drift appears.

3. **`pyphi/system.py::proper_cause_tpm` (lines 186-189).** The slice
   `np.asarray(cause_tpm.squeeze())[..., list(self.node_indices)]`
   relies on the substrate-level SBN trailing axis being indexable by
   substrate node index. Under joint posterior the slice has no
   meaning. Public API surface (Protocol-declared at `protocols.py:175,
   276, 370`). Possible drift even if internal computations stay
   invariant — the public attribute would need a re-definition or
   removal.

## Open questions for Phase 1 math analysis

1. **Is `∏_i (Bernoulli-cause-rep derived from SBN slice)` equal to the
   joint posterior `P(s_t | s_{M,t+1} = state)` for binary substrates,
   up to normalization?** If yes, the unified consumer pipeline can
   read the joint posterior directly without changing the downstream
   φ values. If no, the unification introduces real numerical drift
   and the goldens-invariance bar is unattainable.

2. **What is the per-node decomposition of the joint posterior?** Under
   SBN semantics, `Node.__init__` projects the substrate-level array
   onto each node by indexing the trailing axis. Under joint posterior,
   the equivalent per-node quantity is `marginal of P(s_t | s_{M,t+1})
   onto node i`. Is the existing `_single_node_cause_repertoire(...) *
   ... product` equivalent to the marginalized joint posterior,
   modulo normalization? Specifically: does the `marginalize_out` step
   on line 131 correctly extract the right marginal from the per-node
   SBN-slice representation?

3. **Does the `proper_cause_tpm` Protocol attribute have a
   joint-posterior counterpart with consistent semantics?** This is a
   public-API design question, not a math question per se. The
   migration plan needs to decide whether to redefine, deprecate, or
   replace this attribute.

4. **`tpm_indices()` semantics in the joint-posterior regime.** The
   current implementation conflates "node axes" with "size-2 axes".
   Under joint posterior with shape `(*α,)`, all axes are node axes
   and there's no trailing alphabet axis. Need a clean replacement
   (e.g., `tuple(range(ndim))`) and a check for callers that depend on
   the binary-trailing-axis exclusion behavior.

5. **Macro pipeline status.** The `MacroSystem.__init__` currently
   raises `NotImplementedError`. The dead-code SBN-consuming sites in
   `macro.py` are dead pending the P7b rewrite. Phase 1 should
   decide whether the unification must preserve `rebuild_system_tpm`
   semantics (which only makes sense for SBN-form forward TPMs) or
   whether the macro rewrite can adopt the joint-posterior cause
   representation natively.
