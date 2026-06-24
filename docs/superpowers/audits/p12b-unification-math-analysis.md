# P12b unification math analysis

This audit derives the algebra of the two cause-TPM paths in PyPhi 2.0,
traces the canonical consumer chain end-to-end, verifies the algebra
against the live code with a concrete 3-node binary network, and
categorizes each category B consumer from
`p12b-sbn-consumer-catalogue.md` per its invariance under the proposed
unification (substrate-level cause representation switches from SBN-form
to joint-posterior). The verdict drives the re-spec and re-plan
decisions for the cause-side unification.

## 1. The two paths' math (re-derived from code)

### 1.1 Legacy SBN-form (`_legacy_backward_tpm`)

`pyphi/tpm.py::backward_tpm` (lines 464-498), parametrized by an SBN-form
forward TPM `forward[s_t, i] = P(node_i,t+1 = 1 | s_t)` of shape `(*α,
n)`, the full substrate state `current_state` of length `n`, and the
mechanism `system_indices`.

The body computes, with all `s_t` ranging over `(*α,)`:

```
pr_current_state[s_t]  =  ∏_i  pr_node_i(s_t)            (probability_of_current_state, line 461)
   where  pr_node_i(s_t) = forward[s_t, i]               if current_state[i] == 1
                         = 1 - forward[s_t, i]           otherwise
   shape  (*α, 1)                                        (.prod(axis=-1, keepdims=True))
```

`pr_current_state` is the joint Bernoulli likelihood `P(s_{t+1} =
current_state | s_t)` under the conditional-independence-given-past
factorization. Note `.prod(axis=-1, keepdims=True)` — the trailing axis
is collapsed to size 1, not `n`.

```
pr_bg[s_t]  =  pr_current_state.sum(axis=system_indices, keepdims=True)
            =  Σ_{s_{M,t}} pr_current_state(s_{M,t}, s_{W,t})
   where  M = system_indices, W = background_indices
   shape  (1,...,1, |α_w_1|, ..., |α_w_k|, 1)            (system axes collapsed)
```

This is the marginal likelihood over system past, leaving background past
free.

```
norm  =  pr_current_state.sum()                          scalar
      =  Σ_{s_t} ∏_i pr_node_i(s_t)
      =  P(s_{t+1} = current_state)                       under the maxent prior over s_t
```

The normalizer sums the joint likelihood across all past states.

```
out[s_t, i]  =  (forward * pr_bg / norm).sum(axis=background, keepdims=True)
   shape  (|α_m_1|, ..., |α_m_k|, 1, ..., 1, n)           (background axes collapsed)
```

The trailing axis carries `n`, sized by the number of substrate nodes.
After `remove_background=True` the trailing axis becomes
`len(system_indices)`.

#### What `out[..., i]` represents

For mechanism node `i` (i.e. `i ∈ system_indices`), `out[s_{M,t}, i]` is

```
out[s_{M,t}, i]  =  Σ_{s_{W,t}}  forward[s_t, i] · pr_bg[s_t] / norm
                 =  Σ_{s_{W,t}}  P(node_i,t+1 = 1 | s_t) · P(s_{W,t+1} = obs_W | s_t) / Z
```

i.e. "the probability that node `i` will fire in `t+1`, weighted by how
likely the observed background was, summed over background past". This
quantity is **not** itself a probability over past states — its trailing
axis sums to `n` (one value per substrate node), not 1.

It carries enough Bernoulli information that the consumer chain (per-node
slice plus Bernoulli flip plus per-node-input marginalize plus product
plus global normalize) recovers the correct cause repertoire — see
sections 2 and 3.

#### Subtle but critical: the normalizer

The Phase 0 catalogue raised the question of whether `norm` uses a
per-node sum or a joint product. The code answer: **`norm` is the scalar
joint product summed across past states**, not a per-node sum across
nodes. The product-over-nodes lives in `probability_of_current_state`
(line 461) where `.prod(axis=-1, keepdims=True)` runs before the
`.sum()`. So the normalizer is `Σ_{s_t} ∏_i pr_node_i(s_t)` — the
expected joint likelihood. No "per-node-count scaling" pathology is
introduced.

### 1.2 Native joint-posterior (`_cause_tpm_factored_kary`)

`pyphi/core/tpm/marginalization.py::_cause_tpm_factored_kary` (lines
62-82), parametrized by a `FactoredTPM`, a state of length `n`
(full-substrate), and `node_indices` (the mechanism). Body:

```
likelihood[s_t]  =  ∏_{i ∈ node_indices}  factor_i(s_t)[state[i]]
posterior[s_t]   =  likelihood[s_t] / likelihood.sum()
   shape  (*α,)                                          (no trailing node axis)
```

`factor_i(s_t)[state[i]] = P(node_i,t+1 = state[i] | s_t)`, exactly the
Bernoulli factor for node `i`. The likelihood is multiplied only across
mechanism nodes (`i ∈ node_indices`), not background nodes. This is

```
posterior[s_t]  =  P(s_t | s_{M,t+1} = state_M)
```

— the **joint past posterior conditioned on mechanism observations
alone**. Background nodes' observed states (`state[j]` for `j ∈
background`) are not used.

### 1.3 The mathematical asymmetry

The legacy SBN form conditions on `current_state` for ALL substrate
nodes (system AND background), entering through `pr_current_state =
∏_{i=0..n-1} pr_node_i`. The native k-ary form conditions only on
mechanism nodes (`i ∈ node_indices`). This is the same difference that
the catalogue flagged as "unreachable today" — binary substrates always
route to `_cause_tpm_factored_binary` (which uses `_legacy_backward_tpm`,
conditioning on full state), so the k-ary divergence has never affected
goldens. But for the unification, the choice matters: a naive "use k-ary
form everywhere" path would produce a different mathematical object for
any subset system (one with non-empty background).

## 2. Canonical consumer chain — algebraic trace

The end-to-end path from substrate-level `cause_tpm` to final cause
repertoire used in φ:

```
System.cause_tpm  →  (unwrap)  →  generate_nodes  →  Node.__init__
                                                    →  node.cause_tpm  (per-node shape (*α, a_node))
                                                       ↓
                                       _single_node_cause_repertoire (slice by state, marginalize_out)
                                                       ↓
                                       _cause_repertoire_inner (product over mechanism, normalize)
                                                       ↓
                                                  cause_repertoire
```

### 2.1 `Node.__init__` (`pyphi/node.py:42-105`)

Given substrate-level `cause_tpm` of shape `(*α, n)` (SBN-form):

```
cause_tpm_on   =  cause_tpm[..., self.index]                          # shape (*α,)
cause_tpm_on   =  cause_tpm_on.marginalize_out(cause_non_inputs).tpm   # average over non-input axes, keepdims
cause_tpm_off  =  1 - cause_tpm_on                                    # Bernoulli flip
self.cause_tpm =  JointTPM(stack([cause_tpm_off, cause_tpm_on], -1))  # shape (*α, 2)
```

The per-node TPM has size 1 on every past axis that is NOT an input to
this node, size 2 on input axes, and size 2 on the trailing alphabet
axis. The values represent `P(node_i = on/off | s_t)` under the SBN
weighting.

Under the proposed joint-posterior unification, `cause_tpm` would have
shape `(*α,)` and the four SBN-dependent operations all break:

| Operation                                          | SBN semantic                                          | Joint-posterior counterpart                                              |
|----------------------------------------------------|-------------------------------------------------------|--------------------------------------------------------------------------|
| `cause_tpm[..., self.index]`                       | Slice trailing axis to get this node's Bernoulli      | No trailing axis exists; this slice is undefined                          |
| `cause_tpm.tpm_indices()` (line 72)                | `np.where(shape[:-1] == 2)` — binary-trailing-axis    | All axes are past-state axes; `tuple(range(ndim))` would be the analog    |
| `1 - cause_tpm_on`                                 | Bernoulli flip, valid because slice is a probability  | Joint posterior is summed over `s_node`, not a per-node probability       |
| `np.stack([off, on], axis=-1)`                     | Reassembles a per-node TPM                            | Joint posterior never separates per-node Bernoulli info                   |

This site requires a full rewrite. The natural rewrite reads per-node
factors directly from the upstream `FactoredTPM`, not from a substrate-
level "cause TPM" object. The substrate-level cause TPM ceases to be the
right intermediate abstraction for the consumer chain.

### 2.2 `_single_node_cause_repertoire`
(`pyphi/core/repertoire_algebra.py:122-131`)

```
tpm  =  mechanism_node.cause_tpm[..., mechanism_node.state]              # shape (*α,), per-node slice
return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm      # average over (inputs ∩ non-purview)
```

The slice picks the per-node-state Bernoulli value. The marginalize_out
averages over those of node `m`'s inputs that are not in the purview —
this is the IIT prescription for "irrelevance of non-purview inputs"
(uniform prior, take expectation).

Under joint-posterior unification, the natural rewrite is:

```
factor_m(s_t)  =  factored.factor(m)[..., state[m]]                      # P(node_m = state[m] | s_t)
factor_m       =  apply background weighting (Sec. 3)
factor_m       =  factor_m.marginalize_out(mech_node.inputs - purview)   # same average step
return factor_m
```

The marginalize_out step is identical in shape/semantics. The slice-by-
state step is replaced by reading the appropriate slice of the FactoredTPM's
per-node factor, with the background-state weighting that the legacy
chain bakes into the substrate-level cause_tpm.

### 2.3 `_cause_repertoire_inner`
(`pyphi/core/repertoire_algebra.py:156-172`)

```
joint  =  np.ones(repertoire_shape(node_indices, purview_set))
joint *= ∏ _single_node_cause_repertoire(cs, m, purview_set) for m in mechanism
return _dist.normalize(joint)
```

Pure structural product + final normalize. This is invariant under any
consistent producer+consumer rewrite — its only inputs are the per-node
single-node repertoires and the target purview shape. If the per-node
results are correct, this step is correct.

## 3. Concrete 3-node binary verification

Network: `pyphi.examples.basic_system()` (3-node binary, state
`(1, 0, 0)`, fully reachable). Verified across 76 (subsystem, mechanism,
purview) combinations.

### 3.1 Algorithm under joint-posterior unification

For a candidate system `sub` with `node_indices = M`, `external_indices =
W = substrate \ M`, mechanism `mech ⊆ M`, purview `purv ⊆ M`:

```
forward[s_t, i] = P(node_i = 1 | s_t)  for all substrate nodes i

# Background weighting (preserves legacy SBN background-conditioning)
pr_node_i(s_t) = forward[s_t, i] if state[i] == 1 else 1 - forward[s_t, i]
pr_joint(s_t)  = ∏_{i=0..n-1} pr_node_i(s_t)
pr_bg(s_t)     = Σ_{s_{M,t}} pr_joint(s_t)                               # shape collapses M axes
norm           = pr_joint.sum()                                          # scalar

# Per mechanism node m:
bern_m(s_t)    = pr_node_m(s_t)
weighted_m     = bern_m · pr_bg / norm
weighted_m     = sum over W axes                                         # collapse background past
weighted_m     = average over (m's inputs \ purv) ∪ (non-inputs of m)    # marginalize_out

joint          = ∏_{m ∈ mech} weighted_m
result         = joint / joint.sum()                                     # _dist.normalize
result         = broadcast to repertoire_shape(node_indices, purv)
```

### 3.2 Empirical result

Across 76 (sub, mech, purv) cases on the basic_system network
(subsystems `(0,1,2)`, `(0,1)`, `(0,2)`, `(1,2)`, all admissible
mechanism/purview combinations), the unified-path output matches the
legacy `cause_repertoire(cs, mech, purv)` to within numerical tolerance
`1e-10` on every case.

### 3.3 What changes in the consumer chain

The substrate-level cause TPM disappears as an intermediate abstraction.
The consumer chain reads:

1. `FactoredTPM.factor(m)` per mechanism node (already exists, already
   shape `(*α, a_m)`)
2. `state[m]` to slice the trailing alphabet axis
3. The background-weighting machinery `pr_bg / norm` (currently computed
   once inside `_legacy_backward_tpm`; would need to be computed once
   per System and cached)
4. `Node.inputs` and the purview to drive `marginalize_out`
5. Product + global normalize

This is a refactoring that moves the substrate-level "cause TPM"
computation INTO the per-node consumer chain. The result is the same
math; the data flow is reorganized.

## 4. Per-consumer invariance categorization

Each category B consumer from
`p12b-sbn-consumer-catalogue.md`, with its invariance class under the
proposed unification.

### `pyphi/node.py::Node.__init__` (lines 63, 72, 80, 87-88)

- **Invariance class**: NOT INVARIANT under "naive joint-posterior wrap"
  (i.e. swapping in a `(*α,)` posterior with no other changes). All four
  SBN dependencies break.
- **Invariant under co-rewrite**: YES, under the rewrite of Sec. 3.3.
  The per-node TPM construction moves to reading
  `FactoredTPM.factor(m)[..., state[m]]` and applying background
  weighting + per-node marginalize_out at construction time.
- **Resolution**: rewrite `Node.__init__` to take the upstream
  `FactoredTPM` (or the per-node factor directly) plus the background-
  weighting tensor, rather than the substrate-level cause TPM.

### `pyphi/core/repertoire_algebra.py::_single_node_cause_repertoire` (line 130)

- **Invariance class**: INVARIANT, conditional on `Node.__init__`'s
  rewrite producing a per-node TPM with the same shape and semantic
  content. The slice-by-state + marginalize_out operations are
  shape-agnostic.
- **Resolution**: no rewrite needed at this site. It reads
  `mechanism_node.cause_tpm[..., mechanism_node.state]` and that contract
  is preserved by `Node.__init__`'s rewrite.

### `pyphi/system.py::proper_cause_tpm` (lines 186-189)

- **Code**: `np.asarray(self.cause_tpm.squeeze())[..., list(self.node_indices)]`.
- **Invariance class**: NOT INVARIANT. The trailing-axis slice has no
  meaning on a `(*α,)` joint posterior — the joint posterior has no
  "per-substrate-node" trailing axis to subscript.
- **Public-API impact**: `proper_cause_tpm` is declared in the
  `SystemPublicInterface` Protocol (`protocols.py:175, 276, 370`). Internal
  callers do not use this attribute in the small-φ hot path.
- **Resolution options**:
  - **Redefine** as "the substrate-restricted SBN-form cause TPM,
    computed on demand from `FactoredTPM`" (a derived, non-cached
    re-computation of the legacy formula for binary substrates only).
  - **Deprecate** with a clear migration path.
  - **Remove** from the Protocol if no external user relies on it.
- **Recommendation**: redefine as a binary-only convenience derived
  on-demand from `FactoredTPM`, marked clearly as a binary-only
  visualization aid. The unification then preserves the public-API
  shape for binary callers and raises for k-ary callers (consistent
  with the broader k-ary-effect-TPM `NotImplementedError`).

### `pyphi/core/tpm/joint_distribution.py::tpm_indices` (lines 245-248)

- **Code**: `tuple(np.where(np.array(self.shape[:-1]) == 2)[0])`.
- **Invariance class**: NOT INVARIANT. The helper conflates "node-input
  axis" with "size-2 axis". Under joint-posterior shape `(*α,)`, all
  axes are past-state axes and the size-2 heuristic is meaningless.
- **Resolution**: replace with `tuple(range(ndim))` for the joint-
  posterior representation. Audit every caller — primary caller is
  `Node.__init__` (line 72), which under the rewrite no longer uses
  `tpm_indices` on the substrate-level cause TPM.

### `pyphi/macro.py:103, 266, 271, 286, 371` (macro pipeline)

- **Status**: dead code (`MacroSystem.__init__` raises
  `NotImplementedError`).
- **Invariance class**: NOT INVARIANT. All five sites assume the
  SBN-form substrate cause TPM (SBN→SBS conversion at line 103,
  `tpm_indices` assertion at 266, SBN-trailing-axis squeeze at 271,
  per-node `cause_tpm_on` stack at 286, coarse-grain at 371).
- **Resolution**: defer. The macro rewrite is its own milestone and
  will need to choose its own substrate cause-TPM representation. The
  unification does not need to preserve the SBN form for these dead
  sites.

### `pyphi/macro.py::rebuild_system_tpm` (lines 41-51)

- **Status**: dead code (no live caller after `MacroSystem.__init__`
  raises).
- **Invariance class**: NOT INVARIANT. Stacks per-node `cause_tpm_on`
  slices into a new SBN-form substrate cause TPM. Under joint posterior,
  the substrate cause TPM cannot be reconstructed from per-node
  marginals alone (past nodes are not conditionally independent under
  observation).
- **Resolution**: defer to macro rewrite milestone.

### `pyphi/protocols.py:165, 175, 267, 276, 370`

- **Invariance class**: INVARIANT. Protocol declarations type the
  attributes as `Any`. The Protocol does not pin shape or semantics.
- **Resolution**: none; documentation update only if `proper_cause_tpm`
  semantics are redefined.

### Test sites

- `test/test_core_tpm.py:51-67` (`test_cause_tpm_parity`): byte-identical
  assertion between `cause_tpm(JointTPM, ...)` and `_legacy_backward_tpm`.
  Under unification this test changes contract: the new dispatcher
  produces joint-posterior shape `(*α,)`, not SBN-form `(*α, n)`. The
  test must be rewritten to assert against the joint-posterior formula
  (and a separate parity test verifies the SBN-bridge fallback is
  removed). **Will fail under unification unless rewritten.**
- `test/test_marginalization_factored.py:14-26` (`test_cause_tpm_factored_dispatch_matches_joint`): asserts binary and JointTPM
  paths agree. Becomes vacuous if both branches produce the same joint-
  posterior shape directly. **Update or retire.**
- `test/test_marginalization_kary.py:22-32`: pins joint-posterior
  sum-to-1 for k-ary. **Invariant** — joint-posterior unification
  matches this expectation.
- `test/test_node.py:10-141`: pins per-node `cause_tpm` shape
  `(*α, 2)` byte-identically. Under the unified `Node.__init__`
  rewrite, the per-node shape is preserved (the per-node TPM still
  encodes Bernoulli on/off in a trailing axis of size 2 for binary
  alphabets, or size `a_node` for k-ary). The byte-identical values are
  ALSO preserved if the background weighting is faithfully replicated
  (Sec. 3.2 verifies). **Invariant** under faithful rewrite.
- `test/test_system.py:131-133`, `test/test_tpm.py:115`: shape-
  transparent. **Invariant.**
- `test/test_macro.py:301-306, 363, 380`: dead code; macro rewrite owns.

## 5. Verdict: feasibility of byte-identical goldens

**Mostly yes, with one specific risk site and one public-API design
decision.**

The end-to-end cause repertoire (and therefore every downstream phi /
SIA / Φ goldens) is byte-identical under the unification IF the
consumer chain rewrite faithfully replicates the background-weighting
machinery that `_legacy_backward_tpm` bakes into the substrate-level
SBN-form cause TPM. Section 3 verifies this empirically across 76
(sub, mech, purv) cases on the basic_system network.

The math is not a single-shot "wrap the joint posterior" substitution.
The current `_cause_tpm_factored_kary` is mathematically the WRONG
formula for the unified path — it omits background-state conditioning
that is structurally present in the legacy SBN computation. Naively
routing all paths through `_cause_tpm_factored_kary` would silently
corrupt every subset-system computation (cause repertoires, small-φ,
distinctions, Φ_s).

### Specific risk sites

1. **`Node.__init__` (`pyphi/node.py:42-105`).** The load-bearing
   consumer site. Rewrite must preserve per-node cause TPM byte
   identity. The verified algorithm (Sec. 3.1) gives the recipe; the
   rewrite is mechanical. Risk: subtle off-by-one in the background-
   marginalize step, missing keepdims, or wrong axis ordering. **High-
   priority site for parity testing.**

2. **The `tpm_indices()` heuristic** is binary-and-SBN-specific and
   leaks shape semantics through `Node.__init__` line 72. Replacing it
   with `tuple(range(ndim))` (joint-posterior shape) requires checking
   that no other caller depends on the size-2 filter behavior. The
   catalogue already flagged this site.

3. **`_cause_tpm_factored_kary` math correction.** Currently k-ary
   substrates would produce wrong cause repertoires for any subset
   system. The native k-ary path needs to incorporate the background
   weighting (Sec. 3.1's `pr_bg / norm`) before any non-binary substrate
   touches a candidate system with non-trivial external nodes. This is
   not strictly part of the unification but is uncovered by it.

### Public-API design decision

`proper_cause_tpm` has no natural counterpart under joint-posterior
semantics. The decision to redefine (binary-only convenience), deprecate
(with migration), or remove (from Protocol) must be made before the
unification ships. See Sec. 4 for the recommendation.

### Test-suite expected updates

Two parity tests in `test/test_core_tpm.py` and
`test/test_marginalization_factored.py` need to be rewritten or
retired. Their replacements should pin the joint-posterior contract
explicitly. All other tests (including per-node `cause_tpm` byte-
identity, full-system regression goldens, and the macro-pipeline-dead
test stubs) are invariant under faithful rewrite.

## 6. Open questions for the re-spec / re-plan

1. **`proper_cause_tpm` Protocol attribute.** Public-API surface
   declared in `SystemPublicInterface` (`protocols.py:175, 276, 370`).
   Decision needed: redefine as binary-only convenience, deprecate with
   migration path, or remove from Protocol. Recommendation in Sec. 4.

2. **`node.cause_tpm_on` / `cause_tpm_off` accessors.** Currently
   defined on the per-node Bernoulli TPM (which is binary-only). Under
   unification, are these renamed to a more general
   `node.cause_marginal(state)` form? Or kept as binary-only
   convenience? `pyphi/macro.py` accesses them but is dead code.

3. **Background-weighting machinery placement.** The `pr_bg / norm`
   tensor is computed once per System (depends on `state` and CM, not on
   mechanism / purview). Should it live as a `System.cause_background_weight`
   cached property, or be recomputed inside the consumer? Cached
   property reduces redundant work in the φ hot path. Recommend caching
   on `System`.

4. **`_cause_tpm_factored_kary` correctness fix.** The current native
   k-ary path omits background conditioning. Should the fix land
   alongside the unification, before, or as a separate milestone?
   Recommend alongside: the k-ary path is unreachable today, but the
   unification will reach it, and a silently-wrong path is a worse
   liability than a deferred fix.

5. **`tpm_indices()` semantics in the joint-posterior regime.**
   Replace with `tuple(range(ndim))` for joint posteriors, retain the
   binary-trailing-axis form for legacy SBN consumers (if any remain).
   Audit callers; the `Node.__init__` caller goes away under the rewrite.

6. **Macro pipeline.** Five SBN-dependent sites in `pyphi/macro.py`
   are dead code today. The unification does not need to preserve their
   semantics. The macro rewrite milestone owns the replacement.

7. **Per-node-vs-substrate-cache trade-off.** The substrate-level
   `cause_tpm` is currently cached as a single tensor. Under the
   unification, the per-node Bernoulli factors are read directly from
   `FactoredTPM` and the background weighting is computed once per
   System. The substrate-level cause TPM may no longer be a useful cache.
   Profile to confirm before removing the cache contract.
