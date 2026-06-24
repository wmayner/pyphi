# P12b cause-output shape audit

Foundational audit for the multi-valued (k-ary) units work. Records
the actual shape contract of the substrate-level and per-node cause
TPM that the current code produces and that downstream consumers
depend on. Settles whether the native k-ary `CausePosterior` should
carry a trailing node-axis or use pure `alphabet_sizes` shape.

All shapes below are stated for a binary substrate with `n` nodes.
Generalising to k-ary alphabets is the entire point of the question:
the audit decides which generalisation to commit to.

## Legacy `backward_tpm` output shape

Source: `pyphi/tpm.py:698-732`.

Signature:

```python
def backward_tpm(
    forward_tpm: JointTPM,
    current_state: tuple[int, ...],
    system_indices: Iterable[int],
    remove_background: bool = False,
) -> JointTPM:
```

The input `forward_tpm` is a `JointTPM` wrapping a multidimensional
state-by-node array of shape `(2,)*n + (n,)`: `n` axes indexed by
previous-state node values, and a final axis of size `n` indexed by
the node whose next-state probability is stored at that position.
This is the convention encoded in `JointTPM.number_of_units`
(`pyphi/tpm.py:319-323`), which returns `self._tpm.shape[-1]` for
multidimensional state-by-node arrays.

The body works as follows:

1. `probability_of_current_state(forward_tpm, current_state)` (line
   714, definition at `pyphi/tpm.py:673-695`) collapses the trailing
   node-axis into a singleton by computing
   `state_probabilities.prod(axis=-1, keepdims=True)`. Result shape:
   `(2,)*n + (1,)`.
2. `pr_current_state_given_only_background` sums over the system
   axes with `keepdims=True` (line 717). Shape: still
   `(2,)*n + (1,)` but with size-1 system axes.
3. The factor `forward_tpm * pr_current_state_given_only_background
   / normalization` broadcasts against the forward TPM, restoring the
   trailing `n` axis. After `.sum(axis=background_indices,
   keepdims=True)` the shape is `(2 or 1,)*n + (n,)` — system axes
   keep size 2, background axes are size 1.
4. If `remove_background=True`, line 731 does `backward_tpm[...,
   list(system_indices)]`, which slices the final node-axis down to
   size `len(system_indices)`.

Net result: the returned `JointTPM` always retains a trailing axis
indexed by *output nodes*. With `remove_background=False` that axis
has size `n`; with `remove_background=True` it has size
`len(system_indices)`. In neither case does the multidimensional
state-by-node convention get dropped.

Concrete example for the `examples.basic_substrate()` regression in
`test/test_core_tpm.py:51-67` (binary `n=3`, system_indices =
`(0,1,2)`): the output shape is `(2, 2, 2, 3)` —
`(2,)*n + (n,)`.

`pyphi/core/tpm/marginalization.py:22-66` calls
`_legacy_backward_tpm(joint._inner, state, node_indices)` and wraps
the result as `JointTPM`. The factored fast-path
`_cause_tpm_factored` reconstructs an SBN view via `np.stack([...,
1] for i in range(n))` then delegates to the same legacy function —
so its output shape is identical: `(2,)*n + (n,)`.

## Downstream consumer shape contract

### `pyphi/node.py:42-93` (`Node.__init__` and `generate_nodes`)

`Node.__init__` receives the substrate-level `cause_tpm` (a
`JointTPM` wrapping the array described above) and projects out the
column for this node:

```python
cause_tpm_on = cause_tpm[..., self.index]
```

This indexes the trailing node-axis with the node's substrate
index. The shape contract is hard: the `cause_tpm` *must* have a
trailing axis whose length is at least `self.index + 1` and whose
semantics is "next-state probability for node index `i`".

After marginalising non-input axes, `cause_tpm_on` has shape
`(2,)*n` (with size-1 axes for marginalised positions). The
constructor then stacks the per-node off/on probabilities along a
new trailing axis:

```python
self.cause_tpm = JointTPM(np.stack([cause_tpm_off, cause_tpm_on], axis=-1))
```

so the **per-node** `node.cause_tpm` has shape `(2,)*n + (2,)`. The
trailing axis is the node's own alphabet (size 2 for binary). This
generalises naturally to k-ary alphabets as `(*alphabet_sizes,
a_node)` where `a_node` is the alphabet size of the output node.

### `pyphi/core/repertoire_algebra.py:122-153`

```python
@_memoize
def _single_node_cause_repertoire(cs, mechanism_node_index, purview_set):
    mechanism_node = cs._index2node[mechanism_node_index]
    tpm = mechanism_node.cause_tpm[..., mechanism_node.state]
    return tpm.marginalize_out(mechanism_node.inputs - purview_set).tpm
```

`mechanism_node.cause_tpm[..., mechanism_node.state]` indexes the
**per-node** trailing axis (the alphabet axis for that node) with
the scalar `mechanism_node.state`. So the contract this consumer
relies on is the *per-node* shape `(*alphabet_sizes, a_node)`. It
does **not** index the substrate-level trailing axis.

`_single_node_effect_repertoire` (lines 134-153) calls
`purview_node.cause_tpm.condition_tpm(condition)` —
shape-agnostic, operates on the per-node TPM.

### `pyphi/macro.py`

- Line 103: `convert.state_by_node2state_by_state(system.cause_tpm.tpm)`
  consumes the trailing-`n` axis (the SBN→SBS converter requires it).
- Line 153: `system.cause_tpm = self.cause_tpm` — substrate-shaped
  assignment.
- Line 266: `assert system.node_indices == system.cause_tpm.tpm_indices()`.
  `JointTPM.tpm_indices()` (`pyphi/tpm.py:503-506`) is
  `tuple(np.where(np.array(self.shape[:-1]) == 2)[0])` — assumes a
  trailing axis distinct from the state axes.
- Line 286: `rebuild_system_tpm(node.cause_tpm_on for node in nodes)`
  rebuilds a substrate-level cause TPM by stacking per-node off/on
  columns, producing `(2,)*n + (n,)`.
- Line 341: `system.cause_tpm.marginalize_out(blackbox.hidden_indices)` —
  shape-preserving.
- Line 371: `system.cause_tpm.tpm` consumed by code that expects the
  full multidimensional SBN array (trailing node-axis included).

### `pyphi/system.py:155-243`

- `cause_tpm` returns the unwrapped ndarray from the typed wrapper.
  Stripped via `typed._inner if hasattr(typed, "_inner") else typed`
  (line 167).
- `proper_cause_tpm` (line 186-189) does
  `np.asarray(self.cause_tpm.squeeze())[..., list(self.node_indices)]`
  — explicit reliance on a trailing node-axis indexed by substrate
  node index.
- `nodes` (line 228-243) passes the unwrapped `cause_tpm` array into
  `generate_nodes`, which then performs the
  `cause_tpm[..., self.index]` slicing described above.

### Test sites

`test/test_node.py`, `test/test_tpm.py`, `test/test_macro.py`,
`test/test_system.py`, `test/test_core_tpm.py` all consume the
substrate-level cause TPM through one of two channels: passing it to
`generate_nodes` (which indexes the trailing axis) or comparing it
to a legacy reference via `.array_equal` / `np.testing.assert_array_equal`.
Both channels assume the `(2,)*n + (n,)` shape.

## `System.cause_tpm` consumers — summary

The substrate-level cause TPM shape `(2,)*n + (n,)` is depended on
by every external consumer. The trailing axis is **not** an
implementation detail of `backward_tpm` — it carries the
"per-output-node next-state probability" semantics that
`Node.__init__` projects on (`cause_tpm[..., self.index]`) and that
`proper_cause_tpm` slices on (`[..., list(self.node_indices)]`).

The per-node cause TPM constructed by `Node.__init__` has a
different but related shape: `(2,)*n + (2,)` (or
`(*alphabet_sizes, a_node)` in the k-ary generalisation). The
trailing axis here is the *node's own alphabet*, indexed by
`mechanism_node.state` in `_single_node_cause_repertoire`. This is
the shape that the cause-side hot path of the repertoire algebra
actually reads.

## Canonical shape decision

After reading `pyphi/tpm.py:673-732`,
`pyphi/core/tpm/marginalization.py:22-87`,
`pyphi/core/repertoire_algebra.py:122-153`,
`pyphi/system.py:155-243`, and `pyphi/node.py:42-208`:

- `_legacy_backward_tpm` produces shape `(2,)*n + (n,)` for binary
  `n`-node substrates — system axes retain size 2, background axes
  become size 1, and the trailing axis is **always present** and
  indexed by substrate node index (size `n` when
  `remove_background=False`, size `len(system_indices)` when
  `True`). The current `_cause_tpm_factored` reproduces the same
  shape via SBN stacking before delegating to the legacy function.
- `_single_node_cause_repertoire` depends on the **per-node**
  trailing axis (the node's own alphabet axis) for the
  `mechanism_node.cause_tpm[..., mechanism_node.state]` slice — it
  is *not* shape-agnostic in that statement. However, it operates on
  `node.cause_tpm`, which is the per-node projection built in
  `Node.__init__` from the substrate-level `cause_tpm[...,
  self.index]` slice. So the substrate-level trailing axis is the
  upstream invariant that makes the per-node trailing axis
  meaningful.
- `Node.__init__` (`pyphi/node.py:63`), `System.proper_cause_tpm`
  (`pyphi/system.py:189`), `macro.py`'s SBN→SBS conversions, and
  `JointTPM.tpm_indices` all index or assume the substrate-level
  trailing axis. Removing it would require touching every one of
  those sites.

**Decision: Option B** — `alphabet_sizes + (n_observed_nodes,)`
legacy trailing axis at the substrate level. The native k-ary cause
path should produce shape `(*alphabet_sizes, n_observed_nodes)`,
and `CausePosterior.__init__` should canonicalise inputs to this
form.

Rationale:
- The trailing axis carries load-bearing semantics — it is the
  "next-state probability for output node `i`" axis that
  `Node.__init__` projects on. Without it, `cause_tpm[...,
  self.index]` is meaningless.
- All consumers (substrate level: `node.py`, `system.py`,
  `macro.py`; per-node level: `repertoire_algebra.py`) already
  agree on a trailing axis. Option A would force a refactor of
  every consumer for no semantic gain — the trailing axis is the
  natural place to put the per-output-node alphabet probabilities
  even in the k-ary case.
- The per-node shape `(*alphabet_sizes, a_node)` is already what
  `Node.__init__` constructs and what
  `_single_node_cause_repertoire` consumes. Generalising "size 2"
  to "size `a_node`" is a clean one-axis change with no contract
  shift.
- `_legacy_backward_tpm` slicing `[..., list(system_indices)]`
  (line 731) generalises directly: in the k-ary case the same
  trailing axis exists, sized by output-node count, and the slice
  is meaningful.

Implications for the rest of P12b:

- Native k-ary cause path output shape:
  `(*alphabet_sizes, n_observed_nodes)` where each
  `alphabet_sizes[i]` is the alphabet of node `i` and
  `n_observed_nodes` is the number of nodes whose next-state
  probability is represented (equal to `len(system_indices)` when
  background is removed, else equal to the full substrate size).
- `CausePosterior.__init__` canonicalisation: **yes**, accept
  either form and add the trailing axis if absent. In practice
  this means assert `arr.shape[:-1] == alphabet_sizes` and
  `arr.shape[-1] == n_observed_nodes` (raise on mismatch). No
  silent reshape — the constructor records the contract
  explicitly.
- Downstream consumer changes needed:
  - `Node.__init__` (`pyphi/node.py:63`): generalise
    `cause_tpm[..., self.index]` — still valid under Option B.
  - `Node.__init__` (`pyphi/node.py:87`): the off/on stack
    `np.stack([cause_tpm_off, cause_tpm_on], axis=-1)` must
    generalise to a k-ary alphabet stack. This is a per-node
    construction change, not a substrate-level shape change.
  - `JointTPM.tpm_indices` (`pyphi/tpm.py:503-506`): the
    `np.where(... == 2)` literal is binary-specific. Either
    deprecate in favour of an explicit `alphabet_sizes` query on
    `CausePosterior`, or generalise to `!= 1`. Per-node-cause is
    routed through `CausePosterior` so legacy `JointTPM` callers
    can keep the binary literal.
  - `System.proper_cause_tpm` (`pyphi/system.py:186-189`): the
    `[..., list(self.node_indices)]` slice is well-defined under
    Option B for any alphabet — no change.
  - `macro.py:103, 371` (SBN→SBS conversion): only valid for binary
    substrates anyway. Document the binary-only assumption
    explicitly; no shape change.
