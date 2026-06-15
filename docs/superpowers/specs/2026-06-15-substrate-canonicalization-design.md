# P11.95c (a)+(c) — Substrate Canonicalization: Design

**Status:** draft (awaiting review)
**Date:** 2026-06-15
**Roadmap item:** P11.95c (a)+(c), Wave 2 (pre-freeze surface-affecting)
**Scope note:** Case (b) (`intrinsic_equivalence` API) stays out of 2.0 — the Wave-1
CES-completeness search settled it (CES is complete on complexes; the residual
incompleteness is reducible-system-only). This spec covers (a)+(c) only.

---

## 1. Motivation

Systems related by a node permutation must produce equal results — relabeling
the nodes is not a physical change. PyPhi already satisfies this for system Φ
(`test_sia_phi_symmetric` passes) and for system intrinsic information
(`test_system_intrinsic_information_symmetric` passes). It does **not** satisfy
it for the *per-direction* φ multiset `{φ_c, φ_e}`:
`test_sia_per_direction_phi_multiset_symmetric` is `xfail(strict=True)`.

### 1.1 Root cause (confirmed by experiment, 2026-06-15)

The canonical example pair is `and_xor_substrate()` at state `(0,1)` versus
`xor_and_substrate()` at `(1,0)` — the same all-ones connectivity with the AND
and XOR gates swapped, related by the node permutation π: 0↔1.

Both systems are **reducible**: `sia.phi = 0.0` for each, and they agree on
that. The divergence is purely in how the *zero* decomposes per direction:

| AND-XOR@(0,1) | reports | XOR-AND@(1,0) | reports |
|---|---|---|---|
| chosen pair `c=(1,0), e=(0,1)` | `{φ_c=0.5, φ_e=0}` | chosen pair `c=(1,0), e=(1,0)` | `{φ_c=0, φ_e=0}` |

Dumping the full per-`(cause_state, effect_state)`-pair table shows it is
**exactly permutation-symmetric** under π:

```
AND-XOR@(0,1):  (c=(1,0), e=(0,1)) -> φ_s=0, φ_c=0.5   [chosen]
                (c=(0,1), e=(0,1)) -> φ_s=0, φ_c=0
XOR-AND@(1,0):  (c=(1,0), e=(1,0)) -> φ_s=0, φ_c=0      [chosen]
                (c=(0,1), e=(1,0)) -> φ_s=0, φ_c=0.5
```

The underlying physics is label-invariant: the *set* of per-pair outcomes is
identical under π. The entire divergence is the **tie-break**. Every pair has
`φ_s = 0`, so the state-tie cascade (`resolve_ties.resolve_state_tie`) cannot
separate them at the Integration level (φ_s argmax) and, with the escalation
budget capped at `"Integration"`, returns `UNRESOLVED_WITHIN_BUDGET`. The caller
(`formalism/iit4.sia`) then falls through to `outcome.tied_set[0]` — the **first
pair in enumeration order**. Both substrates enumerate the tied cause states in
the same literal order `[(1,0), (0,1)]` and pick literal `(1,0)` first — but
because the TPMs are permuted, literal `(1,0)` lands on *opposite orbits*, so one
substrate grabs the φ_c=0.5 pair and the other grabs the φ_c=0 pair.

Two corollaries, both verified:

- **Not a short-circuit artifact.** The divergence persists with
  `shortcircuit_sia=False`.
- **Reducible-systems-only.** It manifests only when every candidate ties at
  `φ_s = 0`. This matches the Wave-1 CES-completeness finding that every
  label-invariance counterexample is a reducible system.

### 1.2 Disposition

Per-direction φ on a reducible system's zero-φ MIP is genuinely non-unique (many
partitions/states tie at zero). The fix is to make the *selection among ties*
deterministic up to relabeling, rather than to assign physical meaning to one
arbitrary tied value. A permutation-aware (canonical) tie-break achieves this.
Substrate canonicalization is also a reusable asset in its own right
(label-independent comparison, caching, dedup, test helpers), so we ship it as a
standalone module and wire the tie-break to use it.

---

## 2. Goals

1. A non-invasive `pyphi/automorphism.py` module exposing
   `substrate_automorphisms`, `substrate_canonical_form`,
   `are_substrates_isomorphic`, and the `canonical_state` helper — all defined to
   preserve the substrate's **behavior** (connectivity **and** TPM), not just
   connectivity.
2. A permutation-invariant Determinism step in the state-tie resolution so that
   per-direction φ reporting is deterministic up to relabeling (placement
   per §5.1).
3. Un-xfail `test_sia_per_direction_phi_multiset_symmetric` and keep all other
   permutation-symmetry invariants green.
4. No change to any value on irreducible systems (φ_s > 0); the new tie-break
   fires only when the existing cascade reaches `UNRESOLVED_WITHIN_BUDGET`.

## 3. Non-goals

- Case (b) `intrinsic_equivalence` API (out of 2.0; see scope note).
- Any change to the φ-computation hot path. The tie-break touches only the
  selection among already-computed tied candidates.
- A pynauty dependency or a pluggable canonicalization-backend abstraction
  (see §6).

---

## 4. Part A — `pyphi/automorphism.py` (the sidecar)

### 4.1 Substrate identity and the action of a permutation

A substrate's behavioral identity is `(cm, tpm, alphabet_sizes)`. A node
permutation π (a tuple where `π[i]` is the new index of node `i`) acts by:

- **connectivity:** `cm[ix_(π, π)]`
- **TPM:** `substrate.tpm.to_joint().permute_nodes(π)` (the existing
  `JointDistribution.permute_nodes` transposes the input axes and reindexes the
  output node axis), reconstructed via `FactoredTPM.from_joint(...)`
- **alphabet sizes:** reordered by π

π is a **substrate automorphism** iff all three are invariant under it. This is
the true definition: it preserves each node's mechanism, so an AND node can
never be mapped onto an XOR node even when their wiring is identical.

### 4.2 Engine: exact enumeration (see §6 for why not pynauty)

Candidate permutations are pruned cheaply before any TPM comparison:

- keep only π that preserve `cm` (`cm[ix_(π,π)] == cm`) and `alphabet_sizes`;
- optionally refine by per-node invariants (in/out degree, alphabet size) to
  shrink the candidate set further.

The surviving candidates are checked for TPM invariance via the §4.1 action.
This is exact and directly enforces TPM-preservation. It is fast for every
substrate where Φ is computable: Φ is O(2ⁿ), so n ≲ 8 in practice, where n!
is ≤ 40320 and the expensive TPM check runs only on the (usually tiny) set of
connectivity-automorphisms that survive pruning.

### 4.3 API

```python
def substrate_automorphisms(substrate: Substrate) -> tuple[tuple[int, ...], ...]:
    """All node permutations preserving connectivity AND TPM. Always contains
    the identity. The group's product structure is not materialized; the flat
    tuple of permutations suffices for the consumers here."""

def substrate_canonical_form(
    substrate: Substrate,
) -> tuple[Substrate, tuple[int, ...]]:
    """Return ``(canonical_substrate, canonical_permutation)`` where
    ``canonical_substrate`` is the lexicographically smallest relabeling of
    ``substrate`` over all node permutations, and ``canonical_permutation`` is a
    π achieving it. The canonical form is unique; the permutation is unique up to
    the substrate's automorphism group."""

def are_substrates_isomorphic(s1: Substrate, s2: Substrate) -> bool:
    """True iff some node permutation maps s1's connectivity, TPM, and alphabet
    sizes onto s2's. Implemented as equality of canonical forms."""

def canonical_state(
    substrate: Substrate, state: tuple[int, ...]
) -> tuple[int, ...]:
    """Map ``state`` into the substrate's canonical coordinates, then reduce
    over the automorphism orbit: the lexicographically smallest image of
    ``state`` under every permutation that takes ``substrate`` to its canonical
    form. This is the orbit-invariant identity of a state — equal for
    corresponding states of any two isomorphic substrates (see §5.2). Takes
    ``state`` as an argument; the module never stores state."""
```

The lexicographic ordering for the canonical form is over a fixed, fully
specified serialization of `(cm, alphabet_sizes, joint-TPM bytes)` so the
"smallest" relabeling is well-defined and stable across runs.

`substrate_canonical_form` does **not** carry state — it canonicalizes the
substrate. `canonical_state` is the state-aware helper the tie-break consumes
(§5); it takes state as an argument and stores nothing.

### 4.4 What the module does not import

`pyphi/automorphism.py` imports only `pyphi.substrate` / `pyphi.core.tpm` and
numpy/stdlib. It does not import `pyphi.formalism` or any result models, keeping
it a leaf utility.

---

## 5. Part B — canonical tie-break for the state-tie fallback

`resolve_state_tie` (`pyphi/resolve_ties.py`) currently has two cascade levels:
Integration (argmax φ_s) and Composition (argmax Φ). When both tie and the
budget caps escalation, it returns `UNRESOLVED_WITHIN_BUDGET` and the caller
takes `tied_set[0]` (enumeration order).

Add a **Determinism level** (the pyphi-canonicalization step, mirroring the
existing lex-representative step in `resolve_ac_causal_link_tie` and
`resolve_iit3_complex_tie`): among the still-tied candidates, pick the one whose
state has the **lex-smallest `canonical_state`**.

Because two permutation-related substrates share a canonical form, and their
tied sets are permutation-images of each other, the lex-smallest `canonical_state`
selects orbit-corresponding states on both — so the chosen per-direction φ
multiset is identical by construction.

### 5.0 Why the orbit, not a single permutation

A single `canonical_permutation` is **not** sufficient. For substrates `S` and
`S' = σ(S)`, the canonical-coordinate image of a state `s` on `S` and of its
counterpart `σ(s)` on `S'` agree only **up to an automorphism of `S`** (the two
canonicalizing permutations differ by an element of `Aut(S)`). So the
permutation-invariant key must reduce over the automorphism orbit:
`canonical_state` takes the lex-min image of `s` over *all* permutations
carrying `S` to its canonical form. This is exactly why `substrate_automorphisms`
is load-bearing for the tie-break and not merely a convenience export.

### 5.1 Wiring

The cascade resolver needs the substrate (to obtain its canonical permutation)
and a way to read each candidate's state. Two viable placements:

- **Preferred:** resolve the canonical tie-break in `formalism/iit4.sia` at the
  point that currently does `chosen_key = outcome.tied_set[0]`, replacing that
  line with a selection over `outcome.tied_set` keyed on
  `automorphism.canonical_state(system.substrate, state)` for each tied key's
  cause/effect states. This keeps `resolve_ties` substrate-agnostic (it already
  takes only the per-state MIP mapping) and confines the new dependency on
  `automorphism` to the formalism layer.
- **Alternative:** thread a Determinism-level key function into the cascade via
  `ResolutionContext`. Heavier; only adopt if a second caller needs the same
  canonical tie-break.

The preferred placement is a one-site change at `formalism/iit4/__init__.py`
(around the `chosen_key = outcome.tied_set[0]` fallback) and leaves
`resolve_ties.resolve_state_tie` untouched.

### 5.2 Determinism-level key

For each tied `(cause_state, effect_state)` key, the sort key is
`(canonical_state(substrate, cause_state), canonical_state(substrate, effect_state))`;
pick the minimum. The full tied set is still recorded as tie metadata (no
information is lost; only the representative changes).

---

## 6. Engine decision: exact enumeration, not pynauty

The roadmap named pynauty. The investigation found it the wrong fit, for two
independent reasons:

1. **It cannot see behavior.** pynauty canonicalizes vertex-colored graphs by
   *connectivity*. A substrate automorphism must preserve the **TPM**, and the
   canonical example pair (AND-XOR vs XOR-AND) has identical connectivity and
   differs only in gate behavior. Encoding behavior as a vertex color is
   circular — a node's mechanism is defined over its inputs, whose identities
   are exactly what the relabeling permutes — so any pynauty result must still
   be filtered by an explicit TPM-preservation check. That filter *is* the
   brute-force step; pynauty adds a dependency without removing the work.
2. **Its speed advantage never engages.** pynauty pays off on graphs far larger
   than n! enumeration can handle. Φ is O(2ⁿ), so substrates where Φ is
   computable have n ≲ 8 (n! ≤ 40320); exact enumeration with cheap pruning is
   milliseconds. The regime where pynauty would help is one where Φ itself is
   intractable (a future approximate-Φ path, out of 2.0).

A pluggable-backend abstraction is also rejected (YAGNI): the engine lives in a
single private function and can be swapped if an approximate-Φ path ever needs
it.

---

## 7. Testing

- **Un-xfail** `test/test_invariants.py::test_sia_per_direction_phi_multiset_symmetric`;
  it must pass.
- **Sidecar unit/property tests** (new `test/test_automorphism.py`):
  - identity permutation is always in `substrate_automorphisms`;
  - every returned automorphism genuinely preserves cm + TPM + alphabet sizes;
  - `substrate_canonical_form` is invariant under relabeling: for a random π,
    `canonical_form(relabel(s, π)) == canonical_form(s)`;
  - `are_substrates_isomorphic` is reflexive, symmetric, and agrees with a
    direct permutation search; `and_xor` ≅ `xor_and`, and a known
    non-isomorphic pair is rejected;
  - automorphism group recovers known symmetries (e.g. a symmetric copy-ring);
  - `canonical_state` is orbit-invariant: for `s' = σ(s)` on `σ(substrate)`,
    `canonical_state(σ(substrate), s') == canonical_state(substrate, s)`; and it
    is idempotent on a canonical substrate.
- **Hypothesis** generation over small random substrates for the canonical-form
  invariance property.
- **Full golden suite** (`uv run pytest` with no path argument, to include
  doctests) to catch any reducible-system golden whose *reported specified
  state* shifts under the new tie-break. Any shift is investigated and recorded
  before regenerating — not silently re-baselined.

## 8. Risks

- **Golden drift on reducible systems.** The tie-break can change which tied
  state is reported as the representative for a reducible-system golden (values
  for irreducible systems cannot change). Treat any such shift as a signal to
  confirm, not a number to re-pin reflexively.
- **Canonical-ordering stability.** The lexicographic key must be a fully
  specified, run-stable serialization, or the canonical form is ambiguous. The
  property test (`canonical_form(relabel) == canonical_form`) guards this.

## 9. Changelog

A `changelog.d/*.feature.md` (or `.fix.md`) fragment describing the
permutation-invariant per-direction φ reporting and the `automorphism` module.

## 10. Roadmap update

On landing: flip the P11.95c (a)+(c) dashboard row in `ROADMAP.md` to landed,
note the exact-enumeration engine (pynauty rejected), and record that the
per-direction asymmetry was a reducible-system-only tie-break determinism issue.
