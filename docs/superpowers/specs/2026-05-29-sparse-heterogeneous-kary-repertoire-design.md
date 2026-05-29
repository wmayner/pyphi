# Sparse-Heterogeneous k-ary Repertoires

## Problem

The core IIT cause and effect repertoires are broken for networks that are
**both sparsely connected and heterogeneous-alphabet** (multi-valued units
with differing per-node alphabet sizes). Binary networks and
fully-connected k-ary networks work; the sparse + heterogeneous combination
— the common real-world case — raises a shape error before any phi is
computed.

This was discovered while validating actual-causation (AC) k-ary support
against Albantakis et al. 2019, Figure 11. The defect is **IIT-wide**, not
AC-specific: it reproduces in the plain IIT cause/effect repertoire path
with no AC objects involved. AC inherits it.

### Evidence

Minimal IIT reproducer — a 3-node substrate, `node0 (k=3)`, `node1 (k=3)`
feeding `node2 (k=4)` (sparse `cm`: `0->2`, `1->2`), full-dimension factors:

- **Cause**, `mechanism={node2}` `purview={node0}`:
  `ValueError: non-broadcastable output operand with shape (3,1,1) doesn't
  match the broadcast shape (3,1,4)` — raised in
  `_cause_repertoire_inner` (`pyphi/core/repertoire_algebra.py`) at the
  in-place `joint *= ...` step.
- **Effect**, `mechanism={node0}` `purview={node2}`:
  `ValueError: cannot reshape array of size 16 into shape (1,1,4)` — raised
  in `_single_node_effect_repertoire`'s reshape to `repertoire_shape`.

A fully-connected heterogeneous network (`node0 k=3`, `node1 k=4`, no `cm`)
computes both cause-repertoire directions and a full CES correctly. The
trigger is therefore the combination of **sparse connectivity** and
**heterogeneous alphabets**, independent of `external_indices`
(reproduces identically with and without AC-style background conditioning).

### Why it went uncaught

Every existing k-ary fixture is either fully-connected (the `multivalued_*`
goldens, the AC k=3 smoke test) or binary (all other fixtures). No fixture
is simultaneously k>2 and sparsely connected, so this path was never
exercised. The `p12b-multivalued-units-complete` changelog claim that
"k>2 is supported throughout the SIA/CES pipeline" is therefore overstated:
it holds for fully-connected k-ary, not sparse k-ary.

## Root cause

For a sparse heterogeneous node, the per-node factored cause/effect TPM
carries the node's **own previous-state dimension** (size `k`), and the
single-node repertoire builders do not collapse it to size 1 for arbitrary
alphabets:

- `_single_node_cause_repertoire` (`repertoire_algebra.py:142-143`) indexes
  the node's cause factor at its observed state, marginalizes the node's
  *inputs* outside the purview, and returns the raw array **without
  reshaping to the canonical `repertoire_shape`**. The node's own dimension
  survives at size `k`, so the array does not broadcast against the
  purview-shaped `joint` in `_cause_repertoire_inner`.
- `_single_node_effect_repertoire` (`repertoire_algebra.py:147-170`) *does*
  reshape to `repertoire_shape`, but `marginalize_out` leaves the array
  larger than the reshape target for sparse heterogeneous nodes (e.g. size
  16 or 27 into a 4- or 3-element target), so the reshape fails.

For binary (`k=2`) the surviving dimension is size 2 and the existing
marginalize/reshape arithmetic happens to align; for fully-connected k-ary
the factor spans all node dimensions and the shapes also align. The
heterogeneous + sparse case is the gap.

## Goal

A focused root-cause fix so that, for **any** combination of per-node
alphabet sizes and connectivity, both `_single_node_cause_repertoire` and
`_single_node_effect_repertoire` return arrays conformant to the canonical
`repertoire_shape` (purview nodes at their alphabet size, all other nodes
— including the node's own dimension — collapsed to size 1). Validate that
this unblocks AC k-ary against the Figure 11 paper values.

## Approach

Fix the shared defect once at its root rather than patching each direction
independently. The node's own previous-state dimension must collapse to
size 1 for arbitrary alphabets in the per-node factored repertoire. The
exact locus — the node `cause_tpm` / `effect_tpm` accessor, a shared
collapse/reshape helper, or the two single-node builders — is pinned during
implementation by tracing the minimal reproducer; the constraint is that
both builders emit canonical `repertoire_shape`-conformant arrays through
one shared mechanism.

Iterate: once the first shape gap is fixed, the property-test matrix
(below) surfaces any sibling shape bugs in the cause/effect path; fix those
within this same effort until the matrix is green.

## Testing

Comprehensive, property-based, per the project's preference for property
tests on silent-correctness mathematical bugs.

1. **Hypothesis property tests** parameterized over the cross-product of:
   - alphabet sizes (mixed, e.g. drawn from {2, 3, 4} per node),
   - connectivity (sparse and dense),
   - direction (cause and effect).

   Invariants asserted for each generated (substrate, mechanism, purview):
   - the repertoire's shape equals the canonical
     `repertoire_shape(node_indices, purview, alphabet_sizes)`;
   - the repertoire normalizes to 1;
   - where a sparse construction has an equivalent dense/joint
     construction, the two repertoires are equal to within `PRECISION`.

   Use an isolated, seeded RNG (`np.random.default_rng(seed)`); save the
   seed with any persisted output.

2. **Regression goldens.** Add sparse-heterogeneous IIT fixture(s) to the
   golden zoo (e.g. a 3-node `(k3, k3) -> k4` substrate) and wire them into
   the golden regression suite.

3. **Small fast AC fixture.** A hand-verifiable sparse-heterogeneous AC
   transition (small enough to be a fast committed guard) asserting a
   known alpha value.

4. **fig11 acceptance test** (marked `slow`). See the canonical values
   below.

5. **Invariant.** All existing binary and fully-connected-k-ary goldens
   remain byte-identical — nothing outside the sparse-heterogeneous path
   changes.

### Figure 11 acceptance (the paper oracle)

Network: seven three-state voters `A-G` (candidate index in {0, 1, 2})
feeding one four-state winner `W` (state 0 = no winner; states 1-3 = winning
candidate + 1). `W` is the candidate with a strict majority (>= 4 of 7
votes), else 0. Sparse `cm`: voters -> `W` only. Built with full-dimension
factors (each factor shaped `(*alphabet_sizes, k_i)`); voter factors uniform
`1/3`, the `W` factor the deterministic majority one-hot broadcast over
`W`'s own previous-state dimension.

Transition (paper `{ABCDEFG = 1111122} -> {W = 1}`, candidate "1" -> index 0,
"2" -> index 1): `before` voters `(0,0,0,0,0,1,1)` (five for candidate 0, two
for candidate 1) `-> {W = 1}`.

Expected, from the paper:
- **Actual cause of `{W=1}`:** an undetermined set of **four of the five**
  candidate-0 voters `{A,B,C,D,E}`, **alpha_c^max = 1.893 bits** (assert
  `len(purview) == 4`; the two candidate-1 voters contribute 0).
- Effect ladder (informational): `{A=1}->{W=1}` = 0.718, `{AB=11}` = 0.581,
  `{ABC=111}` = 0.404, `{ABCD=1111}` = 0.190 bits.

## Success criteria / acceptance gates

- fig11 yields alpha_c^max = 1.893 (+/- a small tolerance) with a
  fourth-order undetermined cause among `{A..E}`; candidate-1 voters
  contribute 0.
- The Hypothesis property suite is green across the alphabet x connectivity
  x direction matrix.
- All pre-existing goldens (binary + fully-connected k-ary) are
  byte-identical.
- `uv run pytest` (no path argument) green; pyright and ruff clean.

## Non-goals (deferred to the follow-up, "Approach B")

- Unifying cause/effect shape handling into a single shared helper beyond
  what the root-cause fix requires.
- Validating or normalizing reduced-dimension factors passed to the
  `marginals=` constructor (today they are silently accepted and crash
  downstream).
- Hardening `_cause_tpm_factored`'s full-dimension-factor assumption.
- Any sparse-k-ary audit beyond the cause/effect repertoire path.

## Risks and mitigations

- **Sibling shape bugs** may surface once the first is fixed — the
  property-test matrix is designed to catch them; fix iteratively until
  the matrix is green.
- **fig11 tractability** is unknown until the fix lands (the bug aborts
  before the expensive search). The small fast AC fixture is the real
  committed guard; fig11 is `slow`-marked and time-boxed. If it proves
  intractable, document that and keep the small fixture as the committed
  validation.
- **Core-math regression risk** — mitigated by the byte-identical golden
  invariant on binary and fully-connected-k-ary fixtures.

## Files

- `pyphi/core/repertoire_algebra.py` (the two single-node builders and/or a
  shared collapse helper), and possibly the node `cause_tpm`/`effect_tpm`
  accessor or `pyphi/core/tpm/`.
- `test/` — Hypothesis property tests; sparse-heterogeneous golden
  fixture(s) in the golden zoo; the small AC fixture; the fig11 acceptance
  test (replacing the current `test_paper_fig11_three_candidate_alpha`
  skip stub in `test/test_actual.py`).
- A changelog fragment correcting the P12b "k>2 throughout the pipeline"
  claim — sparse heterogeneous k-ary is now actually supported.

## ROADMAP linkage

Completes the AC k-ary validation referenced under P12 (the deferred AC
k-ary piece) and corrects P12's "k>2 supported throughout the SIA/CES
pipeline" status to reflect that sparse k-ary was incomplete until this
work. The broader unification/hardening follow-up ("Approach B") is tracked
separately.
