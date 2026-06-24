# Intrinsic-units macro machinery — design

**Project:** Macro framework, sub-project 1 (of 3). A paper-faithful
implementation of the Marshall et al. 2024 intrinsic-units formalism's
*evaluation machinery*: macro unit objects, the four-step macro TPM
construction, and a `MacroSystem` seam through which the existing IIT 4.0
pipeline computes φ_s (and full cause-effect structures) for systems of
macro units. Replaces the legacy `pyphi/macro.py` outright. Lands before
the P15 surface freeze.

Sub-project 2 (separate spec): intrinsic-unit criteria (Eqs 15-16),
admissible-system recursion f(U^J, W^J), cross-grain complexes (Eq 19),
and bounded mapping-enumeration helpers. Sub-project 3 (separate spec):
reference goldens frozen from the authors' example repo.

**Sources of truth:**

- The paper: Marshall W, Findlay G, Albantakis L, Tononi G (2024).
  *Intrinsic Units: Identifying a system's causal grain.*
  (`papers/2024__marshall-et-al__intrinsic-units.pdf`; bioRxiv
  10.1101/2024.04.12.589163.)
- The authors' example code: github.com/CSC-UW/Marshall_et_al_2024
  (release v0.1.0, Zenodo 10.5281/zenodo.11211436) — a single ~513-line
  module that builds macro TPMs and hands them to pyphi (pinned at
  commit `941c65a`) for φ_s. Its `results/` directory ships committed
  high-precision φ_s values for ten constructions, including both paper
  examples; these are the acceptance anchors here and the golden corpus
  for sub-project 3.
- The legacy `pyphi/macro.py` is **not** a source of truth: its
  `MacroSubsystem`/`MacroSystem` pipeline implements the pre-2024
  formalism (no sliding-window mappings, no background apportionment, no
  per-unit discounting) and is disabled. Only its coarse-grain and
  blackbox *vocabulary* survives, reimplemented as mapping constructors.

## Scope

In scope:

- `pyphi/macro/` package: `MacroUnit` value objects with hierarchical
  (meso) composition, the four-step macro TPM construction (Eqs 26-40),
  the `MacroSystem` protocol implementer, and `coarse_grain`/`blackbox`
  mapping constructors.
- Structural validity checks needed for the construction to be
  well-defined (Eqs 12 and 18).
- Deletion of legacy `pyphi/macro.py` and its tests; replacement
  coverage per the Testing section.
- Acceptance against the paper's two worked examples at the authors'
  published precision.

Out of scope (sub-project 2): the intrinsic-unit *criteria* (maximal
irreducibility within, Eqs 15-16), candidate-set recursion, complexes
across grains (Eq 19), any search over mappings or apportionments
(`MacroSystem` takes W^J as given), and performance work beyond making
the paper-sized examples comfortable.

## Formalism inventory

All units binary at every grain (a macro unit must have exactly two
states). The universe substrate is binary with a conditionally
independent TPM.

| Object / step | Definition | Paper locus |
|---|---|---|
| Macro unit | J = (U^J, V^J, tau'_J, g'_J, W^J): micro constituents, direct constituents (micro or meso), update grain over V^J updates, state mapping, background apportionment | Eq 11 |
| Derived micro grain | tau_J: product of update grains down the constituent hierarchy | Eq 14 context, Fig 1B |
| Derived micro mapping | g_J: composition of the per-level mappings; maps tau_J-length micro state sequences of U^J to {0,1} | Eq 14 |
| Macro state | j = g_J(0u-ring^J): applied to the sequence of tau_J micro states ending at the current update (sliding window, oldest first) | Eqs 20-22 |
| Step 1: discounting | Per macro unit J, modified unit probabilities p-hat_J: constituents of J keep all connections (Eq 27); other system units and unapportioned background fully noised (Eq 28); apportioned background W^{J_k} keeps inputs from U^{J_k} union W^{J_k} only (Eq 29); product over units (Eq 30) | Eqs 26-30 |
| Step 2: sequences | Sequence probabilities over tau_J micro updates by chaining p-hat_J | Eq 31 |
| Step 3: background | Effect side conditions on the current background micro state (Eq 33); cause side weights background states by Bayes with a uniform prior over the earliest update in the sequence (Eq 34) | Eqs 32-34 |
| Step 4: compression | Sum sequence probabilities over the g_J preimages D_J(j) (Eqs 35-36); map current micro states to macro states with the sequence-proportion weights r(u^S, s) (Eqs 37-39); product over units gives the macro TPMs (Eq 40) | Eqs 35-40 |
| Macro TPMs | T_c, T_e over macro system states; phi_s computed from them exactly as for micro systems, with macro units perturbed uniformly over their two states | Eqs 41-42 |
| Validity (structural) | W^{V_i} subset-of W^J for constituents (Eq 12); (U^J union W^J) disjoint across the system's units (Eq 18) | Eqs 12, 18 |

**Micro-reduction property:** when every unit has tau_J = 1, identity
g_J, singleton constituents, and empty W^J, Steps 1, 2, and 4 are
trivial and T_c/T_e equal the standard micro system TPMs (paper, p. 10).
This is the seam regression test.

## Components

### `pyphi/macro/units.py`

```python
@dataclass(frozen=True)
class MacroUnit:
    constituents: tuple[MacroUnit | int, ...]    # V^J (micro indices or meso units)
    update_grain: int                            # tau'_J >= 1
    mapping: tuple[int, ...]                     # g'_J truth table (0/1 entries)
    background_apportionment: tuple[int, ...] = ()   # W^J (universe indices)
```

`micro_constituents` (U^J) is a derived cached property — the recursive
union of the constituents' micro constituents (a micro index `i`
contributes `{i}`) — so the Eq 11 five-tuple cannot be stated
inconsistently.

- The mapping is an explicit truth table over the
  `2 ** (update_grain * len(constituents))` joint sequence-states of the
  constituents, stored as a tuple of 0/1 ints. **Indexing convention:**
  little-endian over constituents within an update (first constituent
  varies fastest), updates ordered oldest-first, update index varying
  slowest. The convention is pinned by asymmetric-fixture tests.
- **Alphabet generality (design hedge):** all sequence-state and
  truth-table index arithmetic is implemented as mixed-radix math keyed
  to per-constituent alphabet tuples (not binary bit tricks), and the
  mapping codomain size is carried as a property rather than assumed.
  Validation enforces binary at both the micro and macro level (the
  paper's formalism), so behavior is binary-only — but relaxing to
  k-ary micro substrates later is a lift-the-guard extension consistent
  with the core pipeline's existing multivalued-unit support, and
  experimental relaxation of macro-unit binarity (an extrinsic-analysis
  mode, contrary to the paper's intrinsicality argument) is not
  structurally precluded.
- Derived (cached) properties: `micro_grain` (tau_J), `micro_mapping`
  (the composed g_J over micro constituent sequences), and
  `state_from(history)` (Eq 22; `history` is a sequence of micro states
  of U^J, oldest first, length tau_J).
- Mapping constructors (module functions, returning truth tables):
  - `coarse_grain(num_constituents, on_counts)` — update grain 1; macro
    state is 1 when the count of ON constituents is in `on_counts` (the
    paper's "coarse-graining" class, Example 1's mapping; subsumes the
    legacy `CoarseGrain.grouping` semantics).
  - `blackbox(num_constituents, update_grain, output_constituents)` —
    macro state is dictated by the designated output constituents' state
    at the final update of the window (the paper's "black-boxing"
    class, Example 2's mapping).

### `pyphi/macro/tpm.py`

Pure functions implementing Steps 1-4 against a binary `Substrate` and a
tuple of `MacroUnit`s. The public entry point returns per-unit macro
factors for both directions:

```python
def macro_tpms(substrate, units, current_micro_history) -> tuple[FactoredTPM, FactoredTPM]
    """(T_c, T_e) over the macro system's states (Eqs 26-40)."""
```

- Internal helpers mirror the steps (`_discounted_unit_probabilities`,
  sequence chaining, `_background_weights_cause/effect`, preimage
  compression) so each is unit-testable in isolation.
- Steps 2 and 4 are fused: sequence probabilities are accumulated into
  the D_J(j) preimage sums per chaining step where possible, rather than
  materializing the full `(2**n)**tau` sequence tensor. The authors'
  reference implementation demonstrates the factorization; paper-sized
  cases (n = 8, tau = 2) must run comfortably.
- The macro TPMs are state-independent objects defined for all macro
  state pairs; the current micro history enters only Step 3's effect
  conditioning (Eq 33: the current background state) and the cause-side
  Bayes weighting (Eq 34), exactly as in the paper.

### `pyphi/macro/system.py`

```python
@dataclass(frozen=True)
class MacroSystem:
    substrate: Substrate
    units: tuple[MacroUnit, ...]
    micro_history: tuple[tuple[int, ...], ...]   # oldest first, len == max tau_J
```

- Satisfies `SystemPublicInterface` so `pyphi.formalism.iit4.sia()` /
  `ces()` (and everything downstream: distinctions, relations,
  unfolding) consume it unchanged: `cause_tpm`/`effect_tpm` are the
  Step-4 `FactoredTPM`s, `nodes` via the standard `generate_nodes`,
  `state` is the macro state tuple via each unit's `state_from`,
  `node_indices` are the macro unit positions, `cm` is all-ones (the
  formalism defines no macro connectivity; all-ones imposes no pruning).
- A bare micro state (not a history) is accepted and auto-wrapped only
  when all units have micro grain 1; otherwise the history length must
  equal the maximum micro grain.
- Constructor validates: binary substrate; Eq 12; Eq 18; mapping
  surjectivity onto {0,1} (each unit must be able to take both macro
  states); grains >= 1; history length and entry shapes.

## Configuration mapping for acceptance

The authors' results were computed under their committed
`pyphi_config.yml`: IIT 4.0, `GENERALIZED_INTRINSIC_DIFFERENCE` for both
repertoire-distance settings, `DISTINCTION_PHI_NORMALIZATION:
NUM_CONNECTIONS_CUT`, `PARTITION_TYPE: ALL`, `SYSTEM_PARTITION_TYPE:
SET_UNI/BI` with `SYSTEM_PARTITION_INCLUDE_COMPLETE: false`, precision
13. Acceptance tests run under the 2.0 equivalent (IIT_4_0_2023 + GID);
mapping the legacy `SET_UNI/BI` system-partition scheme to its 2.0
registry equivalent is a planning task, and any mismatch in available
schemes must be resolved (or surfaced) before values are compared.

## Error handling

`ValueError` with a specific message on: non-binary substrate; Eq 12
violation; Eq 18 violation (overlapping `U^J union W^J` across units);
a truth table of the wrong length or one that never produces 0 or never
produces 1; update grain < 1; micro history of the wrong length or entry
shape; background apportionment overlapping the system's micro
constituents.

## Testing

1. **Micro-equivalence regression (the seam check):** identity macroing
   of `basic`, `xor`, and `grid3` example systems reproduces
   `System`-based `sia()` and `ces()` results exactly (phi values,
   partitions, distinction counts).
2. **Paper acceptance (authors' committed precision):**
   - Example 1 (4 units, coarse-graining, tau = 1): micro panel
     (φ_s(A) = 0.003976279885291341, φ_s(AB) = 0.044088890564147803,
     ...) and macro φ_s({alpha, beta}) = 1.0039763812908649.
   - Example 2 (8 units, black-boxing, tau = 2): micro panel anchors
     and macro φ_s({alpha, beta}) = 1.1183776016500528.
   The TPMs are constructed from the paper's published rules (Example 1:
   base 0.05, +0.01 self, +0.1 horizontal, +0.8 vertical-and-diagonal;
   Example 2: transcribed from the authors' scripts during planning).
3. **Per-step unit tests:** Step-1 discounting row properties
   (constituent rows untouched; Eq-28 rows equal the uniform-average
   marginal; Eq-29 rows insensitive to non-apportioned inputs);
   stochasticity of every intermediate; Step-4 preimage sums; r(u^S, s)
   proportions sum to 1 per macro state.
4. **Convention pinning:** asymmetric fixtures for the truth-table
   indexing (distinct constituents, distinct updates) and for meso
   composition (a 2-level hierarchy whose composed g_J is hand-checked).
5. **Hand-computed tiny case:** 2 micro units, 1 macro unit, tau = 1,
   every intermediate of Steps 1-4 computed by hand in the test.
6. **Background apportionment path (Eq 29):** exercised by unit tests
   with nonempty W^J (both paper examples have empty background, so this
   path has no published anchor until sub-project 3's goldens; flagged
   here as the least-anchored piece).

## Files

- `pyphi/macro/__init__.py`, `pyphi/macro/units.py`,
  `pyphi/macro/tpm.py`, `pyphi/macro/system.py` — new package
- `pyphi/macro.py` — **deleted** (legacy `CoarseGrain`/`Blackbox`
  pipeline and the disabled `MacroSystem`)
- `test/test_macro_units.py`, `test/test_macro_tpm.py`,
  `test/test_macro_system.py` — new (the latter replaces the dark
  606-line legacy file)
- `test/test_macro.py`, `test/test_macro_blackbox.py` — deleted; the
  generator tests are superseded by the mapping-constructor tests, and
  the legacy blackbox SIA scenarios are superseded by the paper
  acceptance tests
- `changelog.d/intrinsic-units-macro.feature.md` — new
- `ROADMAP.md` — Marshall-2024 macro entry updated (sub-project
  structure; in-2.0 scope decision recorded)

## Notes carried from brainstorming

- Approach: `MacroSystem` as a `SystemPublicInterface` implementer (the
  `TransitionSystem` pattern from actual causation) — zero pipeline
  changes. A derived-Substrate approach was rejected as mathematically
  wrong (the macro T_c is not the Bayes inversion of T_e), and carrying
  an explicit cause TPM on `Substrate` was rejected for blast radius.
- The grain *decision* layer (which units are intrinsic) is sub-project
  2; this sub-project only evaluates what it is given. The W^J
  apportionment is likewise taken as input here; apportionment search
  belongs to sub-project 2.
- Once T_c/T_e are built, "there is no further reference to the
  background units, the grain of the units, or their micro
  constituents" (paper p. 13) — macro units are perturbed uniformly like
  any units. This is what makes the protocol seam sufficient.
- The authors' repo also contains the Fig 2 constructions (`sfn`, `sfs`,
  `sfnn`, `min`, `bu` result sets) at full precision; they become
  sub-project 3 goldens.
