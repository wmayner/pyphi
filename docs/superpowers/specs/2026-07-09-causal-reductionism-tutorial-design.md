# Causal-reductionism frog tutorial and example builders — design

## Goal

Replace the print-only `frog_example` demo with two clean pieces:

1. **Reusable example builders** in `pyphi/examples.py` that construct the
   frog substrates and transitions from Grasso, Albantakis, Lang & Tononi
   (2021), *Causal reductionism and causal structures* (Nat. Neurosci.), and
   **return** them (no printing, no config side effects).
2. **An executable tutorial** in the docs that reproduces the paper's
   actual-causation analysis and demonstrates its core idea: composite
   (higher-order) mechanisms have irreducible causes that a first-order,
   reductionist account misses.

The paper is committed at
`papers/2021__grasso-et-al__causal-reductionism.pdf`.

## Background: what the paper does

The paper argues against *causal reductionism* — the assumption that once every
elementary unit's cause is fixed, there is nothing causal left to explain. It
uses three simulated "frog" organisms (species **F3**, **F2**, **F1**) with
different wiring between sensors, central units, and motor units. For each, it
analyzes a state transition with **actual causation** (an `account` of "what
caused what") and shows that the causal account contains *composite*
second-order causes (e.g. two sensors jointly causing a central unit) that are
irreducible to the first-order causes of the individual units.

The three species form a comparison: as the wiring changes, the causal
structure — and which composite causes appear — changes with it.

## Example builders (`pyphi/examples.py`)

Remove the `frog_example` print-demo (and its `@config.override` decorator).
Replace it with pure builders:

- `frog_substrate(species="F3")` → the frog `Substrate` for `species` in
  `{"F1", "F2", "F3"}` (default `"F3"`, the paper's main example). Raises
  `ValueError` on an unknown species.
- `frog_transition(species="F3")` → the paper's `Transition` for that species
  (the substrate plus the before/after states and the cause/effect unit sets
  the paper analyzes), ready to pass to `pyphi.actual.account`.

Both are registered with `@register_example`. The substrate-construction logic
(the current inner `get_net`, with its Gaussian / Naka-Rushton / LogFunc /
inhibiting-input mechanisms) is kept as a module-level private helper
`_frog_net(...)`, deduplicated across the three species.

**No config on the builders.** Constructing a `Substrate` or `Transition` does
not depend on the IIT-3.0 measure/partition config; only *computing the
account* does. So the builders carry no `@config.override`; setting the
formalism is the caller's (and the tutorial's) responsibility. This matches
every other example function, which never pins config. (Verification point: if
`Transition` construction trips `validate_system_states` under the default
config, the builder documents the required `validate_system_states=False` or is
constructed so the states validate — decided during implementation, not by
pinning global config.)

**Node labels and states are taken from the paper's figures**, reconciling the
current code's inconsistent labels (F1 uses generic `S1..M2`; F2/F3 use
anatomical `SL/SC/SR/CL/...`). Each species' wiring, labels, before/after
states, and cause/effect unit sets are verified against the corresponding
figure in the paper before landing.

## Tutorial (`docs/tutorials/causal-reductionism.md`)

An executable MyST page, jupytext-paired to a `.ipynb` with a Colab badge, in
the same style as the existing tutorials (`worked-example`,
`cause-effect-structure`). Added to the `docs/tutorials/index.md` toctree.

Structure:

1. **The frog world and the reductionist claim.** Introduce the paper and the
   thesis: if every unit's cause is fixed, is there anything causal left? Frame
   the frog as a toy organism whose behavior we will explain causally.
2. **Setting the formalism.** The paper is IIT-3.0 actual causation. Show the
   config the analysis requires — built from the `iit3` preset with the
   frog-specific choices (`WEDGE_TRIPARTITION` mechanism partitions, the `AID`
   measure, `WPMI` α-measure) — inside a `config.override` block, with a
   sentence on why each is set.
3. **F3 in depth.** Build `frog_substrate("F3")` / `frog_transition("F3")`,
   compute `pyphi.actual.account(...)`, and walk through the result: the
   first-order (micro) causal links *and* the composite second-order links
   (e.g. a pair of sensors jointly causing a central unit) that a purely
   first-order account cannot represent. This is the paper's central
   demonstration — the composite causes reductionism misses.
4. **Comparing F2 and F1.** Build the other two species, compute their
   accounts, and show how the causal structure changes with the wiring — the
   paper's cross-species comparison, kept lighter than F3.
5. **Takeaway.** Reductionism is blind to composite mechanisms; the
   actual-causation account makes their causes explicit.
6. **Where to go next.** Link the actual-causation theory/reference and the
   relevant how-tos.

Every code cell executes at build time (the whole-notebook AC computation for
all three species runs in well under a second, so the 300 s page timeout is not
a concern). Numbers shown in prose are taken from the executed output.

## Out of scope

- Reproducing every figure/panel of the paper verbatim; the tutorial
  demonstrates the core argument, not a figure-by-figure replica.
- Any change to the actual-causation implementation itself.

## Verification

- `frog_substrate(s)` and `frog_transition(s)` build for `s in {F1,F2,F3}`
  under the **default** config; `account()` on each transition succeeds under
  the documented IIT-3.0 override and matches the paper's structure.
- The tutorial builds green under `-W` with every cell executed; the paired
  output-free `.ipynb` is generated and committed.
- The old `frog_example` symbol is gone and nothing references it.
- `just` doc build and the test suite stay green.
