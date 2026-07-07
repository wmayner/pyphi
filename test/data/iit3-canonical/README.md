# IIT 3.0 Canonical Reference Values

This directory holds independently-verified canonical values for the IIT 3.0
formalism's outputs on PyPhi's standard substrates. They are the *correctness
baseline* against which fixture regeneration must be cross-checked, distinct
from the *behavioral snapshots* in `test/data/golden/v1/`.

## Files

- `basic_sia_phi_canonical.json` — `pyphi.examples.basic_substrate()` in
  state `(1, 0, 0)`. The canonical IIT example.
  - **Canonical target**: `sia.phi = 2.3125`, CES has 4 distinctions
    `[B, C, AB, ABC]`, MIP cut `[B, C] -/-> [A]`.
  - This is the value PyPhi's test suite has asserted since 2015-05-13.

- `background_conditioning_oracle.json` — anchors the cause-side
  background-conditioning conventions (`CONDITION_CURRENT_STATE` vs
  `CAUSAL_MARGINALIZATION`) to a genuine PyPhi 1.2.0 install. Reproduced by
  `scripts/gen_iit3_background_oracle.py`; consumed by
  `test/integration/test_background_conditioning_oracle.py`. Records, on
  the proper-subset system S={A,B} (background W={C}) of a 3-unit
  noisy-OR substrate in state `(1, 0, 0)`: the observed cause repertoire
  of mechanism {A} over purview {B} under genuine 1.x semantics
  (`[0.1, 0.9]`, matching `CONDITION_CURRENT_STATE`) alongside the IIT 4.0
  Eq. 4 marginalized prediction (`[0.40566..., 0.59434...]`) that 1.x does
  *not* match; the end-to-end IIT 3.0 SIA phi for that system (`0.72`,
  matching `CONDITION_CURRENT_STATE`); and SIA phi for every proper
  subset of the `basic` example network in state `(1, 0, 0)` as
  independent anchors for the complex-search values (`(0, 1, 2)`: 2.3125,
  `(1, 2)`: 1.0, `(0, 2)`: 1.0, `(0, 1)`: 0.0).

## The 1.917 vs 2.3125 history

Both the IIT 3.0 paper (Oizumi 2014, Fig 14/15) and the PyPhi paper (Mayner
2018, Fig 1 and p.12 code listing) report `Φ = 1.92` (precisely `1.916666...`)
for this substrate. PyPhi's test suite has asserted `2.3125` since
2015-05-13.

Both can be correct under their respective formalism choices. The 2015-05-12
commit `9fc0c0ab` ("Ensure no concepts are moved around within a
constellation in EMD") changed the diagonal blocks of the EMD distance
matrix from zero to `max(pairwise_distance) + 1`, enforcing that EMD mass
can only flow between the unpartitioned and partitioned constellations or
to the null concept, never within. The theoretical motivation: concepts in a
constellation are not interchangeable. The numerical effect: `Φ` increases
from `1.917` (paper) to `2.312` (post-fix).

The current `_emd` implementation in `pyphi/metrics/ces.py:194-213`
preserves this 2015 refinement. PyPhi's canonical IIT 3.0 answer is
therefore `2.3125`, not the paper's `1.917`.

## Open question

How the PyPhi paper (2018) produced `1.916665` when PyPhi already produced
`2.312` since 2015 is not fully explained. Most likely the paper's
demonstration was generated against a pre-2015 PyPhi snapshot. The fields
`historical_provenance.sources` in the JSON document the candidates.

## Consumer

These values are the bisect target for hunting the 2024-2026 regression
that drives the current observed `sia.phi = 0.5`. The bisect predicate
script in `phase_4_bisect_predicate.predicate_python` is consumable as-is.
