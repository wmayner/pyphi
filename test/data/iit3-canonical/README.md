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
