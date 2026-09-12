# FAQ and troubleshooting

## Why is φ_s zero?

Two reasons, and `analysis.sia.explain()` tells them apart. Either one side
of the system is reducible outright (some partition makes no difference to
its cause or its effect repertoire), or, under the default IIT 4.0 (2026),
the system provides itself no repertoire of alternatives, so its intrinsic
information is zero: every deterministic network is in this second case.
{doc}`read-result` walks through both; {doc}`../theory/intrinsic-information`
explains the requirement. To see the value without it, pass
`formalism="IIT_4_0_2023"`.

## My numbers differ from a paper, or from PyPhi 1.x

Check the formalism first: `analysis.formalism`. Published IIT 4.0 values
from 2023 need `formalism="IIT_4_0_2023"`; 1.x values need
`formalism="IIT_3_0"` (1.x's Φ is `analysis.phi` under that formalism,
not `analysis.big_phi`). {doc}`../theory/formalism-versions` lists what
each changes; {doc}`../migration/migration-2.0` covers 1.x. If the formalism
matches and a value still differs in the last decimals, compare with
`pyphi.numerics.eq`, which respects the configured precision.

## `StateUnreachableForwardsError`

The state you asked about cannot be produced by the substrate's own
dynamics: no current state leads to it. IIT evaluates a system in a state
it could have reached, so PyPhi refuses rather than assigning a structure
that rests on an impossible past. Deterministic toy networks hit this
often: in the three-XOR network every reachable state has even parity.
Choose a reachable state (`pyphi.sweep(substrate, states="all").skipped`
lists the unreachable ones) or check the matrix's row order
({doc}`build-substrate`).

## `ConditionallyDependentError`

The matrix says two units' next states depend on each other at the same
time step, which a substrate of conditionally independent units cannot
express; it signals a hidden common cause. Add the shared variable as a
unit, or rebuild the matrix from each unit's own input–output function.
{doc}`../theory/conditional-independence`.

## The estimate says `capped=True`

The counting walk stopped at its budget; the counts are lower bounds. Raise
`limit` for an exact count, or read the counts as "at least this much" and
reduce the work: {doc}`estimate-cost`.

## The analysis is taking hours

It is probably past the practical ceiling: about 10–12 units for φ_s and
6–8 for the full Φ-structure on a fully connected substrate. Stop it,
count the work with `pyphi.cost.estimate_analysis`, and reduce it
({doc}`estimate-cost`). PyPhi has no checkpointing; a killed run loses its
progress, so for long sweeps turn on the disk result cache
({doc}`cache`).

## The result says "effectively tied"

Two partitions or two specified states came within the configured
precision of each other; small symmetric networks do this constantly. The
selection is still deterministic and follows the postulates, and the
margins say how close it was. {doc}`tie-breaking`.

## Two calls with the same inputs gave different results

Something in the configuration changed between them: an `override` block
still open, a `pyphi_config.yml` in one working directory and not the
other, or a preset applied in one session. `print(pyphi.config)` shows the
active settings; pin the formalism per call with `formalism=`.
{doc}`configure`.

## Where is the API reference?

Under {doc}`Reference </reference/index>` on the built site (it is
generated from the docstrings and is not in the source tree), and in the
interpreter with `help()`.
