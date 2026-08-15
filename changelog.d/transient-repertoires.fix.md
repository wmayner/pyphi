The full-state repertoire sweeps — the unconstrained forward effect repertoire
and the forward cause repertoire — no longer store their per-state
intermediates in the kernel cache. Each intermediate is a full repertoire read
exactly once, so caching them cost the product of the state count and the
repertoire size (order 4ⁿ cells for an n-unit system) for no hits at all: a
16-unit specified-state search exhausted memory before finishing. The sweeps
now run under the new `pyphi.core.repertoire_algebra.transient_repertoires`
scope, which returns computed repertoires without admitting them. Peak memory
for a 14-unit search falls from 1.2 GiB to 0.14 GiB, and a 16-unit search
completes in 0.19 GiB where it previously ran out of memory.

The size bound on full-state sweeps now covers the cause direction as well as
the effect one. Only the unconstrained forward effect repertoire checked it,
and `Direction.both()` walks the cause direction first, so an oversized system
spent its entire cause sweep before anything refused.
