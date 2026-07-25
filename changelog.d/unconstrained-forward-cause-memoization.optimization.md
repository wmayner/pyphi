`unconstrained_forward_cause_repertoire` is no longer memoized. The work it
depends on, `forward_cause_repertoire`, has its own cache; what the outer
function adds is a mean, an allocation, and a fill, worth about 4 µs against the
0.7 µs a cache lookup costs — and that margin stays flat as mechanisms grow,
since the purview sets it. Reaching the cache at all needs the same
`(mechanism, purview)` pair evaluated twice, which does not happen within an
analysis: `intrinsic_information` is called once per pair. On a scoped
cause-effect structure sweep it accumulated 30,625 entries and served **zero**
hits.

Its effect-direction counterpart keeps its cache. That one averages a forward
effect repertoire over every mechanism state, so a hit is worth 10× a lookup at
a one-unit mechanism and 99× at three, growing with mechanism size.
