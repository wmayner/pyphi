The specified-state computation no longer materializes the full state
space as Python objects: `intrinsic_information` finds the winner and
tie family by vectorized argmax with a tolerance mask (`numerics.eq_mask`,
elementwise-equivalent to `numerics.eq`), and
`unconstrained_forward_effect_repertoire` accumulates a running mean
instead of stacking one repertoire per mechanism state — memory drops
from 2ⁿ full repertoires (hundreds of TB at 21 units) to one. Winner
selection, tolerance-based tie families, and runner-up semantics are
unchanged and pinned by tests. Requests whose mechanism-state count
exceeds a feasibility bound now raise immediately with the estimated
cost, and `pyphi.campaign.collect` warns before computing a missing
resolution state on systems where that computation would dominate.
