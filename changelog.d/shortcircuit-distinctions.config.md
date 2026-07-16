Added `formalism.iit.shortcircuit_distinctions` (default `True`): distinction
evaluation now skips the remaining MICE search when the distinction is already
known to be reducible — an empty candidate effect-purview set, or a cause MICE
with φ = 0. The skipped direction is a null MICE carrying the new
`OTHER_DIRECTION_REDUCIBLE` reason; set the option to `False` to always
evaluate both directions in full (exact selection margins and complete ties).
