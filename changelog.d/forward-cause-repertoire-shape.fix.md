Fixed two `forward_cause_repertoire` defects: an unsorted purview tuple no
longer yields values at transposed coordinates (results are now
purview-order-independent, matching the other repertoire functions), and an
empty purview returns the multiplicative-identity repertoire `[1.0]` (matching
the `cause_repertoire` convention) instead of an integer-truncated `[0]`.
