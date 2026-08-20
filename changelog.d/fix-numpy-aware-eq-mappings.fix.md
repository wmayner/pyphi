Fixed structural equality on mappings: `numpy_aware_eq` compared dicts by
zipping their keys positionally, so two dicts with identical keys but
different values compared equal (and equal dicts with different insertion
order compared unequal). Mappings now compare by key set with values compared
recursively.
