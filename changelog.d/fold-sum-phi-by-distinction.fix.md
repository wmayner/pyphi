Fixed `AnalyticalFoldRelations.sum_phi_by_distinction` returning wrong (even
negative) values: it inherited the full-structure formula, subtracting a
whole-structure total from the fold's incident total. The fold now differences
against the fold of the remaining distinctions with the seed set restricted
accordingly, matching the concrete incident relations exactly.
