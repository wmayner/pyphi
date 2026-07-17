Fixed `intrinsic_information` dropping specified states from the tie family
when their intrinsic-information values differed from the maximum only by
float-path noise: tie membership is now clustered within
`config.numerics.precision` (matching the tie-resolution convention), the
returned specification is the first tied state in enumeration order, and the
runner-up is the best-ranked competing state other than the winner.
