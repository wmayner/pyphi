Null IIT 4.0 system irreducibility analyses no longer fabricate an
`intrinsic_differentiation` of zero. A null SIA carrying a real
`system_state` previously reported `intrinsic_information == 0.0` as if it
had been computed, even when the true ii(s) is nonzero; both fields now
report `None` when the intrinsic differentiation was not computed.
