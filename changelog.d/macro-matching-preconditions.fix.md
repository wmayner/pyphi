Public macro/matching entry points now validate their preconditions
instead of returning silently wrong results: `build_triggered_tpm` and
`PerceptualSystem` reject non-binary substrates (a current
implementation limit) and unsorted index tuples; triggered-TPM queries
accept the mechanism in any order by canonicalizing the
(mechanism, state) pairs; `macro_tpms` validates history length and
unit disjointness (a one-entry-short history previously wrapped around
and produced a wrong cause TPM); and `MacroUnit` rejects negative
background-apportionment indices.
