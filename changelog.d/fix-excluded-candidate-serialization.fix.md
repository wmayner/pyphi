Complexes found with the certified intrinsic-information prune (the default
under the 2026 formalism) could not be saved: gated excluded candidates carry
`phi=None`, which crashed the serializer. Gated candidates now serialize, and
the certification record (`ii_ceiling`, `gated`) survives the round-trip —
previously it was silently dropped, and `ExcludedCandidate` equality did not
compare the ceiling for measured candidates, so the loss went unnoticed.
