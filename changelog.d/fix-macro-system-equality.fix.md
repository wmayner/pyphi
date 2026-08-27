A `MacroSystem` no longer compares equal to a plain `System` over its macro
substrate. The two are different analyses (the macro construction overrides
the cause TPM and yields a different φ), but the fallback comparison saw only
the shared fields, breaking the equality/hash contract and making set and
dict membership inconsistent.
