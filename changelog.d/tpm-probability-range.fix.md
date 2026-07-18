TPM validation now rejects probabilities outside [0, 1] (within the numerics
tolerance); out-of-range TPMs previously constructed successfully and silently
produced φ values.
