The conditional-independence check on state-by-state TPMs now runs at the
configured `numerics.precision` (absolute tolerance) instead of numpy's
loose defaults, which silently accepted dependence up to ~1e-5.
