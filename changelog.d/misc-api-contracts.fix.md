`CompositionalState.has_conflicts(mechanism=...)` reports conflicts for the
given mechanism instead of any mechanism; `simulate(timesteps=None)` with a
single clamp raises a clear error; analytical relation computation rejects
unsupported keyword arguments instead of silently ignoring them; structure
views raise a clear error from `.save()`; cause-side repertoires reject state
overrides loudly and `forward_repertoire` threads `mechanism_state` on the
cause side; `optimize` retains only the best candidate's SIA, bounding memory
on long runs.
