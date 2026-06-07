Added `pyphi.dynamics.settle`, the deterministic complement to `simulate`: it
iterates the most-probable-transition map to a fixed point and returns the
trajectory (the fixed point is the last element; the settling time is its
length minus one). Raises `NonConvergenceError` if the trajectory enters a
limit cycle. Supports `clamp` (units held fixed each step), reusing the
existing clamp machinery.
