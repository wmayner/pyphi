Added `pyphi.landscape`: continuous-parameter analysis of IIT quantities
over substrate space. `landscape_section(builder, state, grid)` evaluates the
SIA along a 1-D parameter axis into a tidy DataFrame that tracks φ, the
identity of every discrete selection (MIP partition, specified states), the
selection margins, and selection-regime boundaries; `perturb(builder, state,
theta)` estimates local derivatives by central finite differences, including
one-sided derivatives, margin derivatives, and `switch_distances` — the
linearized distance in parameter units to each kind of selection switch.
`weight_axis` builds a parameter axis varying one connection weight of a
`substrate_generator` substrate. All three are exported at the top level.
