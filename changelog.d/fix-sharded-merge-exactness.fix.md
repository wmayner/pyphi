Sharded campaign merges are now exact. Campaign strides report every
specified-state candidate's local minimum — one entry per pin at the
distinction level, one per (cause, effect) pair at the SIA level — so the
merge takes the cross-stride minimum per candidate before running the same
selection the unsharded search runs. Previously each stride reported only its
local winners, so a sharded campaign could report a reducible distinction as
real (φ = 0.2075 where the full sweep gives φ = 0), select a different system
MIP, or resolve congruence against a different specified system state. Under
IIT 4.0 (2026) the intrinsic-information cap is now applied at merge time,
after the global MIP per pair is chosen, matching the unsharded definition.
