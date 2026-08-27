`pyphi.utils.all_states(())` now yields the single empty state instead of
nothing — the empty product has exactly one assignment — fixing crashes on
computations over empty unit sets and fully-clamped systems.
