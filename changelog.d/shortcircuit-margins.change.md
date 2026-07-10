`shortcircuit_sia=False` now also disables the sweep-level short-circuit that
stops the system- and mechanism-level partition searches at the first
reducible partition. Disabling evaluates every partition — computed φ values
are unchanged, reducible cases pay the full sweep, and selection margins are
exact everywhere (including when φ_s = 0).
