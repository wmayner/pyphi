Renamed user-facing types to match IIT 4.0 paper terminology.
``pyphi.Network`` is now ``pyphi.Substrate``; ``pyphi.Subsystem`` is now
``pyphi.System``. The ``pyphi.network_generator`` package is now
``pyphi.substrate_generator``. Configuration keys ``CLEAR_SUBSYSTEM_*`` and
``VALIDATE_SUBSYSTEM_*`` are now ``CLEAR_SYSTEM_*`` / ``VALIDATE_SYSTEM_*``.
The ``compute`` module's relocated names follow the same pattern:
``compute.network`` is now ``pyphi.substrate``-level helpers (``systems``,
``reachable_systems``, ``possible_complexes``).
