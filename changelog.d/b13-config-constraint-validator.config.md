Added eager config-combination validation (roadmap B13). Cross-field
constraints in :mod:`pyphi.conf.constraints` now reject silently-wrong
combinations of individually-valid config options at configuration time —
on ``config.override(...)`` and ``config.load_yaml(...)`` — with a
``ConfigurationError`` naming the two conflicting fields and a concrete fix,
instead of failing deep in the math at compute time (or not at all). The
initial constraint makes the existing reactive ``check_measure_compatible``
boundary eager: a measure paired with an IIT version that doesn't define it
(e.g. ``IIT_3_0`` with ``INTRINSIC_INFORMATION``, or ``IIT_4_0_2023`` with
``EMD``) is rejected up front. Validation is conservative — every shipped
preset passes, and only combinations confirmed wrong are flagged (notably,
``IIT_4_0_2023`` + ``INTRINSIC_INFORMATION`` is *allowed*: the Eq. 23 cap is
keyed on the measure, not the version, so it correctly applies the cap).
Opt out with ``config.override(validate_config=False)`` (new
``infrastructure.validate_config`` flag, default ``True``). A rejected
override or load restores the prior config rather than leaving a half-applied
state.
