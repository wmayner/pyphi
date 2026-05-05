**Breaking (2.0):** Replace ``config.IIT_VERSION`` with ``config.FORMALISM``.

The IIT version + repertoire-distance double-dispatch in
``Subsystem.find_mip`` and ``Subsystem.sia`` is gone. The active formalism
(selected by string name in ``config.FORMALISM``) owns mechanism MIP search
and system-level SIA; ``Subsystem`` is a thin dispatcher.

Migration::

    # before
    pyphi.config.IIT_VERSION = 3.0    # or 4.0
    # after
    pyphi.config.FORMALISM = "IIT_3_0"   # or "IIT_4_0_2023" or "IIT_4_0_2026"

``config.IIT_VERSION`` is removed. Loading a ``pyphi_config.yml`` with
``IIT_VERSION:`` raises ``ConfigurationError`` at config-load time.

The 2026 IIT 4.0 variant (``ii(s) = min(i_diff, i_spec)`` cap from Eq. 23
of Mayner, Marshall, Tononi 2026) gets its own formalism class
(``IIT_4_0_2026``) instead of being inferred from
``REPERTOIRE_DISTANCE = "INTRINSIC_INFORMATION"``.

``config.REPERTOIRE_DISTANCE`` remains for now — its broader removal is
part of the metric-API unification work.
