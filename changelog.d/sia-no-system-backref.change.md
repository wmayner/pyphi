Removed the heavy parent-object back-references from the IIT 3.0
``SystemIrreducibilityAnalysis`` and ``AcSystemIrreducibilityAnalysis``
result types.

``SystemIrreducibilityAnalysis`` no longer carries ``.system`` or
``.partitioned_system`` ``System`` references. The metadata previously read
off the system is now stored directly: ``partition`` (lifted from the
partitioned system), ``node_indices``, ``node_labels``, ``current_state``,
and ``substrate``. ``AcSystemIrreducibilityAnalysis`` no longer carries a
``.transition`` reference; it stores ``before_state``, ``after_state``,
``size``, ``node_indices``, and ``node_labels`` directly. ``__eq__`` and
``__hash__`` now compare on the stored metadata rather than on heavy parent
object identity, so two SIAs produced from mathematically equivalent
inputs (e.g. one fresh, one round-tripped through JSON) compare equal and
hash identically. JSON output no longer embeds the full input system /
transition.

Migration: callers reading ``sia.system.node_indices`` should read
``sia.node_indices``; callers reading ``ac_sia.transition.before_state``
should read ``ac_sia.before_state``.
