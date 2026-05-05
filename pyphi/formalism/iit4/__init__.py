"""IIT 4.0 formalism (Albantakis et al. 2023; Mayner, Marshall, Tononi 2026).

State-aware intrinsic-information-based phi computation, distinctions, and
relations. Two registered variants:

- ``IIT_4_0_2023``: uses ``GENERALIZED_INTRINSIC_DIFFERENCE`` as the default
  metric (Albantakis et al. 2023).
- ``IIT_4_0_2026``: uses ``INTRINSIC_INFORMATION`` with the ``ii(s) = min(i_diff,
  i_spec)`` cap from Eq. 23 of the 2026 paper.

The concrete formalism classes and the moved SIA / Φ-structure
implementations land in the next commit; this module currently exists as a
placeholder anchoring the package layout.
"""
