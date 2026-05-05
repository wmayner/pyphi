Add ``pyphi.protocols`` with runtime-checkable Protocols for PyPhi's
dispatch points and core abstractions:

- ``DistanceMetric``: contract for distance functions registered in
  ``pyphi.metrics.distribution.measures``.
- ``PartitionScheme``: contract for partition schemes registered in
  ``pyphi.partition.partition_types``.
- ``PhiFormalism``: placeholder contract for the strategy that bundles
  partition scheme + distance metric + algorithms (concrete shape lands
  with the IIT 3.0 / 4.0 split).
- ``SubsystemPublicInterface``: cross-module contract that ``Subsystem``
  exposes — generated from ``dir(Subsystem)`` plus instance attributes,
  filtered to names actually accessed outside ``pyphi/subsystem.py``,
  ``macro.py``, and ``actual.py``.
- ``SubsystemInternalInterface``: members used only inside the subsystem
  family, listed for visibility but free to change.

Both partition and distribution-measure registries now validate registered
objects against their Protocols at registration time. The
``test/test_subsystem_surface.py`` suite fails CI when ``Subsystem``'s
public surface drifts from the Protocol declaration, so adding or
removing a public attribute is an explicit change to the contract.
