IIT 3.0 ``iit3.ces()`` now returns a
:class:`~pyphi.models.ces.CauseEffectStructure` wrapping the SIA, the
distinctions, and an empty :class:`~pyphi.relations.NullRelations`.
This matches IIT 4.0's topology where the CES is the dominant outer
container; previously IIT 3.0 returned a bare ``UnresolvedDistinctions``
and the SIA stored ``ces`` / ``partitioned_ces`` fields directly.

Related changes on :class:`~pyphi.models.sia.IIT3SystemIrreducibilityAnalysis`:

- ``ces`` field removed. Read the unpartitioned distinctions from
  ``iit3.ces(system).distinctions``.
- ``substrate`` field removed. Callers hold the substrate reference
  externally, matching IIT 4.0's decoupled SIA.
- ``partitioned_ces`` renamed to ``partitioned_distinctions``. This bag
  stays on the SIA as the IIT 3.0-specific compute receipt — IIT 3.0
  phi is computed as ``ces_distance(unpartitioned, partitioned,
  system)``, so the partitioned bag is intrinsic to the computation.
- ``unorderable_unless_eq=["substrate"]`` removed. Comparisons between
  SIAs from different substrates now fall back to phi-ordering.

The :func:`~pyphi.models.fmt.fmt_sia` helper no longer accepts a ``ces``
kwarg; the SIA's ``__str__`` renders just the SIA fields. Use
``iit3.ces(system)`` to render the full cause-effect structure.

Adds :class:`~pyphi.relations.NullRelations`, an empty
:class:`~pyphi.relations.Relations` subclass with ``sum_phi() == 0``,
``num_relations() == 0``, and empty iteration. Used by IIT 3.0's
:class:`~pyphi.models.ces.CauseEffectStructure` to indicate that the
formalism does not define relations between distinctions.

The 4 IIT 3.0 EMD golden fixtures regenerate with the new shape.
``sia.phi`` values are unchanged.
