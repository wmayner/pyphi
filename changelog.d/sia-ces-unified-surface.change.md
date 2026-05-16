Unified the cross-formalism surface of system-irreducibility analysis
results.

- Added :class:`~pyphi.models.protocols.SIAInterface`,
  :class:`~pyphi.models.protocols.CauseEffectStructureInterface`, and
  :class:`~pyphi.models.protocols.AcSIAInterface` Protocols
  (``runtime_checkable``). Both formalisms' SIA classes implement
  ``SIAInterface``; ``CauseEffectStructure`` is used by both formalisms
  (IIT 3.0 wraps an empty :class:`~pyphi.relations.NullRelations`);
  :class:`~pyphi.models.actual_causation.AcSystemIrreducibilityAnalysis`
  implements ``AcSIAInterface``.

- All four classes now share a common ``__repr__`` template via
  :func:`~pyphi.models.fmt.fmt_sia_columns`,
  :func:`~pyphi.models.fmt.fmt_ces_columns`,
  :func:`~pyphi.models.fmt.fmt_ac_sia_columns`. Formalism-specific
  extras (IIT 4.0's ``normalized_phi``, ``intrinsic_differentiation``,
  etc.) extend the shared column list.

- Added ``_repr_html_`` to SIA / CES / AcSIA for Jupyter rendering. Same
  column source as the text ``__repr__``, so text and HTML stay in
  sync.

- ``__eq__`` on each class now returns ``NotImplemented`` when ``other``
  is a different type. Cross-class comparisons (``iit3_sia == iit4_sia``)
  evaluate to ``False`` without exception, with the type-mismatch made
  explicit at the dunder level.

- The canonical JSON shape for each result type is documented in
  :mod:`pyphi.jsonify`'s module docstring. A future msgspec migration
  adopts these shapes via tagged unions keyed on ``__class__``.
