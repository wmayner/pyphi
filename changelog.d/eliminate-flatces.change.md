**Breaking (2.0):** Remove ``FlatCauseEffectStructure`` and the
``CauseEffectStructure.flatten()`` / ``unflatten()`` round-trip. ``FlatCES``
was a subclass of ``CES`` whose container element type differed (``MICE``
instead of ``Concept``); the inheritance was a kludge that produced
Liskov-incompatible method/property overrides on ``purviews`` and ``flat``.
Its only production callers (``pyphi.relations.all_relations``,
``pyphi.relations.AnalyticalRelations``, ``pyphi.compositional_state``)
already worked at the concept level and called ``unflatten()`` on entry to
normalize input back to ``CauseEffectStructure``. Those calls have been
removed; passing a sequence of ``MICE`` to ``relations`` is no longer
supported (use ``CauseEffectStructure`` instead). The vestigial helpers
``FlatCES.specifiers``, ``FlatCES.maximum_specifier``, and
``FlatCES.specified_purviews`` had no production callers and have been
removed.
