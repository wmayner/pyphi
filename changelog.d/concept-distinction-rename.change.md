Renamed user-facing IIT 3.0/4.0 vocabulary to align with the IIT 4.0 paper.
The class ``pyphi.models.Concept`` has been renamed to
``pyphi.models.Distinction`` (Albantakis et al. 2023). ``Concept`` remains as
a thin alias for callers using the IIT 3.0 idiom. The query
``pyphi.formalism.concept`` (with the IIT 3.0 ``purviews=`` keyword shape)
has moved to ``pyphi.formalism.iit3.concept``; the canonical user-facing
query is ``pyphi.formalism.distinction`` (already the simpler 4.0 idiom).
``System`` gains ``ces()`` and ``phi_structure()`` convenience methods that
mirror ``sia()``; ``System.concept()`` is removed in favor of
``System.distinction()``.
