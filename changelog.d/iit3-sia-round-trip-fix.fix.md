**Fixed IIT 3.0 SIA JSON round-trip.** Two bugs prevented the IIT 3.0
``SystemIrreducibilityAnalysis`` from round-tripping:

- Its ``to_json`` / ``from_json`` referenced a non-existent
  ``small_phi_time`` attribute, so any serialization raised
  ``AttributeError``.
- It shared a class name with the IIT 4.0
  ``SystemIrreducibilityAnalysis``, so the
  ``pyphi.jsonify._loadable_models()`` registry (keyed by
  ``cls.__name__``) silently kept only the IIT 4.0 class. Any
  hypothetical IIT 3.0 SIA JSON would deserialize into an IIT 4.0
  object.

The IIT 3.0 class is renamed to ``IIT3SystemIrreducibilityAnalysis`` to
disambiguate. Existing IIT 4.0 fixtures keep working unchanged (they
still use the bare ``SystemIrreducibilityAnalysis`` marker, which now
unambiguously resolves to the IIT 4.0 class). The IIT 3.0
``to_json`` / ``from_json`` reference ``_sia_attributes`` only. A new
test in ``test/test_json.py::test_iit3_sia_round_trip`` exercises the
full encode/decode cycle and asserts the round-trip equality.

Public-API impact: ``pyphi.models.SystemIrreducibilityAnalysis`` is now
``pyphi.models.IIT3SystemIrreducibilityAnalysis``. The IIT 4.0 class at
``pyphi.formalism.iit4.SystemIrreducibilityAnalysis`` is unchanged.
