**Breaking (2.0):** Move ``pyphi.new_big_phi`` to ``pyphi.formalism.iit4``.
The IIT 4.0 SIA, Φ-structure, and intrinsic-information implementations now
live under the ``pyphi.formalism`` package alongside the (still-pending)
IIT 3.0 module. Update imports::

    # before
    from pyphi import new_big_phi
    from pyphi.new_big_phi import sia, phi_structure, SystemIrreducibilityAnalysis

    # after
    from pyphi.formalism import iit4
    from pyphi.formalism.iit4 import sia, phi_structure, SystemIrreducibilityAnalysis

Also register the three concrete formalism instances
(``IIT3Formalism``, ``IIT4_2023Formalism``, ``IIT4_2026Formalism``) in
``pyphi.formalism.FORMALISM_REGISTRY``. Each delegates to the existing SIA
/ Φ-structure paths; the cut-over from ``Subsystem``'s ``IIT_VERSION`` /
``REPERTOIRE_DISTANCE`` dispatch to formalism-driven dispatch follows in
the next commit.
