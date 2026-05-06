**Breaking (2.0):** Apply the ``|·|+`` operator (Eqs. 19-20 of the IIT 4.0
paper) to system-level integrated information.

``SystemIrreducibilityAnalysis.phi`` is now the paper-faithful, non-negative
clamped value (always ≥ 0). The raw signed value is preserved as
``signed_phi`` metadata for "preventative cause" visibility. Same for
``normalized_phi`` / ``signed_normalized_phi``.

Concretely::

    sia = subsystem.sia()
    sia.phi          # max(0, signed_phi) — paper-faithful φ
    sia.signed_phi   # raw computed value, may be negative

Add ``pyphi.utils.positive_part(x)`` (the ``|·|+`` operator) for callers
that want the same clamp.

The IIT 4.0 (2026) ``ii(s)`` cap (Eq. 23) is now applied to ``|·|+``-clamped
``i_diff`` and ``i_spec`` values, keeping the cap aligned with paper-
faithful φ throughout.

Test fixtures: grid3 (1,0,0) regenerated from ``phi = -0.0729`` to
``phi = 0.0`` with ``signed_phi = -0.0729``; the change-detection oracle
is preserved by the explicit ``signed_phi`` field.

Mechanism-level ``RepertoireIrreducibilityAnalysis`` will receive the same
treatment when ``models/mechanism.py`` is split (P8); for now it continues
to expose raw signed phi.
