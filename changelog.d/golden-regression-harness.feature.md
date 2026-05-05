Add a golden regression harness (``test/golden/``, ``test/test_golden_regression.py``)
that captures raw numerical outputs of repertoires, mechanism MIPs, system
SIAs, and full Φ-structures across 17 fixtures spanning IIT 3.0, IIT 4.0
(2023), and the IIT 4.0 (2026, intrinsic-cause-effect-power) formalisms.
Fixtures are stored in ``test/data/golden/v1/`` as paired JSON + ``.npz``
files in a format independent of ``pyphi.jsonify``. Run
``pytest test/test_golden_regression.py --regenerate-golden`` to update
fixtures after intentional formula changes.
