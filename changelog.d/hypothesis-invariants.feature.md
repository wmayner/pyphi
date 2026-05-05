Add Hypothesis property-based invariant tests (``test/test_invariants_hypothesis.py``)
covering 19 invariants from the IIT 4.0 paper across repertoires (sum-to-one,
non-negativity, correct shape, empty-mechanism = unconstrained), partition
counts (closed-form formulas for bipartitions and directed bipartitions),
direction duality, and metric properties (EMD non-negativity and identity of
indiscernibles, IIT 3.0 MIP phi non-negativity). Extends ``test/hypothesis_utils.py``
with PyPhi-specific strategies (``small_network``, ``small_subsystem``,
``mechanism_purview_pair``) for generating random binary networks.
