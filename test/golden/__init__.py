"""Golden regression harness for PyPhi (P1).

This package provides infrastructure for capturing raw numerical outputs from
PyPhi computations and comparing them against pinned reference values across
refactors. The format is intentionally independent of ``pyphi.jsonify`` so that
fixtures survive any future serialization rewrite.

Usage::

    @pytest.mark.golden
    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=lambda f: f.name)
    def test_golden_regression(fixture):
        fixture.assert_matches(fixture.compute())

To add a new fixture, edit ``test/golden/zoo.py`` and run::

    uv run pytest test/test_golden_regression.py --regenerate-golden -k <name>

Fixture data layout::

    test/data/golden/v1/<fixture_name>.json     # human-readable structured data
    test/data/golden/v1/<fixture_name>.npz      # binary arrays (repertoires etc.)
"""

from .canonicalize import canonical_partition
from .canonicalize import canonical_state_set
from .fixture import GoldenFixture
from .fixture import load_fixture
from .fixture import store_fixture
from .zoo import ALL_FIXTURES

__all__ = [
    "ALL_FIXTURES",
    "GoldenFixture",
    "canonical_partition",
    "canonical_state_set",
    "load_fixture",
    "store_fixture",
]
