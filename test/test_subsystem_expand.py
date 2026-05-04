import pytest

from pyphi import Direction

# NOTE: test_expand_cause_repertoire and test_expand_effect_repertoire were removed
# because they relied on Concept.expand_cause_repertoire and
# Concept.expand_effect_repertoire methods which were removed during the
# IIT 3.0 -> 4.0 migration. These methods now exist only on Subsystem, not on Concept.


def test_expand_repertoire_purview_must_be_subset_of_new_purview(s):
    mechanism = (0, 1)
    purview = (0, 1)
    new_purview = (1,)
    cause_repertoire = s.cause_repertoire(mechanism, purview)
    with pytest.raises(ValueError):
        s.expand_repertoire(Direction.CAUSE, cause_repertoire, new_purview)
