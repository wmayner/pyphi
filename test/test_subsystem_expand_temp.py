import pytest
import numpy as np

# TODO: move back to test.test_subsystem_expand when system-level
# subsystem methods are fixed


def test_expand_repertoire_purview_can_be_None(s):
    mechanism = (0, 1)
    purview = None
    cause_repertoire = s.cause_repertoire(mechanism, purview)
    # None purview gives same results as '()' purview
    assert np.array_equal(
        s.expand_repertoire('past', purview, cause_repertoire),
        s.expand_repertoire('past', (), cause_repertoire))


def test_expand_repertoire_purview_must_be_subset_of_new_purview(s):
    mechanism = (0, 1)
    purview = (0, 1)
    new_purview = (1,)
    cause_repertoire = s.cause_repertoire(mechanism, purview)
    with pytest.raises(ValueError):
        s.expand_repertoire('past', purview, cause_repertoire, new_purview)
