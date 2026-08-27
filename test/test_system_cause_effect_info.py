import pytest

from pyphi.measures.distribution import hamming_emd

from .conftest import IIT_3_CONFIG
from .conftest import skip_if_no_emd_backend


@pytest.mark.emd
@skip_if_no_emd_backend
@IIT_3_CONFIG
def test_cause_info(s):
    mechanism = (0, 1)
    purview = (0, 2)
    answer = hamming_emd(
        s.cause_repertoire(mechanism, purview),
        s.unconstrained_cause_repertoire(purview),
    )
    assert s.cause_info(mechanism, purview) == answer


@pytest.mark.emd
@skip_if_no_emd_backend
@IIT_3_CONFIG
def test_effect_info(s):
    mechanism = (0, 1)
    purview = (0, 2)
    answer = hamming_emd(
        s.effect_repertoire(mechanism, purview),
        s.unconstrained_effect_repertoire(purview),
    )
    assert s.effect_info(mechanism, purview) == answer


@pytest.mark.emd
@skip_if_no_emd_backend
@IIT_3_CONFIG
def test_cause_effect_info(s):
    mechanism = (0, 1)
    purview = (0, 2)
    answer = min(s.cause_info(mechanism, purview), s.effect_info(mechanism, purview))
    assert s.cause_effect_info(mechanism, purview) == answer
