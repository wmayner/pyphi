import pytest

import pyphi
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


@pytest.mark.parametrize("query", ["cause_info", "effect_info", "cause_effect_info"])
def test_info_queries_reject_an_unknown_keyword(s, query):
    with pytest.raises(TypeError, match="bogus"):
        getattr(s, query)((0, 1), (0, 2), bogus=1)


@pytest.mark.parametrize("query", ["cause_info", "effect_info", "cause_effect_info"])
def test_info_queries_name_the_iit4_quantities_outside_iit3(s, query):
    with (
        pyphi.config.override(**pyphi.iit4_2026),
        pytest.raises(TypeError, match=r"IIT 3\.0 quantity.*System\.mic\(\)"),
    ):
        getattr(s, query)((0, 1), (0, 2))


@pytest.mark.emd
@skip_if_no_emd_backend
@pytest.mark.parametrize("query", ["cause_info", "effect_info", "cause_effect_info"])
def test_info_queries_run_under_iit3(s, query):
    with pyphi.config.override(**pyphi.iit3):
        value = getattr(s, query)((0, 1), (0, 2))
    assert value >= 0
