from pyphi import metrics


def test_default_distribution_measures():
    assert set(metrics.distribution.measures.all()) == set(
        [
            "EMD",
            "L1",
            "KLD",
            "ENTROPY_DIFFERENCE",
            "PSQ2",
            "MP2Q",
            "AID",
            "KLM",
            "BLD",
            "ID",
            "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
            "IIT_4.0_SMALL_PHI",
            "APMI",
            "GENERALIZED_INTRINSIC_DIFFERENCE",
        ]
    )


def test_default_asymmetric_distribution_measures():
    assert set(metrics.distribution.measures.asymmetric()) == set(
        [
            "IIT_4.0_SMALL_PHI_NO_ABSOLUTE_VALUE",
            "IIT_4.0_SMALL_PHI",
            "APMI",
            "KLD",
            "MP2Q",
            "AID",
            "KLM",
            "BLD",
            "ID",
            "GENERALIZED_INTRINSIC_DIFFERENCE",
        ]
    )


def test_default_ces_measures():
    assert set(metrics.ces.measures.all()) == set(
        [
            "EMD",
            "SUM_SMALL_PHI",
        ]
    )


def test_default_actual_causation_measures():
    assert set(metrics.distribution.actual_causation_measures.all()) == set(
        [
            "PMI",
            "WPMI",
        ]
    )
