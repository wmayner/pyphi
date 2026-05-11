from pyphi import metrics


def test_default_distribution_measures():
    all_metric_names = (
        set(metrics.distribution.distribution_measures.all())
        | set(metrics.distribution.state_aware_measures.all())
        | set(metrics.distribution.composite_measures.all())
        | set(metrics.distribution.stateful_distribution_measures.all())
    )
    assert all_metric_names == {
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
        "INTRINSIC_DIFFERENTIATION",
        "INTRINSIC_INFORMATION",
        "INTRINSIC_SPECIFICATION",
    }


def test_default_asymmetric_distribution_measures():
    asymmetric_names = (
        {
            name
            for name, fn in metrics.distribution.distribution_measures.items()
            if getattr(fn, "asymmetric", False)
        }
        | {
            name
            for name, fn in metrics.distribution.stateful_distribution_measures.items()
            if getattr(fn, "asymmetric", False)
        }
        | {
            name
            for name, fn in metrics.distribution.composite_measures.items()
            if getattr(fn, "asymmetric", False)
        }
    )
    assert asymmetric_names == {
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
        "INTRINSIC_INFORMATION",
        "INTRINSIC_SPECIFICATION",
    }


def test_default_ces_measures():
    assert set(metrics.ces.measures.all()) == {
        "EMD",
        "SUM_SMALL_PHI",
    }


def test_default_actual_causation_measures():
    assert set(metrics.distribution.actual_causation_measures.all()) == {
        "PMI",
        "WPMI",
    }
