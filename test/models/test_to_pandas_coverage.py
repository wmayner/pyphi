"""N9: every Displayable result type exports to a labeled pandas structure."""

from __future__ import annotations

import pandas as pd

from pyphi import examples
from pyphi.conf import config
from pyphi.conf import presets


def test_sia_to_pandas_series_has_phi():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    s = sia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)


def test_iit3_sia_to_pandas_series_has_phi():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit3):
        sia = substrate.sia(state)
    s = sia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)


def test_complex_and_excluded_to_pandas_series():
    from pyphi.models.complex import Complex
    from pyphi.models.complex import ExcludedCandidate

    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    cx = Complex(sia=sia, substrate=substrate, is_maximal=True, excluded=())
    s = cx.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["phi"]) == float(sia.phi)
    assert bool(s["is_maximal"]) is True

    ec = ExcludedCandidate(node_indices=(0, 1), phi=0.0)
    es = ec.to_pandas()
    assert isinstance(es, pd.Series)
    assert tuple(es["node_indices"]) == (0, 1)


def test_ces_to_pandas_dataframe_of_distinctions():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        ces = substrate.ces(state)
    df = ces.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(ces.distinctions)


def test_relations_to_pandas_dataframe():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        ces = substrate.ces(state)
    relations = ces.relations
    df = relations.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"phi", "degree"}
    assert len(df) == relations.num_relations()


def _or_gate_transition():
    import numpy as np

    from pyphi import actual
    from pyphi.substrate import Substrate

    tpm = np.array(
        [
            [0, 0.5, 0.5],
            [0, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
            [1, 0.5, 0.5],
        ]
    )
    cm = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])
    substrate = Substrate(tpm, cm)
    return actual.Transition(substrate, (0, 1, 1), (1, 0, 0), (1, 2), (0,))


def test_ac_family_to_pandas():
    from pyphi import actual
    from pyphi.direction import Direction

    transition = _or_gate_transition()
    acsia = actual.sia(transition)
    s = acsia.to_pandas()
    assert isinstance(s, pd.Series)
    assert float(s["alpha"]) == float(acsia.alpha)

    account = actual.account(transition, Direction.CAUSE)
    df = account.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(account)

    link = account[0]
    ls = link.to_pandas()
    assert isinstance(ls, pd.Series)
    assert float(ls["alpha"]) == float(link.alpha)
    assert isinstance(link._ria.to_pandas(), pd.Series)


def test_factored_tpm_to_pandas_state_by_node():
    substrate = examples.basic_substrate()
    tpm = substrate.factored_tpm
    df = tpm.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == tpm.n_nodes  # one column per unit (binary)


def test_substrate_to_pandas_delegates_to_tpm():
    substrate = examples.basic_substrate()
    df = substrate.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == substrate.factored_tpm.n_nodes


def test_system_to_pandas_per_unit_state():
    from pyphi import System

    substrate = examples.basic_substrate()
    state = examples.basic_state()
    system = System(substrate, state)
    df = system.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(system.node_indices)
    assert "state" in df.columns


def test_partition_to_pandas_cut_grid():
    substrate = examples.basic_substrate()
    state = examples.basic_state()
    with config.override(**presets.iit4_2023):
        sia = substrate.sia(state)
    partition = sia.partition
    df = partition.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == df.shape[1]  # square from x to grid


def test_unitstate_to_pandas_series():
    from pyphi.models.state_specification import UnitState

    us = UnitState(index=1, state=1, label="B")
    s = us.to_pandas()
    assert isinstance(s, pd.Series)
    assert int(s["state"]) == 1
