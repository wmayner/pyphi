#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# test_nonbinary.py

import json
from pathlib import Path

import numpy as np
import pytest

import pyphi


EXAMPLE_DIR = Path("test/data/nonbinary")


@pytest.mark.parametrize("example_path", tuple(EXAMPLE_DIR.glob("*.json")))
def test_nonbinary_example(example_path):
    with Path(example_path).open(mode="rt") as f:
        example = json.load(f)

    # TODO(nonbinary): test with other options
    with pyphi.config.override(
        PARTITION_TYPE="TRI",
        MEASURE="KLM",
        USE_SMALL_PHI_DIFFERENCE_FOR_CES_DISTANCE=True,
        ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS=True,
    ):
        network = pyphi.Network(
            example["tpm"],
            cm=example["cm"],
            num_states_per_node=example["num_states_per_node"],
        )
        subsystem = pyphi.Subsystem(network, example["state"])
        sia = pyphi.compute.sia(subsystem)
        assert example["big_phi"] == sia.phi
        assert len(example["cause_purviews"]) == len(sia.ces)
        for (
            concept,
            small_phi,
            cause_purview,
            cause_repertoire,
            effect_purview,
            effect_repertoire,
        ) in zip(
            sia.ces,
            example["small_phis"],
            example["cause_purviews"],
            example["cause_repertoires"],
            example["effect_purviews"],
            example["effect_repertoires"],
        ):
            assert small_phi == concept.phi
            assert tuple(cause_purview) == concept.cause.purview
            assert np.array_equal(np.array(cause_repertoire), concept.cause.repertoire)
            assert tuple(effect_purview) == concept.effect.purview
            assert np.array_equal(
                np.array(effect_repertoire), concept.effect.repertoire
            )
