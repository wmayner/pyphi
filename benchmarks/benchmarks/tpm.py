# -*- coding: utf-8 -*-

"""
Benchmarking for TPM functions.
"""

from pathlib import Path
import numpy as np

import pyphi

ENTROPY = 19283591005102385916723052837195786192730157108218751289951982


class SimulateTPM:
    def setup(self):
        test_data = Path(__file__).resolve().parent / "../../test/data"
        self.tpm = np.load(test_data / "ising_tpm.npy")

        seed_sequence = np.random.SeedSequence(ENTROPY)
        self.rng = np.random.default_rng(seed_sequence)

    def time_simulate(self):
        pyphi.tpm.simulate(self.tpm, 0, int(1e5), self.rng)
