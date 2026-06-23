"""Micro-benchmark: the EMD distance kernel (POT backend)."""

from __future__ import annotations

import numpy as np

from pyphi.measures.distribution import hamming_emd


class Emd:
    params = [4, 8, 16]
    param_names = ("n_states",)

    def setup(self, n_states: int) -> None:
        rng = np.random.default_rng(2026)
        self.p = rng.dirichlet(np.ones(n_states))
        self.q = rng.dirichlet(np.ones(n_states))

    def time_emd(self, n_states: int) -> None:
        hamming_emd(self.p, self.q)
