
import os

import numpy as np

import pyphi


"""
A set of benchmarks demonstrating the difference between the standard ``pyemd``
computation and Billy's efficient EMD for effect repertoires.

To run these benchmarks::

    asv run develop --steps=1 --bench=emd

"""

def load_repertoire(name):
    """Load an array of repertoires in ./data/emd/."""
    root = os.path.abspath(os.path.dirname(__file__))
    filename = os.path.join(root, 'data', 'emd', name)

    return np.load(filename)


def generate_repertoire():
    """Generate a random 9-node repertoire."""
    return np.random.rand(*([2] * 9))


class BenchmarkEmdRandom:
    """Benchmark EMD on random repertoires."""
    timeout = 100

    def setup(self):
        self.d1 = generate_repertoire()
        self.d2 = generate_repertoire()

    def time_effect_emd(self):
        pyphi.subsystem.effect_emd(self.d1, self.d2)

    def time_hamming_emd(self):
        pyphi.utils.hamming_emd(self.d1, self.d2)


class BenchmarkEmdRule152:
    """Benchmark EMD on data taken from the Rule 152 5-node network.

    The data is the first 100,000 repertoires encountered when computing
    ``big_phi`` for the network.
    """
    timeout = 100

    def setup(self):
        self.d1 = load_repertoire('1.npy')
        self.d2 = load_repertoire('2.npy')

    def time_effect_emd(self):
        for d1, d2 in zip(self.d1, self.d2):
            pyphi.subsystem.effect_emd(d1, d2)

    def time_hamming_emd(self):
        for d1, d2 in zip(self.d1, self.d2):
            pyphi.utils.hamming_emd(d1, d2)
