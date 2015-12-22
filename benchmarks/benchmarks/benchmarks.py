
from pyphi import examples, Subsystem

"""
PyPhi performance benchmarks.

TODO: More representative benchmarks. Is it better to test larger, slower,
runs or smaller systems? Both?
TODO: How does parallelization effect the benchmarks? Should there be
separate benchmarks for parallel execution paths? Caching options?

The `setup` and `setup_cache` functions are called before each repeat of
the benchmark but *not* before each iteration within the repeat. You
must clear all caches *inside* the benchmark function for accurate results.
"""


def clear_subsystem_caches(subsys):
    """Clear subsystem caches"""
    try:
        # New-style caches
        subsys._repertoire_cache.clear()
        subsys._mice_cache.clear()
    except TypeError:
        try:
            # Pre cache.clear() implementation
            subsys._repertoire_cache.cache = {}
            subsys._mice_cache.cache = {}
        except AttributeError:
            # Old school, pre cache refactor
            subsys._repertoire_cache = {}
            subsys._repertoire_cache_info = [0, 0]
            subsys._mice_cache = {}


class BenchmarkSubsystem():

    def setup(self):
        # 7-node network
        self.network = examples.fig16()
        self.state = (0,) * 7
        self.idxs = self.network.node_indices
        self.subsys = Subsystem(self.network, self.state, self.idxs)

    def time_cause_repertoire(self):
        clear_subsystem_caches(self.subsys)
        self.subsys.cause_repertoire(self.idxs, self.idxs)

    def time_cause_repertoire_cache(self):
        clear_subsystem_caches(self.subsys)
        for i in range(3):
            self.subsys.cause_repertoire(self.idxs, self.idxs)

    def time_effect_repertoire(self):
        clear_subsystem_caches(self.subsys)
        self.subsys.effect_repertoire(self.idxs, self.idxs)

    def time_effect_repertoire_cache(self):
        clear_subsystem_caches(self.subsys)
        for i in range(3):
            self.subsys.effect_repertoire(self.idxs, self.idxs)
