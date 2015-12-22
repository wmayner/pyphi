
from pyphi import examples, config, Subsystem

"""
PyPhi performance benchmarks

TODO: More representative benchmarks. Is it better to test larger, slower,
runs or smaller systems? Both?
TODO: How does parallelization effect the benchmarks? Should there be
separate benchmarks for parallel execution paths? Caching options?

The `setup` and `setup_cache` functions are called before each repeat of
the benchmark but *not* before each iteration within the repeat. You
must clear all caches *inside* the benchmark function for accurate results.

Can't use `@config.override` because it doesn't exist in the entire
project history and will break on import of older revisions.
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


def clear_network_caches(network):
    try:
        network.purview_cache.clear()
    except TypeError:
        try:
            network.purview_cache.cache = {}
        except AttributeError:
            network.purview_cache = {}


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

    # Potential purviews benchmark.
    # TODO: this isn't representative of what actually happens.
    # Can we capture a sample run of multiple calls to
    # subsys._potential_purviews?

    def _do_potential_purviews(self):
        for i in range(100):
            self.subsys._potential_purviews('past', self.idxs)

    def time_potential_purviews_no_cache(self):
        # Network purview caches disabled
        clear_network_caches(self.subsys.network)
        default = config.CACHE_POTENTIAL_PURVIEWS
        config.CACHE_POTENTIAL_PURVIEWS = False
        self._do_potential_purviews()
        config.CACHE_POTENTIAL_PURVIEWS = default

    def time_potential_purviews_with_cache(self):
        # Network purview caches enabled
        clear_network_caches(self.subsys.network)
        default = config.CACHE_POTENTIAL_PURVIEWS
        config.CACHE_POTENTIAL_PURVIEWS = True
        self._do_potential_purviews()
        config.CACHE_POTENTIAL_PURVIEWS = default
