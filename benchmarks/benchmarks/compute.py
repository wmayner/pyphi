
from pyphi import compute, constants, examples, Subsystem
from .subsystem import clear_subsystem_caches


def _clear_joblib_cache():
    constants.joblib_memory.clear()


class BenchmarkConstellation:

    params = ['basic', 'rule154', 'fig16']

    def setup(self, network):
        if network == 'basic':
            self.subsys = examples.basic_subsystem()

        elif network == 'rule154':
            network = examples.rule154_network()
            state = (1,) * 5
            self.subsys = Subsystem(network, state, network.node_indices)

        elif network == 'fig16':
            network = examples.fig16()
            state = (0,) * 7
            self.subsys = Subsystem(network, state, network.node_indices)

        else:
            raise

    def time_constellation(self, network):
        clear_subsystem_caches(self.subsys)
        # network purview caches are left intact
        compute.constellation(self.subsys)
