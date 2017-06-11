import numpy as np

from pyphi import utils

cm0 = [
    [1, 1, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 1, 1, 1],
    [1, 0, 0, 1, 1]]

cm1 = [
    [0, 1, 1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 1, 1]]

cm2 = [
    [1, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1]]

cm3 = [[1] * 16] * 16


matrices = [cm0, cm1, cm2, cm3]


class BenchmarkBlockCm():

    params = [0, 1, 2, 3]

    def setup(self, m):
        self.cm = np.array(matrices[m])

    def time_block_cm(self, m):
        utils.block_cm(self.cm)


class BenchmarkReducibility():

    params = [[0, 1, 2, 3], ['block', 'full']]
    param_names = ['cm', 'type']

    def setup(self, m, _type):
        self.cm = np.array(matrices[m])

    def time_reducibility(self, m, _type):
        idxs = (tuple(range(self.cm.shape[0])),
                tuple(range(self.cm.shape[1])))

        if _type == 'block':
            utils.block_reducible(self.cm, *idxs)
        elif _type == 'full':
            utils.fully_connected(self.cm, *idxs)
