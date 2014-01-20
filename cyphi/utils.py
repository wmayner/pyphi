import numpy as np
from itertools import chain, combinations
from scipy.misc import comb


# TODO: implement this in cython/numpy despite no gain for small sets?
# http://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.comb.html#scipy.misc.comb


# see http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
def comb_indices(n, k):
    """
    Return the indices that yeild the :math:`k`-combinations of :math:`n`
    elements.

        >>> n, k = 3, 2
        >>> data = np.arange(6).reshape(2, 3)
        >>> print(data[:, comb_indices(n, k)])

    :param a: array to get combinations from
    :type a: ``np.ndarray``
    :param k: length of combinations
    :type k: ``int``

    """
    # Count the number of combinations for preallocation
    count = comb(n, k, exact=True)
    # Get numpy iterable from ``itertools.combinations``
    indices = np.fromiter(chain.from_iterable(combinations(n, k)),
                          int,
                          count=count*k)
    # Reshape output into the array of combination indicies
    return indices.reshape(-1, k)


def powerset(iterable):
    """Return the power set of an iterable

        >>> list(powerset([0,1,2]))


    """

    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
