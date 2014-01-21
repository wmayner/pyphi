import numpy as np
from itertools import chain, combinations
from scipy.misc import comb


# see http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
def combs(a, r):
    """
    Numpy implementation of itertools.combinations.

    Return successive :math:`r`-length combinations of elements in the array `a`.

    :param a: the array from which to get combinations
    :type a: ``np.array``
    :param r:  the length of the combinations
    :type r: ``int``

    :returns: an array of combinations
    :rtype: ``np.array``

    """
    # Special-case for 0-length combinations
    if r is 0:
        return np.asarray([])

    a = np.asarray(a)
    data_type = a.dtype if r is 0 else np.dtype([('', a.dtype)]*r)
    b = np.fromiter(combinations(a, r), data_type)
    return b.view(a.dtype).reshape(-1, r)


# see http://stackoverflow.com/questions/16003217/n-d-version-of-itertools-combinations-in-numpy
def comb_indices(n, k):
    """
    N-d version of itertools.combinations.

    Return indices that yeild the :math:`r`-combinations of :math:`n` elements

        >>> n, k = 3, 2
        >>> data = np.arange(6).reshape(2, 3)
        >>> print(data)
        [[0 1 2]
         [3 4 5]]
        >>> print(data[:, comb_indices(n, k)])
        [[[0 1]
          [0 2]
          [1 2]]
         [[3 4]
          [3 5]
          [4 5]]]

    :param a: array from which to get combinations
    :type a: ``np.ndarray``
    :param k: length of combinations
    :type k: ``int``

    :returns: indices of the :math:`r`-combinations of :math:`n` elements
    :rtype: ``np.array``

    """
    # Count the number of combinations for preallocation
    count = comb(n, k, exact=True)
    # Get numpy iterable from ``itertools.combinations``
    indices = np.fromiter(
        chain.from_iterable(combinations(range(n), k)),
        int,
        count=count*k)
    # Reshape output into the array of combination indicies
    return indices.reshape(-1, k)


# TODO: implement this with numpy?
def powerset(iterable):
    """
    Return the power set of an iterable (see `itertools recipes
    <http://docs.python.org/2/library/itertools.html#recipes>`_).

        >>> ps = powerset(np.arange[2])
        >>> print(list(ps))
        [(), (0,), (1,), (0, 1)]

    :param iterable: an iterable from which to generate the power set

    :returns: an iterator over the power set
    :rtype: iterator

    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class ValidationException(Exception):
    """
    To be thrown when a user-provided value is incorrect.
    """
    pass
