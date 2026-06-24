**Breaking (2.0):** Remove ``DistanceResult.__array__`` and add
``DistanceResult.values_array(results, dtype=None)`` classmethod.

The implicit ``np.array(results)`` coercion silently dropped per-result
metadata (``method``, ``state``, ``selectivity``, etc.) at unpredictable
points in batch workflows — defeating the class's purpose. The
metadata-loss boundary is now explicit::

    # before (implicit, silent metadata loss)
    arr = np.array(results)

    # after (explicit, intentional metadata loss)
    arr = DistanceResult.values_array(results)

The new classmethod uses ``np.fromiter`` to produce a 1-D float array
without forcing the caller through NumPy's object-dtype fallback. An
explicit ``dtype`` argument is supported.
