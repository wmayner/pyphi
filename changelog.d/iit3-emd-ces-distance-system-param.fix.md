Restored the IIT 3.0 EMD CES distance code path, which was broken between
the silent rename in ``pyphi.metrics.ces`` (``_emd`` dispatch replaced by
``distribution.EMD.compute``, calling ``pyemd.emd`` with ``Concept`` lists
instead of float arrays) and the removal of the ``Distinction.system`` and
``Distinction.expand_*_repertoire`` back-references. ``emd_concept_distance``,
``_emd_simple``, ``_emd``, and the ``EMD`` CES measure now take an explicit
``system`` argument, threaded through ``ces_distance`` as a keyword.
``ces_distance`` is unchanged for ``SUM_SMALL_PHI``-style measures that
don't need a system context. User-registered custom CES measures should
accept ``system=None`` in their signature.
