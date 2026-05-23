``Substrate`` now stores its TPM as a per-node-factored ``FactoredTPM``
internally.  ``Substrate(tpm=joint_array, ...)`` continues to work via
auto-conversion; the new keyword ``marginals=[factor0, factor1, ...]``
accepts factors directly (mutually exclusive with ``tpm=``), and
``Substrate.from_factored(factored)`` builds a substrate from an
existing :class:`pyphi.core.tpm.factored.FactoredTPM`.  ``substrate.tpm``
returns the ``FactoredTPM``; the joint conditional ndarray is available
on demand via ``substrate.joint_tpm()``.
