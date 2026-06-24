Substrate now stores a per-node-factored TPM (``FactoredTPM``) canonically;
the joint conditional ``P(s_{t+1} | s_t)`` is derived on demand via
``substrate.joint_tpm()``. Existing ``Substrate(tpm=joint_array, ...)``
keeps working — the constructor auto-converts. A new keyword
``marginals=[per_node_factors]`` and factory ``Substrate.from_factored()``
provide direct factored construction. Passing a ``FactoredTPM`` via ``tpm=``
raises ``ValueError``; use ``marginals=`` or ``from_factored()`` instead.
``Substrate(tpm=joint_array, ...)`` continues to work via auto-conversion;
``substrate.tpm`` returns the ``FactoredTPM``; the joint conditional ndarray
is available on demand via ``substrate.joint_tpm()``. The TPM Protocol drops
``squeeze`` (it remains on ``JointTPM`` as a numpy-cleanup affordance, where
it has a coherent meaning). ``Unit`` gains an ``alphabet_size: int = 2``
field; internal math is parameterized by alphabet size.

Internal storage is a swappable backend (default ndarray; xarray
opt-in via ``pip install pyphi[xarray]``). The default backend was
selected by an in-project benchmark; see
``benchmarks/results/factored-tpm-backend-2026-05-22.md``.
