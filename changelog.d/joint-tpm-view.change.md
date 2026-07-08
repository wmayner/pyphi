`JointTPM` is now a read-only *view* of a substrate's joint conditional TPM —
the joint peer of `FactoredTPM` under the `TPM` protocol, holding the joint in
explicit-alphabet layout for both binary and k-ary substrates.
`Substrate.joint_tpm()` returns a `JointTPM` rather than a bare `numpy.ndarray`;
the view is array-convertible (`numpy.asarray(...)`) and indexable, so existing
array-style usage is unchanged. The former `JointDistribution` base class and
its numpy-proxy machinery are removed.
