``Substrate`` no longer depends on the legacy binary ``JointTPM`` class.
Joint-array input (2-D state-by-node, state-by-state, and multidimensional
forms) is normalized directly via ``pyphi.convert``, and the binary-only
``Substrate._legacy_binary_joint()`` renderer has been removed in favor of
``Substrate.joint_tpm()``, which is uniform for binary and k-ary substrates.
