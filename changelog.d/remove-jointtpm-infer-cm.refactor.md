Removed the unused ``JointTPM.infer_cm`` and ``JointTPM.infer_edge`` methods
(``pyphi/core/tpm/joint_distribution.py``). Connectivity inference is handled by
``FactoredTPM.infer_cm`` (used by ``pyphi.validate.connectivity``), which works
per factor without materializing the joint and supports non-binary alphabets.
The removed joint-form variant was binary-only, used exact float comparison, and
had no callers outside tests. The two tests that exercised it now assert the
factored path directly. No computed value changes.
