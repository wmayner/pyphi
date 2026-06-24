Fixed expected SIA phi for ``noisy_selfloop_single`` under the
canonical IIT 3.0 preset. The two regression tests
(``test_sia_single_micro_node_selfloops_have_phi`` and
``TestConfigurationDependentValues::test_sia_selfloop_node_phi_with_emd``)
previously expected ``0.6868774943095``, a value that derived from
an intermediate refactor of ``evaluate_partition`` on the IIT 3.0
restoration branch and is not produced by any current code path.
The canonical IIT 3.0 + EMD value is ``0.2736`` — the EMD CES
distance between the one-concept unpartitioned CES (lone concept
on mechanism ``(1,)`` with ``small_phi=0.36``) and the empty
partitioned CES.
