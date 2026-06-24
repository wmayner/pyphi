Added ``test/test_eq23_cap_oracle.py`` (B4): an independent differential oracle
for the IIT 4.0 (2026) intrinsic-information cap (Eq. 23,
``phi_s = min{phi_c, phi_e, ii(s)}``). It re-derives
``i_diff = -log2 P_forward(state)`` from scratch and confirms it matches the
value production stores on the SIA; verifies the cap-composition identity
``phi_2026 = |min(phi_2023, i_spec_c, i_diff_c, i_spec_e, i_diff_e)|+`` (the cap
terms are partition-independent, so the 2026 MIP reduces to capping the
uncapped 2023 phi); and pins that the cap *strictly binds* with non-zero phi on
the ``logistic3_k8`` network (phi_2023 ≈ 0.037 → phi_2026 ≈ 0.003) while never
increasing phi. Closes the cap's previously un-falsifiable blind spot and
confirms the cap-biting network that N1/P17 need.
