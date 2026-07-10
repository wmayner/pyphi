``intrinsic_differentiation`` now excludes surprisals that are zero up to
``config.numerics.precision`` when selecting the smallest positive surprisal.
A purview node whose probability is 1 up to floating-point noise
(``0.9999999999999998``) previously yielded a surprisal of ~3e-16 that
survived the raw ``> 0`` filter and became a spurious minimum, corrupting
``ii`` and the Eq. 23 capped system φ under
``system_phi_measure = "INTRINSIC_INFORMATION"``.
