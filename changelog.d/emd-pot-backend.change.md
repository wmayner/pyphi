Replaced the EMD backend: the now-deprecated `pyemd` is swapped for
[POT](https://pythonot.github.io) (`ot.emd2`), and the `pyphi[emd]` extra now
installs `pot` instead of `pyemd`. The exact EMD cost is the unique
optimal-transport optimum, so the two backends agree to machine epsilon on
PyPhi's Hamming-ground-metric EMD (verified across binary and k-ary inputs);
results are unchanged. As part of this, the IIT 3.0 cause-effect-structure EMD
distance (`ces_measure="EMD"`) was reformulated to a proper non-negative
optimal-transport problem: when a partitioned constellation carries more φ than
the unpartitioned one, the φ deficit is now assigned to each side's null concept
(keeping both signatures non-negative and the distance symmetric) instead of
placing a negative mass on one null. The previous form relied on `pyemd`'s
undefined behavior for signed inputs; the reformulation reproduces the prior
published golden values exactly.
