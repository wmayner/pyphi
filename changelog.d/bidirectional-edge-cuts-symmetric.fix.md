Fixed `EDGE_CUT_BIDIRECTIONAL` generating asymmetric cut matrices and silently
omitting half of the valid bidirectional cuts for systems of 4 or more units
(`numpy.triu_indices`/`tril_indices` enumerate mirror-image cells at the same
flat position only up to n = 3). φ_s minimized under this partition scheme on
4+ units previously searched the wrong partition family. The default scheme is
unaffected.
