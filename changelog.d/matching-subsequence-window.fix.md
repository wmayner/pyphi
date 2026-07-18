`MatchingAnalysis.matching(subsequence_max=True)` now maximizes only over
windows with `a < b`, the maximization domain of the matching definition
(Mayner et al., Eq. 21), and raises a clear `ValueError` for `k < 2`;
single-stimulus windows no longer inflate the estimate.
