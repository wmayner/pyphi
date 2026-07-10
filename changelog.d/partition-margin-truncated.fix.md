`SystemIrreducibilityAnalysis.partition_margin` is now `None` when the
partition sweep stopped early on a reducible partition, instead of reporting
a gap over the truncated prefix that is not the true margin. Set
`shortcircuit_sia=False` for exact margins on reducible systems.
