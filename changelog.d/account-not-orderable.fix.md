`Account` no longer declares itself orderable. It never implemented an ordering, so comparisons raised `NotImplementedError`; they now raise the standard `TypeError` for unorderable types.
