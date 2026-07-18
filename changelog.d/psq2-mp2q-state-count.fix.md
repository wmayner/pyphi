`PSQ2` and `MP2Q` now use the number of states (`p.size`) rather than the
first-axis length in their normalization, matching their documented formulas
on multidimensional repertoires, and `MP2Q` no longer returns NaN when the two
distributions share a zero-probability state (those terms contribute zero).
