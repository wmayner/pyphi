`FrozenMap` now hashes its key–value pairs together rather than hashing the key
set and the value set separately. The previous hash satisfied the equality
contract but did not distinguish mappings that differ only in which key holds
which value, so the 2ⁿ mechanism-state conditions the repertoire cache keys on
collapsed onto three hashes: every cache operation degenerated to a linear scan
of the bucket under `Mapping.__eq__`, making the cache quadratic in its own
size. No computed value changes; the specified-state search over a 10-unit
system drops from 55 seconds to 0.2, and one over 16 units from days to
roughly a minute.
