`FrozenMap` now compares itself to another `FrozenMap` by comparing the two
underlying dicts directly. The equality it inherits from `Mapping` first
rebuilds a dict from each operand one key at a time, which every cache lookup
pays for on a hit — 18.4 million times while unfolding the distinctions of a
6-unit system. Comparison against any other kind of mapping still uses the
inherited equality, so comparing a `FrozenMap` to a plain dict behaves as
before. Unfolding the IIT 4.0 Fig 6D distinctions is about 9% faster.
