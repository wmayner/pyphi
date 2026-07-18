`k_partitions(collection, k)` returns an empty iterable when `k` exceeds the
collection size, instead of a single invalid pseudo-partition padded with
empty blocks.
