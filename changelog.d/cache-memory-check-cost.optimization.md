The cache memory check no longer builds a new `psutil.Process` on every call.
It runs on every cache miss, and constructing the handle cost about ten times as
much as reading resident memory from an existing one: 14.5 µs per call against
1.4 µs. On a scoped cause-effect structure sweep making 1.3 million misses, that
was 19 seconds of a 208-second run, and the share grows the more the cache
misses. The handle is now reused, and rebuilt after a fork.
