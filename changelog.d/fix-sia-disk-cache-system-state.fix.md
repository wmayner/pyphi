`System.sia(system_state=...)` shared the plain `sia()` disk-cache entry, so
with `disk_cache_results` enabled a caller-supplied (possibly non-canonical)
state specification could poison — and be served by — the cached canonical
result, persisting across processes. Forced-state calls now bypass the disk
result cache entirely.
