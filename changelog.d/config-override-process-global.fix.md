Documented that `config.override()` applies to the whole process, not the
current thread: while an override is active, every thread reads the
overridden values, so concurrent computations under different configurations
must use separate processes (PyPhi's process-based parallel backends already
give each worker its own configuration copy). Also fixed the one internal
misuse: the macro grain search opened an override inside each parallel
worker, which raced on the shared configuration under the thread backend;
the override is now a single parent-side scope around the dispatch.
