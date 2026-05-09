**Massive speedup (~60-300x on hot paths) by removing dead config-write
infrastructure.** Every ``config.override`` and any direct ``config.X = Y``
assignment was triggering a full YAML serialization of the entire config
to disk via the legacy ``_conf_legacy.on_change_global`` callback. The
callback was added in 2022 to bridge config state to Ray workers spawned
under Python ≥ 3.8 on non-Unix systems (where ``spawn`` replaced ``fork``
and child processes lost parent state). Ray was later removed, and P10's
``ConfigSnapshot`` infrastructure now handles cross-process config
sharing via pickled closures — so the on-change disk write was 3 years
of dead overhead.

Concrete impact: the full golden suite drops from ~13 minutes to ~13
seconds, and ``xor_iit4_2026`` alone goes from 162 s to 2.1 s. Every
code path that mutates config (i.e., essentially the whole compute
pipeline) is faster proportional to how many nested ``config.override``
calls it makes.

Cross-process config sharing is unaffected: the new
:class:`pyphi.conf.snapshot.ConfigSnapshot` mechanism (P10) packages
config state into worker payloads explicitly.
