# Running expensive analyses: caching and checkpointing

Computing Φ is combinatorially expensive. An exact analysis is practical up to
about **10–12 units**; beyond that a full analysis can exhaust memory and thrash
or hang the machine. The `analyze` tool guards a large substrate behind
`confirm_large` for this reason. When you write PyPhi code in a shell, the
responsibility is yours — plan for cost *before* starting a run, not after it
hangs.

Cost is countable before you commit: the `estimate_cost` tool reports the
workload of an `analyze` call — system partitions, purview evaluations,
mechanism-partition sweeps, specified-state evaluations — without computing
anything (in Python, `pyphi.estimate_analysis`). The `analyze` guard itself
runs on these counts, so `confirm_large` is requested exactly when the workload
is actually large under the active formalism, scheme, and connectivity.

Watch the specified-state axis in particular on a sparse substrate. The other
axes shrink as connectivity thins, because pruning leaves fewer purviews to
evaluate; that one does not. Searching for the system's specified cause and
effect states (Eq. 53) evaluates one forward repertoire per system state per
direction, over an array of that same size, so it grows fourfold per unit added
no matter how the units are wired. It is the axis that dominates a large
sparse system, and the only one whose cost follows from the unit count alone.

Besides the caching and checkpointing covered here, the other lever for
expensive work is running on multiple cores. That has its own topic and its
own pitfalls — in particular, the global `parallel` flag alone does nothing —
so read `get_iit_reference("parallelization")` before reaching for it.

## Caching

PyPhi caches at two levels.

**In-memory (on by default).** Repertoire and potential-purview computations are
memoized within a process, so work is reused within a single analysis. The
combined footprint is bounded by `memory_ceiling_percentage` (default 50%
of the memory the process may use — its cgroup allowance under a scheduler or
container, total RAM otherwise); past that, occupancy holds steady and new
entries displace the least recently used ones. Inspect with `pyphi.cache.info()`, which reports hits,
misses, entry count, bytes, and evictions per cache, and clear with
`pyphi.cache.clear_all()`. Analyzing many systems in a loop and
watching memory climb? Set `clear_system_caches_after_computing_sia = True` to
trade recomputation for a lower ceiling. Under a batch scheduler, a container,
or a cgroup, the allowance is detected from the cgroup; set
`memory_ceiling_bytes` explicitly where none is reported.

Both options bound **total resident memory**, not the caches alone, so size
them from what the process may use rather than from how big you expect the
caches to get. On a sampled 21-unit cause-effect-structure shard the caches
held 70–130 MB while the process held 2.6 GB — the rest being the interpreter,
the substrate TPM, and numpy working space — so the allowance the caches
actually receive is the ceiling less that baseline.

Note also what the ceiling means for a scheduled job: because it follows the
memory actually granted, asking the scheduler for more memory grows the caches
to match rather than leaving the extra as free headroom. To buy headroom
without growing them, raise the request *and* pin
`memory_ceiling_bytes` to the ceiling you want.

**Disk-backed result cache (opt-in, off by default).** This is the one to reach
for on expensive work:

```python
pyphi.config.disk_cache_results = True
```

It persists whole SIA and cause-effect-structure results to a `__pyphi_cache__/`
directory in the working directory, keyed by the substrate, the state, the
relevant configuration, and a code-version component. A repeated analysis of the
same system returns the stored result instead of recomputing — and, crucially,
**a completed result survives a crash, a kill, or a restart**. Two conditions
must hold for a result to be cached:

- The working tree must be clean when running from a git checkout of PyPhi
  (the key uses the commit hash; an installed release uses the package version
  and always caches).
- No result-affecting keyword arguments are passed to `sia()` or `ces()`.

## Surviving a crash

**PyPhi has no built-in checkpointing.** A long computation that hangs or is
killed loses everything in progress. Whenever a run is large enough to
plausibly thrash or hang, build recovery into the script *before* launching it.
The right mechanism depends on the shape of the work — use judgment rather than a
fixed recipe:

- **A sweep you control** — many systems, states, or parameter settings in a
  loop — is the case where saving progress as you go pays off. Persist completed
  results and, on restart, skip inputs already done, so the run resumes instead
  of recomputing. Match the granularity to the work: one file per system is
  natural for a moderate sweep, but if each unit of work is itself large or
  numerous, a single appended file or a periodic flush may be better than one
  file each. Save the inputs (state, and the `seed` if any randomization is
  used) alongside each result so the run is reproducible.
- **A single expensive analysis** — one `sia()` or `ces()` on a large substrate.
  In this case, it may be necessary to store intermediate objects incrementally.
  PyPhi's API exposes every level of the computation, so intermediate results can
  be composed and assembled into a final structure; however, care must be taken to
  do so properly. Make sure you understand the code before attempting this if you
  deem it necessary for performance reasons.

A sketch of the sweep case, to adapt rather than copy verbatim:

```python
import json
import pathlib

import pyphi

pyphi.config.disk_cache_results = True  # completed results survive a crash
out = pathlib.Path("phi_results")
out.mkdir(exist_ok=True)

for label, (substrate, state) in systems.items():
    path = out / f"{label}.json"
    if path.exists():          # done in an earlier run — skip
        continue
    sia = pyphi.System(substrate, state).sia()
    path.write_text(json.dumps({"label": label, "state": list(state), "phi": sia.phi}))
```
