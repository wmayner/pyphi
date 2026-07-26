---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Cache results

Computing $\Phi$ is expensive. PyPhi avoids repeating work at two levels: an
**in-memory cache** that memoizes the intermediate quantities (repertoires,
potential purviews, partition enumerations) reused within a single process, and
an optional **disk-backed result cache** that persists whole SIA and
cause-effect-structure results across processes and sessions.

All of the relevant options are under `pyphi.config.infrastructure`, and can
also be read or written through the top-level `pyphi.config` shortcut.

```{code-cell} python
import pyphi

pyphi.config.progress_bars = False
```

## In-memory caches

By default PyPhi memoizes repertoire and potential-purview computations. These
caches are process-local: each worker in a parallel run keeps its own copy, and
nothing is shared across threads.

```{code-cell} python
pyphi.config.cache_repertoires, pyphi.config.cache_potential_purviews
```

The combined footprint of every in-memory cache is bounded by
`memory_ceiling_percentage` (a percentage of physical RAM). Once the
process crosses that fraction, no new entries are stored, but entries already
cached are still served.

```{code-cell} python
pyphi.config.memory_ceiling_percentage
```

That percentage is measured against the machine's total RAM, which is the
wrong quantity when the process may only use part of the machine — a batch job
with a memory request, a container, or a cgroup. In that case set
`memory_ceiling_bytes` to what the process is actually allowed, and the
caches stop storing at that figure instead. Campaign shards do this for
themselves, using the memory each shard requested.

```{code-cell} python
with pyphi.config.override(memory_ceiling_bytes=2 * 1024**3):
    print(pyphi.config.memory_ceiling_bytes)
```

### Inspecting the caches

`pyphi.cache.info()` returns per-cache hit/miss/size statistics for every
registered cache. After a computation, the repertoire caches show the reuse
they enabled:

```{code-cell} python
substrate = pyphi.examples.iit4_2023_fig1a_substrate()
system = pyphi.System(substrate, (0, 1, 1))

pyphi.cache.clear_all()
sia = system.sia()

pyphi.cache.info()["kernel._cause_repertoire_inner"]
```

The `hits` count is the number of repertoire computations that were served from
the cache instead of recomputed.

### Clearing the caches

Clear every registered cache with `clear_all()`, or a single named cache with
`clear(name)`. The total number of stored entries drops to zero:

```{code-cell} python
pyphi.cache.clear_all()
sum(stats.currsize for stats in pyphi.cache.info().values())
```

Setting `clear_system_caches_after_computing_sia = True` clears the per-system
caches automatically after each SIA computation, trading recomputation for a
lower memory ceiling when analyzing many systems in a loop.

### Turning caching off

Disable a cache when you want to measure uncached performance or bound memory
tightly. Use a temporary override so the change is scoped:

```{code-cell} python
with pyphi.config.override(cache_repertoires=False):
    uncached = system.sia()

round(uncached.phi, 4)
```

## Disk-backed result cache

The result cache persists top-level results (SIA and cause-effect structures)
to disk, so a repeated analysis of the same system returns the stored result
instead of recomputing it. It is off by default:

```{code-cell} python
pyphi.config.disk_cache_results
```

Enable it by setting the flag (typically in your `pyphi_config.yml`):

```{code-cell} python
pyphi.config.disk_cache_results = True
```

Results are written under a `__pyphi_cache__/` directory in the working
directory, one file per key. The cache key incorporates the substrate, the state,
the relevant configuration, and a code-version component, so a change to any of
these produces a different key rather than a stale hit — you never have to clear
the cache manually after changing the theory or the network.

Two conditions must hold for a result to be cached:

- **The working tree must be clean** when running from a git checkout of PyPhi.
  The code-version component of the key is the git commit hash; if the tree has
  uncommitted changes, the key is undefined and the computation runs uncached.
  An installed release (a wheel, not a checkout) uses the package version
  instead and caches normally.
- **No result-affecting keyword arguments** may be passed to `sia()` or `ces()`.
  Calls that pass such arguments bypass the cache, since the key cannot capture
  them.

The disk cache registers under the name `disk.results`, so its statistics
appear alongside the in-memory caches:

```{code-cell} python
pyphi.cache.info()["disk.results"]
```

```{code-cell} python
:tags: [remove-cell]
pyphi.config.disk_cache_results = False
```

## Configuration reference

| Option | Default | Effect |
| --- | --- | --- |
| `cache_repertoires` | `True` | Memoize cause/effect repertoire computations. |
| `cache_potential_purviews` | `True` | Memoize potential-purview enumeration. |
| `memory_ceiling_percentage` | `50` | Upper bound on in-memory cache size, as a percentage of RAM. |
| `memory_ceiling_bytes` | `None` | Upper bound in bytes, for a process confined to less than the whole machine. Replaces the percentage when set. |
| `clear_system_caches_after_computing_sia` | `False` | Clear per-system caches after each SIA. |
| `disk_cache_results` | `False` | Persist SIA and CES results to `__pyphi_cache__/`. |

See {doc}`configure` for how to set these persistently in `pyphi_config.yml`.
