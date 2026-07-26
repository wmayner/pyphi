---
name: pyphi-troubleshooting
description: Use when hitting a specific PyPhi error or surprising behavior — floating-point phi comparisons, pyphi_config.yml not being picked up, missing optional-dependency import errors, TPM format confusion, system state validation failures, cache memory growth, parallel computation not engaging, or mixing IIT 3.0/4.0 code paths.
---

# PyPhi common pitfalls

## 1. Numerical precision issues

**Problem**: Floating-point comparisons fail due to precision.
```python
# Bad
if phi == 0.0:  # May fail due to floating point error

# Good
from pyphi import numerics
if numerics.is_zero(phi):  # Respects config.numerics.precision
```

**Key functions** (in `pyphi.numerics`):
- `numerics.is_zero(x)`
- `numerics.is_positive(x)`
- `numerics.eq(x, y)`

Comparisons are tolerant up to `config.numerics.precision` (default 13
decimal places). Tie resolution clusters candidates equal up to this
tolerance, so equivalent selections co-select regardless of iteration order.

## 2. Configuration not in working directory

**Problem**: `pyphi_config.yml` must be in the directory where Python starts.
```bash
cd /somewhere/else
python -c "import pyphi"  # Uses defaults!

cd /my/project
python -c "import pyphi"  # Loads ./pyphi_config.yml
```

The file uses the **nested** layout with top-level keys `formalism`,
`infrastructure`, and `numerics`. A legacy flat YAML (pre-2.0 UPPER_CASE keys
at the top level) is rejected at load time with a rename map — port it to the
nested keys. At runtime, a top-level write such as
`pyphi.config.precision = 6` routes to the correct layer; the explicit path
`pyphi.config.numerics.precision` also works. Use
`with pyphi.config.override(...)` for temporary scopes.

## 3. Missing optional dependencies

**Problem**: Import errors for optional features. The real extras are:
```bash
uv pip install pyphi[visualize]  # matplotlib, plotly, seaborn, networkx
uv pip install pyphi[caching]    # redis
uv pip install pyphi[emd]        # POT (ot.emd2), for EMD distance measures
uv pip install pyphi[xarray]     # xarray-backed TPM export
uv pip install pyphi[mcp]        # the pyphi-mcp server
```

Parallelism needs **no** extra — it uses the standard-library process pool.
There is no `parallel` or `graphs` extra (networkx ships under `visualize`).

## 4. TPM format confusion

**Problem**: A TPM can be in different formats.

- **State-by-node**: rows are states, columns are nodes (P(node = 1) for
  binary; per-unit distributions for k-ary).
- **State-by-state**: rows are current states, columns are next states.
- **Multidimensional**: high-dimensional array indexed by node states.

**Solution**: Use the conversion utilities in `pyphi.convert`. The core TPM
types live in `pyphi.core.tpm` (there is no top-level `pyphi.tpm`); a
`Substrate` holds a `FactoredTPM`, and `Substrate.joint_tpm()` gives the
read-only joint view.

## 5. System state validation

**Problem**: Building a `System` with an inconsistent state.

`config.infrastructure.validate_system_states` (default on) checks the state
against the substrate. Note the 2.0 rename: the core value types are
`Substrate` (formerly `Network`) and `System` (formerly `Subsystem`).

**Solution**: Use a reachable state, or scope the check off with
`pyphi.config.override(validate_system_states=False)` for special cases.

## 6. Cache memory growth

**Problem**: Large substrates can fill memory with cached repertoires.

**Solution**: The content cache evicts by refcount as substrates are garbage
collected, so releasing references frees entries. To bound or disable it:
```python
pyphi.config.cache_repertoires = False
# or bound the footprint:
pyphi.config.memory_ceiling_percentage = 25
```
`config.infrastructure.disk_cache_results` (default off) additionally persists
`System.sia()`/`ces()` results to `__pyphi_cache__/`.

## 7. Parallel computation not engaging

**Problem**: Set `parallel = True` but computation still runs sequentially.

**Checklist**:
1. Is `config.infrastructure.parallel` on, and the specific operation's flag
   enabled? Each site is a dict: e.g.
   `config.infrastructure.parallel_concept_evaluation["parallel"] = True`
   (siblings: `parallel_purview_evaluation`, `parallel_partition_evaluation`,
   `parallel_relation_evaluation`, `parallel_complex_evaluation`,
   `parallel_mechanism_partition_evaluation`, `parallel_macro_system_evaluation`).
2. Is the problem size above that site's `sequential_threshold`? Below it,
   the dispatch stays sequential by design.
3. Backend is stdlib `ProcessPoolExecutor` by default
   (`config.infrastructure.parallel_backend`); no Ray, no cluster setup.

## 8. IIT version mismatch

**Problem**: Mixing IIT 3.0 and IIT 4.0 code paths or config.

**Solution**: The version is `config.formalism.iit.version` (e.g.
`IIT_4_0_2023`, `IIT_4_0_2026`, `IIT_3_0`); presets live in
`pyphi.conf.presets`. The entry points are formalism-agnostic — there is no
`pyphi.new_big_phi` or `pyphi.compute` in 2.0:
- `System(substrate, state).sia()` — system irreducibility (Φ).
- `System(substrate, state).ces()` — the cause-effect structure.
- `pyphi.analyze(substrate, state, formalism=…)` — one high-level entry point.

The 2026 formalism selects the MIP exactly as 2023 does, then applies the
Eq. 23 ii-cap, so `IIT_4_0_2026` Φ is never above `IIT_4_0_2023` Φ.
