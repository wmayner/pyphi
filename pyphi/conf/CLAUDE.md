# Configuration options reference

The layered configuration mechanism is documented in the root `CLAUDE.md`
("Configuration System" → "How Configuration Works"). This file is the full
option-by-option reference, loaded only when working with this directory.

## Computational Behavior (`config.formalism.iit`)

- **`version`**: `"IIT_3_0"` / `"IIT_4_0_2023"` / `"IIT_4_0_2026"`
- **`shortcircuit_sia`**: Short-circuit if reducibility detected (default: true)
- **`background_conditioning`**: cause-side background handling —
  `"CAUSAL_MARGINALIZATION"` (IIT 4.0 Eq. 4; default) or
  `"CONDITION_CURRENT_STATE"` (PyPhi 1.x convention; set by
  `presets.iit3`). Only affects proper-subset systems.

## Numerics (`config.numerics`)

- **`precision`**: Numerical precision for phi comparisons (default: 13)

## Performance & Parallelization (`config.infrastructure`)

- **`parallel`**: Global switch for parallelization (default: false)
- **`parallel_workers`**: CPU cores to use (default: -1 = all)
- **`parallel_backend`**: `"local"` (ProcessPoolExecutor) or `"auto"`
- **`parallel_*_evaluation`**: Fine-grained per-level dicts with keys
  `parallel` / `chunksize` / `sequential_threshold` / `progress`
  (e.g. `parallel_concept_evaluation`, `parallel_complex_evaluation`,
  `parallel_partition_evaluation`, `parallel_purview_evaluation`,
  `parallel_mechanism_partition_evaluation`, `parallel_relation_evaluation`)

## Caching (`config.infrastructure`)

- **`cache_repertoires`**: Cache repertoire computations (default: true)
- **`cache_potential_purviews`**: Cache purviews (default: true)
- **`cache_macro_construction`**: Cache mapping-independent macro-construction intermediates (default: true)
- **`clear_system_caches_after_computing_sia`**: Clear after each SIA (default: false)
- **`maximum_cache_memory_percentage`**: Memory limit for in-memory caches (default: 50)

## Measures (`config.formalism.iit`)

- **`mechanism_phi_measure`**: Mechanism-level repertoire-distance measure
  (default: `"GENERALIZED_INTRINSIC_DIFFERENCE"`)
- **`system_phi_measure`**: System-level phi measure
  (default: `"GENERALIZED_INTRINSIC_DIFFERENCE"`; `"INTRINSIC_INFORMATION"`
  enables the Eq. 23 cap in IIT 4.0 2026)
- **`ces_measure`**: Cause-effect-structure distance measure
  (default: `"SUM_SMALL_PHI"`)
- **`config.formalism.actual_causation.alpha_measure`**: AC alpha measure
  (default: `"PMI"`)

## Partitioning (`config.formalism.iit`)

- **`mechanism_partition_scheme`**: Default `"JOINT_PARTITION_ALL"`
- **`system_partition_scheme`**: Default `"DIRECTED_SET_PARTITION"`

## Debugging & Output (`config.infrastructure`)

- **Logging**: PyPhi is silent by default (a `NullHandler` on the `pyphi`
  logger; the root logger is untouched and no log file is written). Opt in with
  `pyphi.enable_logging(level="INFO", file=None)` — console (progress-bar
  safe) when `file` is omitted, or a file path when given. (Not a config
  option; not set through `pyphi_config.yml`.)
- **`progress_bars`**: Show progress bars (default: true)
- **`repr_verbosity`**: Detail level in `repr()` output (default: 2)
- **`welcome_off`**: Suppress welcome message (default: false)
