Changelog
=========


_Next version_
--------------

### API additions
- Added `pyphi.tpm.is_deterministic()`

### API changes
- Updated the implementation of `pyphi.convert.state_by_node2state_by_state()`:
  - Can now handle "TPMs" where the number of nodes in the previous and next states differ
  - Improved performance for nondeterministic TPMs


1.2.0
-----
_2019-06-21_

### Fixes

- Fixed a bug introduced into `pyphi.utils.load_data()` by a breaking change
  in recent versions of NumPy that caused an error on import.
- Fixed a bug where changing `config.PRECISION` dynamically did not change
  `constants.EPSILON`, causing some comparisons that relied on
  `constants.EPSILON` to not reflect the new precision.
- Changing `config.FS_CACHE_DIRECTORY` and `config.FS_CACHE_VERBOSITY` now
  causes a new `joblib.Memory` cache to be created. Previously, changing these
  options dynamically had no effect.
- Made test suite compatible with stricter usage of `pytest` fixtures
  required by recent versions of `pytest`.

### API additions

- Added `pyphi.tpm.reconstitute_tpm()`.

### API changes

- Renamed `pyphi.partition.partition_registry` to
  `pyphi.partition.partition_types`.
- Renamed `pyphi.distance.bld()` to `pyphi.distance.klm()`.
- Fixed the connectivity matrix of the `disjunction_conjunction_network()`.
- Removed `'expanded_*_reperotire'` attributes of JSON-serialized `Concept`s.

### Config

- Added the `WELCOME_OFF` option to turn off the new welcome message.

### Documentation

- Added documentation for the `partition_types` registry.
- Added documentation for the filesystem and database caches.


1.1.0
-----
_2018-05-30_

### Fixes

- Fixed a memory leaked when concepts returned by parallel CES computations
  were returned with distinct subsystem objects. Now all objects in a CES share
  the same subsystem reference.
- Fixed a race condition caused by newly introduced `tqdm` synchronization.
  Removed the existing `ProgressBar` implementation and pinned `tqdm` to
  version >= 4.20.0.
- Made model hashes deterministic (6b59061). This fixes an issue with the Redis
  MICE cache in which cached values were not shared between processes and
  program invokations.
- Fixed the connectivity matrix in `examples.disjunction_conjunction.network()`.

### API additions

- Added a `NodeLabels` object for managing the labels of network elements. Most
  models now carry a `NodeLabels` instance that is used for string formatting.
- Added the `cut_node_labels` property to `Subsystem` and `MacroSubsystem`.
- Added `utils.time_annotated` decorator to measure execution speed.

### API changes

- Specifying the nodes of a `Subsystem` is now optional. If not provided, the
  subsystem will cover the entire network.
- Removed the `labels2indices`, `indices2labels` and `parse_node_indices`
  methods from `Network`, and the `indices2labels` method from `Subsystem`.
- Renamed `config.load_config_file` to `config.load_file`, and
  `config.load_config_dict` to `config.load_dict`
- Removed backwards-compatible `Direction` import from `constants` module.
- Renamed `macro.coarse_grain` to `coarse_graining`.
- Exposed `coarse_grain`, `blackbox`, `time_scale`, `network_state` and
  `micro_node_indices` as attributes of `MacroSubsystem`.

### Config

- Removed the `LOG_CONFIG_ON_IMPORT` configuration option.


1.0.0 :tada:
------------
_2017-12-21_

### API changes

#### Modules

- Renamed:
  - `compute.big_phi` to `compute.network`
  - `compute.concept` to `compute.subsystem`
  - `models.big_phi` to `models.subsystem`
  - `models.concept` to `models.mechanism`

#### Functions

- Renamed:
  - `compute.main_complex()` to `compute.major_complex()`
  - `compute.big_mip()` to `compute.sia()`
  - `compute.big_phi()` to `compute.phi()`
  - `compute.constellation()` to `compute.ces()`
  - `compute.conceptual_information()` to `compute.conceptual_info()`
  - `subsystem.core_cause()` to `subsystem.mic()`
  - `subsystem.core_effect()` to `subsystem.mie()`
  - `subsystem.mip_past()` to `subsystem.cause_mip()`
  - `subsystem.phi_mip_past()` to `subsystem.phi_cause_mip()`
  - `subsystem.phi_mip_future()` to `subsystem.phi_effect_mip()`
  - `distance.small_phi_measure()` to `distance.repertoire_distance()`
  - `distance.big_phi_measure()` to `distance.system_repertoire_distance()`
  - For all functions in `convert`:
    - `loli` to `le` (little-endian)
    - `holi` to `be` (big-endian)
- Removed `compute.concept()`; use `Subsystem.concept()` instead.

#### Arguments

- Renamed `connectivity_matrix` keyword argument of `Network()` to `cm`

#### Objects

- Renamed `BigMip` to `SystemIrreducibilityAnalysis`
  - Renamed the `unpartitioned_constellation` attribute to `ces`
  - `sia` is used throughout for attributes, variables, and function names
    instead of `big_mip`
- Renamed `Mip` to `RepertoireIrreducibilityAnalysis`
  - Renamed the `unpartitioned_repertoire` attribute to `repertoire`
  - `ria` is used throughout for attributes, variables, and function names
    instead of `mip`
- Renamed `Constellation` to `CauseEffectStructure`
  - `ces` is used throughout for attributes, variables, and function names
    instead of `constellation`
- Renamed `Mice` to `MaximallyIrreducibleCauseOrEffect`
  - `mic` or `mie` are used throughout for attributes, variables, and function
    names instead of `mip`

- Similar changes were made to the `actual` and `models.actual_causation`
modules.

#### Configuration settings

- Changed configuration settings as necessary to use the new object names.

#### Constants

- Renamed `Direction.PAST` to `Direction.CAUSE`
- Renamed `Direction.FUTURE` to `Direction.EFFECT`

### API additions

#### Configuration settings

- Added `CACHE_REPERTOIRES` to control whether cause/effect repertoires are
  cached. Single-node cause/effect repertoires are always cached.
- Added `CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA` to control whether
  subsystem caches are cleared after calling `compute.sia()`.

#### Objects

- Added two new objects, `MaximallyIrreducibleCause` and
  `MaximallyIrreducibleEffect`, that are subclasses of
  `MaximallyIrreducibleCauseOrEffect` with a fixed direction.

### Refactor

- Moved network-level functions in `compute.big_phi` to
  `pyphi.compute.network`
- Moved subsystem-level functions in `compute.big_phi` and `compute.concept` to
  `compute.subsystem`

### Documentation

- Added a description of TPM representations.
- Improved the explanation of conditional independence and updated the example
  to reflect that PyPhi now raises an error if a conditionally-dependent TPM is
  provided.
- Added detailed installation instructions.
- Little-endian and big-endian replace LOLI and HOLI terminology
- Added documentation for the following modules:
  - `distribution`
  - `cache`
  - `compute.parallel`
  - `compute` top-level module
  - `module` top-level module


0.9.1
-----
_2017-12-21_

### Fixes
- Refactored parallel processing support to fix an intermittent deadlock.


0.9.0
-----
_2017-12-04_

### API changes
- Many functions have been refactored to different modules; see the "Refactor"
  section for details.
- `compute.possible_complexes` no longer includes the empty subsystem.
- Made `is_cut` a property.
- Renamed `macro.list_all_partitions` and `macro.list_all_groupings` to
  `all_partitions` and `all_groupings`. Both are now generators and return
  nested tuples instead of lists.
- Moved `macro.make_mapping` to `CoarseGrain.make_mapping`.
- Moved `macro.make_macro_tpm` to `CoarseGrain.macro_tpm`.
- Added blackbox functionality to `macro.emergence`. Blackboxing and coarse-
  graining are now parametrized with the `blackbox` and `coarse_grain`
  arguments.
- Removed `utils.submatrix`.
- Made `Network.tpm` and `Network.cm` immutable properties.
- Removed the `purview` argument from `Subsystem.expand_repertoire`.
- Moved `validate.StateUnreachableError` and `macro.ConditionallyDependentError`
  to the `exceptions` module.
- Removed perturbation vector support.
- Changed `tpm.marginalize_out` to take a list of indices.
- Fixed `macro.effective_info` to use the algorithm from the macro-micro paper.
- Replace `constants.DIRECTIONS`, `constants.PAST`, and `constants.FUTURE` with
  a proper `Enum` class: `constants.Direction`. Past and future are now
  represented by `constants.Direction.PAST` and `constants.Direction.FUTURE`.
- Simplifed logging config to use `config.LOG_STDOUT_LEVEL`,
  `config.LOG_FILE_LEVEL` and `config.LOG_FILE`.
- Removed the `location` property of `Concept`.

### API additions
- Added `subsystem.evaluate_partition`. This returns the φ for a particular
  partition.
- Added `config.MEASURE` to choose between EMD, KLD, or L1 for distance
  computations.
- Added `macro.MacroSubsystem`. This subclass of `Subsystem` is used to performs
  macro computations.
- Added `macro.CoarseGrain` to represent coarse-grainings of a system.
- Added `macro.Blackbox` to represent system blackboxes.
- Added `validate.blackbox` and `validate.coarse_grain`.
- Added `macro.all_coarse_grains` and `macro.all_blackboxes` generators.
- Added `Subsystem.cut_indices` property.
- Added `Subsystem.cm` connectivity matrix alias.
- Added `utils.all_states`, a generator over all states of an `n`-element
  system.
- Added `tpm.is_state_by_state` for testing whether a TPM is in state-by-state
  format.
- `Network` now takes an optional `node_labels`  argument, allowing nodes to be
  referenced by a canonical name other than their indices. The nodes of a
  `Subsystem` can now be specified by either their index or their label.
- Added `models.normalize_constellation` for deterministically ordering a
  constellation.
- Added a `Makefile`.
- Added an `exceptions` module.
- Added `distribution.purview` for computing the purview of a repertoire.
- Added `distribution.repertoire_shape`.
- Added `config.PARTITION_TYPE` to control the ways in which φ-partitions are
  generated.
- Added more functions to the `convert` module:
  - `holi2loli` and `loli2holi` convert decimal indices between **HOLI** and
    **LOLI** formats.
  - `holi2loli_state_by_state` and `loli2holi_state_by_state` convert between
    **HOLI** and **LOLI** formats for state-by-state TPMs.
  - Added short aliases for some functions:
    - `h2l` is `holi2loli`
    - `l2h` is `loli2holi`
    - `l2s` is `loli_index2state`
    - `h2s` is `holi_index2state`
    - `s2h` is `state2loli_index`
    - `s2l` is `state2holi_index`
    - `h2l_sbs` is `holi2loli_state_by_state`
    - `l2h_sbs` is `loli2holi_state_by_state`
    - `sbn2sbs` is `state_by_node2state_by_state`
    - `sbs2sbn` is `state_by_state2state_by_node`
- Added the `Constellation.mechanisms`, `Constellation.labeled_mechanisms`, and
  `Constellation.phis` properties.
- Add `BigMip.print` method with optional `constellations` argument that allows
  omitting the constellations.

### Refactor
- Refactored the `utils` module into the `connectivity`, `distance`,
  `distribution`, `partition`, `timescale`, and `tpm` modules.
- Existing macro coarse-grain logic to use `MacroSubsystem` and `CoarseGrain`.
- Improved string representations of PyPhi objects.
- Refactored JSON support. The `jsonify` module now dumps PyPhi models to a
  a format which can be loaded to reproduce the full object graph of PyPhi
  objects. This causes backwards incompatible changes to the JSON format of
  some model representations.
- Refactored `pyphi.config` to be an object. Added validation and callbacks for
  config options.

### Optimizations
- Added an analytic solution for the EMD computation between effect
  repertoires.
- Improved the time complexity of `directed_bipartition_of_one` from
  exponential to linear.

### Documentation
- Updated documentation and examples to reflect changes made to the `macro` API
  and usage.
- Added documentation pages for new modules.


0.8.1
------------------
_2016-02-11_

### Fixes
- Fixed a bug in `setup.py` that prevented installation.


0.8.0
------------------
_2016-02-06_

### API changes
- Mechanisms and purviews are now passed to all functions and methods in node
  index form (e.g. `(0, 1, 3)`). Previously, many functions took these
  arguments as `Node` objects. Since nodes belong to a specific `Subsystem` it
  was possible to pass nodes from one subsystem to another subsystem's methods,
  leading to incorrect results.
- `constellation_distance` no longer takes a `subsystem` argument because
  concepts in a constellation already reference their subsystems.
- Moved `utils.cut_mechanism_indices` and `utils.mechanism_split_by_cut` to
  to `Cut.all_cut_mechanisms` and `Cut.splits_mechanism`, respectively;
  moved `utils.cut_mice` to `Mice.damaged_by_cut`.
- `Concept.__eq__`: when comparing concepts for equality, we no longer directly
  check equality of their subsystems. Concept equality is now defined as
  follows:
    - Same φ
    - Same mechanism node indices cause/effect purview node indices
    - Same mechanism state
    - Same cause/effect repertoires
    - Same networks
  This allows two concepts to be equal when _e.g._ the only difference between
  them is that one's subsystem is a superset of the other's subsystem.
- `Concept.__hash__`: the above notion of concept equality is also implemented
  for concept hashing, so two concepts that differ only in that way will have
  the same hash value.
- Disabled concept caching; removed the `config.CACHE_CONCEPTS` option.

### API Additions
- Added `config.REPR_VERBOSITY` to control whether `__reprs__` of PyPhi models
  use pretty string formatting and control the verbosity of the output.
- Added a `Constellation` object.
- Added `utils.submatrix` and `utils.relevant_connections` functions.
- Added the `macro.effective_info` function.
- Added the `utils.state_of` function.
- Added the `Subsystem.proper_state` attribute. This is the state of the
  subsystem's nodes, rather than the entire network state.
- Added an optional Redis-backed cache for Mice objects. This is enabled with
  `config.REDIS_CACHE` and configured with `config.REDIS_CONFIG`.
- Enabled parallel concept evaluation with `config.PARALLEL_CONCEPT_EVALUATION`.

### Fixes
- `Concept.eq_repertoires` no longer fails when the concept has no cause or
  effect.
- Fixed the `Subsystem.proper_state` attribute.

### Refactor
- Subsystem Mice and cause/effect repertoire caches; Network purview caches.
  Cache logic is now handled by decorators and custom cache objects.
- Block reducibility tests and Mice connection computations.
- Rich object comparisons on phi-objects.

### Documentation
- Updated documentation and examples to reflect node-to-index conversion.


0.7.5 [unreleased]
------------------
_2015-11-02_

### API changes
- Subsystem states are now validated rather than network states. Previously,
  network states were validated, but in some cases there can be a
  globally-impossible network state that is locally possible for a subsystem
  (or vice versa) when considering the subsystem's TPM, which is conditioned
  on the external nodes (i.e., background conditions). It is now impossible to
  create a subsystem in an impossible state (a `StateUnreachableError` is
  thrown), and accordingly no 𝚽 values are calculated for such subsystems; this
  may change results from older versions, since in some cases the calculated
  main complex was in fact in an impossible. This functionality is enabled by
  default but can be disabled via the `VALIDATE_SUBSYSTEM_STATES` option.


0.7.4 [unreleased]
------------------
_2015-10-12_

### Fixes
- Fixed a caching bug where the subsystem's state was not included in its hash
  value, leading to collisions.


0.7.3 [unreleased]
------------------
_2015-09-08_

### API changes
- Heavily refactored the `pyphi.json` module and renamed it to `pyphi.jsonify`.


0.7.2 [unreleased]
------------------
_2015-07-01_

### API additions
- Added `convert.nodes2state` function.
- Added `constrained_nodes` keyword argument to `validate.state_reachable`.

### API changes
- Concept equality is now more permissive. For two concepts to be considered
  equal, they must only have the same φ, the same mechanism and purviews (in
  the same state), and the same repertoires.


0.7.1
------------------
_2015-06-30_

### API additions
- Added `purviews`, `past_purviews`, `future_purviews` keyword arguments to
  various concept-calculating methods. With these, the purviews that are
  considered in the concept calculation can be restricted.

### API changes
- States are now associated with subsystems rather than networks. Functions in
  the `compute` module that operate on networks now also take a state.

### Fixes
- Fixed a bug in `compute._constellation_distance_emd` where partitioned
  concepts were unable to be moved to the null concept for the EMD calculation.
  In some cases, the partitioned system has *greater* ∑φ than the unpartitioned
  system; therefore it must be possible for the φ of partitioned-constellation
  concepts to be moved to the null concept, not just vice versa.
- Fixed a bug in `compute._constellation_distance_emd` where it was possible to
  move concepts around within their own constellation; the distance matrix now
  disallows any such intraconstellation paths. This is important because in
  some cases paths from a concept in one constellation to a concept the other
  can actually be shorter if a detour is taken through a different concept in
  the same constellation.
- Fixed a bug in `validate.state_reachable` where network states were
  incorrectly validated.
- `macro.emergence` now always returns a macro-network, even when 𝚽 = 0.
- Fixed a bug in `repr(Network)` where the perturbation vector and connectivity
  matrix were switched.

### Documentation
- Added example describing “magic cuts” that, counterintuitively, can create
  more concepts.
- Updated existing documentation to the new subsystem-state association.


0.7.0
------------------
_2015-05-08_

### API additions
- `pyphi.macro` provides several functions to analyze networks over different
  spatial scales.
- `convert.conditionally_independent(tpm)` checks if a TPM is conditionally
  independent.

### API changes
- Φ and φ values are now rounded to `config.PRECISION` when stored on objects.

### Fixes
- Tests for `Subsystem_find_mip_parallel` and `Subsystem_find_mip_sequential`.
- Slow tests for `compute.big_mip`.

### Refactor
- Subsystem cause and effect repertoire caching.

### Documentation
- Added XOR and Macro examples.


0.6.0
------------------
_2015-04-20_

### Optimizations
- Pre-compute and cache possible purviews.
- Compute concept distance over least-common-purview rather than whole system.
- Store `relevant_connections` on MICE objects for MICE cache checking.
- Only recheck concepts and cut mechanisms after a system cut.

### API additions
- The new configuration option `CUT_ONE_APPROXIMATION` gives an approximation
  of Φ by only considering cuts that cut off a single node.
- Formerly, the configuration was always printed when PyPhi was imported. Now
  this can be suppressed by setting the `LOG_CONFIG_ON_IMPORT` option to
  `false` in the `pyphi_config.yml` file.

### Fixes
- Bipartition function.
- MICE caching.


0.5.0
------------------
_2015-03-02_

### Optimizations
- Concepts are only recomputed if they could have been changed by a cut.
- Cuts are evaluated individually, rather than in bidirectional pairs, which
  allows for better parallel performance.

### API changes
- Removed the unused `validate.nodelist` function.

### API additions
- The new configuration option `ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS` gives
  an approximation of Φ by only recomputing concepts that exist in the
  unpartitioned constellation. This is much faster in certain cases.
- The methods used in determining whether a cut could effect a concept are
  exposed via the `utils` module as:
    - `utils.cut_mechanism_indices`
    - `utils.cut_concepts`
    - `utils.uncut_concepts`
- Added the `pyphi.Subsystem.connections_relevant_for_concept` method.
- Added the `pyphi.Subsystem.cut_matrix` property.

### Fixes
- `pyphi.compute.main_complex` now returns an empty `BigMip` if there are no
  proper complexes.
- No longer using LRU-caches implemented as a circular, doubly-linked list;
  this was causing a huge number of recursive calls when pickling a `Subsystem`
  object (since caches are stored on subsystems since v0.3.6) as `pickle`
  traversed the (potentially very large) cache.
- `pyphi.json.make_encodable` now properly handles NumPy numeric types.


0.4.0
-----
_2015-02-23_

### Optimizations
- `compute.big_mip` is faster for reducible networks when executed in parallel;
  it returns immediately upon finding a reducible cut, rather than evaluating
  all cuts. **NOTE:** This introduces a race condition in cases where there is
  more than one reducible cut; there is no guarantee as to which cut will be
  found first and returned.
- `compute.complexes` prunes out subsystems that contain nodes without either
  inputs or outputs (any subsystem containing such a node must necessarily have
  zero Φ).

### API changes
- `compute.complexes`: returns only irreducible MIPs; see optimizations.
- `compute.big_mip`
  - New race condition with cuts; see optimizations.
  - The single-node and null `BigMip`'s constellations are now empty tuples
    instead of empty lists and `None`, respectively.
- `models.Concept.eq_repertoires` no longer ensures that the networks of each
  concept are equal; it only checks if the repertoires are the same.

### API additions
- `compute.all_complexes` returns the `BigMip` of every subsystem in the
  network's powerset (including reducible ones).
- `compute.possible_main_complexes` returns the subsystems that survived the
  pruning described above.

### Fixes
- Network tests.

### Refactor
- Network attributes. They're now implemented as properties (with getters and
  setters) to facilitate changing them properly. It should be possible to use
  the same network object with different states.
- Network state validation.
- `utils.phi_eq` is used wherever possible instead of direct comparisons to
  `constants.EPSILON`.
