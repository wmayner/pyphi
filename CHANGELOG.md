Changelog
=========


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
- Added `config.READABLE_REPRS` to control whether `__reprs__` of PyPhi models
  default to using pretty string formatting.
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


0.7.5
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


0.7.4
------------------
_2015-10-12_

### Fixes
- Fixed a caching bug where the subsystem's state was not included in its hash
  value, leading to collisions.


0.7.3
------------------
_2015-09-08_

### API changes
- Heavily refactored the `pyphi.json` module and renamed it to `pyphi.jsonify`.


0.7.2
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
