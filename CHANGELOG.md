Changelog
=========


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
