# API reference

The modules a user of PyPhi works with, grouped by what they do. Each entry
opens the module's page: its functions, classes, and submodules. Modules
that only serve the package's internals (import deferral, registries,
protocols, type aliases, utilities) are not listed.

## The top-level namespace

`import pyphi` gives the everyday API in one place: `Substrate`, `System`,
`analyze`, `sweep`, `examples`, and `config`.

The [example networks](examples.md) page shows every registered example
rendered from the objects themselves.

```{eval-rst}
.. autosummary::
   :toctree: _autosummary

   pyphi
```

## Substrates and systems

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyphi.substrate
   pyphi.system
   pyphi.substrate_generator
   pyphi.examples
   pyphi.estimate
   pyphi.labels
   pyphi.connectivity
   pyphi.convert
   pyphi.validate
```

## Analysis

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyphi.analyze
   pyphi.sweep
   pyphi.cost
   pyphi.conf
   pyphi.formalism
   pyphi.partition
   pyphi.direction
   pyphi.relations
   pyphi.resolve_ties
   pyphi.condensation
   pyphi.compositional_state
   pyphi.macro
   pyphi.actual
   pyphi.matching
   pyphi.landscape
   pyphi.optimize
   pyphi.dynamics
```

## Results

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyphi.models
   pyphi.display
   pyphi.numerics
   pyphi.measures
   pyphi.serialize
   pyphi.provenance
   pyphi.graph
   pyphi.visualize
   pyphi.exceptions
```

## Infrastructure

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyphi.parallel
   pyphi.cache
   pyphi.campaign
   pyphi.mcp
   pyphi.log
```

## Lower-level building blocks

```{eval-rst}
.. autosummary::
   :toctree: _autosummary
   :recursive:

   pyphi.core
   pyphi.distribution
   pyphi.timescale
   pyphi.automorphism
```
