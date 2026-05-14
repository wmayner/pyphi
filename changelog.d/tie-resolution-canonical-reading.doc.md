Added two specs that pin the canonical reading of IIT 4.0 tie-resolution rules and the corresponding implementation approach:

- ``docs/superpowers/specs/2026-05-13-tie-resolution-canonical-reading.md`` — per-level cascade rules per Albantakis et al. 2023 S1 Text ("Resolving ties in the IIT algorithm"), the IIT actual causation paper (Albantakis et al. 2019), and pyphi-specific design decisions.
- ``docs/superpowers/specs/2026-05-13-cascade-execution-model.md`` — generator-based cascade primitive with ``ResolutionContext`` carrying the entry-point escalation budget and memoization caches; type-level ``UnresolvedSIA`` / ``ResolvedSIA`` split mirroring the equivalent distinction-level split.
