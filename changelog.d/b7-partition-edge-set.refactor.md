Unified the partition/cut edge-set interface (B7). Every partition type
(``pyphi/models/partitions.py``) now exposes total ``removed_edges()`` and
``num_connections_cut()`` methods, derived once on ``_PartitionBase`` and
overridden per type with an efficient structural form (no ``n x n``
materialization), validated against ``cut_matrix``. Added a refinement partial
order (``refines()``/``coarsens()`` — superset of severed edges) and
``lex_key``-keyed total-ordering comparison operators, so partitions sort
deterministically. Replaced the ``except AttributeError: return None`` fallback
in distinction-φ normalization with explicit ``None``-partition handling: a
null/unconstrained analysis still normalizes to ``None``, but real
AttributeErrors are no longer swallowed. No computed value changes.
