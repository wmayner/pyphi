Internal helpers and display formatting follow the IIT 4.0 vocabulary
introduced earlier: ``pyphi.core.repertoire_algebra.null_concept`` is now
``null_distinction`` (with the old name retained as an alias);
``System.null_concept`` is now ``System.null_distinction`` (the
``null_concept`` property becomes a thin alias);
``pyphi.models.fmt.fmt_concept`` is now ``fmt_distinction`` (with alias).
Distinction docstrings updated to match. The IIT 3.0-specific
``emd_concept_distance`` helper, the private ``_concept_sort_key``, and
the user-facing ``parallel_concept_evaluation`` config option are
intentionally retained — the first two are 3.0-native, and renaming the
config option would be a separate breaking change.
