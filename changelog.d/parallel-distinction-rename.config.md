Renamed the `parallel_concept_evaluation` configuration option to
`parallel_distinction_evaluation`. The option governs the loop that computes
distinctions (one task per mechanism) under every formalism, so it now uses
the current name for those objects; "concept" survives only in IIT
3.0-specific surfaces, where it is the term used in the paper. There is no
compatibility alias — update `pyphi_config.yml` files and `override(...)`
calls that use the old name.
