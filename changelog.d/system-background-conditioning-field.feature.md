`System` accepts `background_conditioning` (default `None` = follow
`config.formalism.iit.background_conditioning` at compute time) to pin an
instance to one cause-side background convention. Cause factors, nodes, and
kernel cache entries are keyed per convention, so config overrides apply to
already-constructed systems.
