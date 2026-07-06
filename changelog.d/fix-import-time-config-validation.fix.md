An invalid `pyphi_config.yml` (e.g. a measure incompatible with the
configured IIT version) now raises `ConfigurationError` at `import pyphi`
instead of importing cleanly and failing at compute time. Set
`infrastructure.validate_config: false` to opt out.
