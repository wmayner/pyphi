Fixed the disk result cache (`disk_cache_results`) returning stale results
after a configuration change: the cache key digested only 8 hand-picked
config fields, so changing any other result-affecting option (e.g.
`relation_computation`, tie resolution, `shortcircuit_sia`) silently returned
the previous configuration's cached result. The key now digests every
formalism and numerics field.
