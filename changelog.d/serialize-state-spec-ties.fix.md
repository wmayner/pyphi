A `StateSpecification` whose tie family is just itself no longer loses that
family on round-trip: `ties` was restored as empty instead of the documented
self-containing tuple whenever there were no tied peers.
