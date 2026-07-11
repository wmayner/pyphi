`CauseEffectStructure.__hash__` now includes the system irreducibility analysis,
matching `__eq__`. Two structures with identical distinctions and relations but
different underlying analyses no longer collide in sets and dictionaries.
