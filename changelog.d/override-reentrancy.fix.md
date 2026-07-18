`pyphi.config.override(...)` contexts are now safely reentrant: using one
context object as a decorator on a recursive function (or nesting it) no
longer permanently leaks the override into the global config.
