`from pyphi import *` works on a base install (submodules requiring optional
dependencies are no longer eagerly imported by the star-import), `__all__` has
no duplicates and includes `LandscapeSection`, `Perturbation`, and
`SweepResult`, and `dir(pyphi)` lists all lazily importable submodules.
