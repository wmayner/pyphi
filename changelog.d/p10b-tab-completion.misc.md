``pyphi.config.<TAB>`` now lists leaf setting names directly
(``precision``, ``parallel``, ``repertoire_distance``, …) instead of
just the three layer objects. Implemented via ``__dir__`` on the
top-level config facade. Reading still supports both flat
(``config.precision``) and layered (``config.numerics.precision``)
forms; writing remains via the flat form or ``config.override(...)``.

This is the first phase of P10b (finish the config cutover). Later
phases will dissolve the legacy ``_conf_legacy.py`` module entirely
and migrate validators / callbacks / YAML I/O onto the new layered
backend.
