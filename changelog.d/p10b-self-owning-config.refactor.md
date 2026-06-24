``pyphi.config`` is now a self-owning layered configuration global. The
three frozen dataclass layers (``FormalismConfig``,
``InfrastructureConfig``, ``NumericsConfig``) are stored directly on
``_GlobalConfig`` and replaced via :func:`dataclasses.replace` on field
writes; there is no longer a wrapped legacy ``PyphiConfig`` instance
behind the layered facade.

Public surface unchanged: both flat (``config.precision``) and layered
(``config.numerics.precision``) reads work; legacy uppercase access
(``config.PRECISION``) is preserved as syntax sugar via case-folding.
``config.override(...)``, ``config.snapshot()``, and
``config.install_snapshot(...)`` keep their semantics.

Wholesale layer replacement (``config.numerics =
NumericsConfig(precision=6)``) is now supported — previously blocked
during the cutover phase.

Validation, logging-callback, and YAML auto-load logic moved from
``pyphi/_conf_legacy.py`` to dedicated modules
(``pyphi/conf/_callbacks.py``, ``pyphi/conf/_helpers.py``); the auto-load
path now uses the layered nested-format YAML, and uppercase top-level
keys raise :class:`ConfigurationError` with a pointer to the rename map.
``pyphi_config.yml`` is migrated from the 1.x flat format to the 2.0
layered format.
