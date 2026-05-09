"""Layered facade over the legacy ``PyphiConfig`` instance.

The :class:`_GlobalConfig` instance exposes the three frozen layers
(``config.formalism``, ``config.infrastructure``, ``config.numerics``) as
live views computed from the current legacy values, so layered reads
always reflect the current state regardless of which access pattern wrote
it. ``__getattr__`` and ``__setattr__`` delegate uppercase-legacy access
to the wrapped :class:`PyphiConfig` so existing code continues to work
during the cutover.

After the cutover (P10 Phase 6), this wrapper is replaced with a
self-owning ``_GlobalConfig`` whose layers are stored directly and the
legacy ``PyphiConfig`` is deleted.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

from pyphi.conf._field_routing import FIELD_TO_LAYER
from pyphi.conf._field_routing import ConfigurationError
from pyphi.conf.formalism import FormalismConfig
from pyphi.conf.infrastructure import InfrastructureConfig
from pyphi.conf.numerics import NumericsConfig
from pyphi.conf.snapshot import ConfigSnapshot

if TYPE_CHECKING:
    from pyphi._conf_legacy import PyphiConfig


class _GlobalConfig:
    """Layered facade wrapping a legacy :class:`PyphiConfig` instance."""

    def __init__(self, legacy: PyphiConfig) -> None:
        object.__setattr__(self, "_legacy", legacy)

    @property
    def formalism(self) -> FormalismConfig:
        legacy = self._legacy
        return FormalismConfig(
            formalism=legacy.FORMALISM,
            assume_cuts_cannot_create_new_concepts=legacy.ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS,
            repertoire_distance=legacy.REPERTOIRE_DISTANCE,
            repertoire_distance_specification=legacy.REPERTOIRE_DISTANCE_SPECIFICATION,
            repertoire_distance_differentiation=legacy.REPERTOIRE_DISTANCE_DIFFERENTIATION,
            ces_distance=legacy.CES_DISTANCE,
            actual_causation_measure=legacy.ACTUAL_CAUSATION_MEASURE,
            partition_type=legacy.PARTITION_TYPE,
            system_partition_type=legacy.SYSTEM_PARTITION_TYPE,
            system_partition_include_complete=legacy.SYSTEM_PARTITION_INCLUDE_COMPLETE,
            system_cuts=legacy.SYSTEM_CUTS,
            distinction_phi_normalization=legacy.DISTINCTION_PHI_NORMALIZATION,
            relation_computation=legacy.RELATION_COMPUTATION,
            state_tie_resolution=legacy.STATE_TIE_RESOLUTION,
            mip_tie_resolution=(
                list(legacy.MIP_TIE_RESOLUTION)
                if isinstance(legacy.MIP_TIE_RESOLUTION, (list, tuple))
                else [legacy.MIP_TIE_RESOLUTION]
            ),
            purview_tie_resolution=legacy.PURVIEW_TIE_RESOLUTION,
            shortcircuit_sia=legacy.SHORTCIRCUIT_SIA,
            single_micro_nodes_with_selfloops_have_phi=legacy.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI,
        )

    @property
    def infrastructure(self) -> InfrastructureConfig:
        legacy = self._legacy
        return InfrastructureConfig(
            parallel=legacy.PARALLEL,
            parallel_complex_evaluation=dict(legacy.PARALLEL_COMPLEX_EVALUATION),
            parallel_cut_evaluation=dict(legacy.PARALLEL_CUT_EVALUATION),
            parallel_concept_evaluation=dict(legacy.PARALLEL_CONCEPT_EVALUATION),
            parallel_purview_evaluation=dict(legacy.PARALLEL_PURVIEW_EVALUATION),
            parallel_mechanism_partition_evaluation=dict(
                legacy.PARALLEL_MECHANISM_PARTITION_EVALUATION
            ),
            parallel_relation_evaluation=dict(legacy.PARALLEL_RELATION_EVALUATION),
            parallel_workers=legacy.PARALLEL_WORKERS,
            parallel_backend=legacy.PARALLEL_BACKEND,
            maximum_cache_memory_percentage=legacy.MAXIMUM_CACHE_MEMORY_PERCENTAGE,
            cache_repertoires=legacy.CACHE_REPERTOIRES,
            cache_potential_purviews=legacy.CACHE_POTENTIAL_PURVIEWS,
            clear_subsystem_caches_after_computing_sia=legacy.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA,
            log_file=legacy.LOG_FILE,
            log_file_level=legacy.LOG_FILE_LEVEL,
            log_stdout_level=legacy.LOG_STDOUT_LEVEL,
            progress_bars=legacy.PROGRESS_BARS,
            repr_verbosity=legacy.REPR_VERBOSITY,
            print_fractions=legacy.PRINT_FRACTIONS,
            label_separator=legacy.LABEL_SEPARATOR,
            welcome_off=legacy.WELCOME_OFF,
            validate_subsystem_states=legacy.VALIDATE_SUBSYSTEM_STATES,
            validate_conditional_independence=legacy.VALIDATE_CONDITIONAL_INDEPENDENCE,
            validate_json_version=legacy.VALIDATE_JSON_VERSION,
        )

    @property
    def numerics(self) -> NumericsConfig:
        return NumericsConfig(precision=self._legacy.PRECISION)

    def snapshot(self) -> ConfigSnapshot:
        return ConfigSnapshot(
            formalism=self.formalism,
            infrastructure=self.infrastructure,
            numerics=self.numerics,
        )

    def override(self, **kwargs: Any) -> Any:
        """Override config options scoped to a context manager or decorator.

        Returns the legacy ``_override`` instance, which is a
        :class:`contextlib.ContextDecorator` — usable as ``with config.override(...):``
        or as a decorator (``@config.override(...)``). Accepts both legacy
        uppercase names and new lowercase layered names; unknown names raise
        :class:`ConfigurationError`.
        """
        legacy_kwargs: dict[str, Any] = {}
        for name, value in kwargs.items():
            if name.isupper():
                legacy_kwargs[name] = value
                continue
            if name in FIELD_TO_LAYER:
                legacy_kwargs[name.upper()] = value
                continue
            raise ConfigurationError(
                f"Unknown config option: {name!r}. "
                "See changelog.d/p10-config-split.refactor.md for the rename map."
            )
        return self._legacy.override(**legacy_kwargs)

    def __getattr__(self, name: str) -> Any:
        # Uppercase legacy access (config.PRECISION, etc.).
        if name.isupper():
            return getattr(self._legacy, name)
        # Anything else (legacy methods like load_dict, snapshot_legacy_dict, etc.)
        # falls through to the wrapped instance.
        legacy = object.__getattribute__(self, "_legacy")
        try:
            return getattr(legacy, name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!r} object has no attribute {name!r}"
            ) from None

    def load_yaml(self, path: str | Path) -> None:
        """Load a 2.0 nested-format YAML config file.

        Each layer's section is applied via ``dataclasses.replace`` on the
        existing layer; only fields present in the file are overridden.
        Raises :class:`ConfigurationError` on unrecognized keys or 1.x
        flat format.
        """
        from pyphi.conf._io import load_yaml as _load

        data = _load(path)
        # Route every nested key through __setattr__ so layered writes go
        # to the legacy backend just like config.x = v does.
        for fields_dict in data.values():
            for field_name, value in fields_dict.items():
                setattr(self, field_name, value)

    def to_yaml(self, path: str | Path) -> None:
        """Write the current config in 2.0 nested-format YAML."""
        from dataclasses import asdict as _asdict

        import yaml as _yaml

        data = {
            "formalism": _asdict(self.formalism),
            "infrastructure": _asdict(self.infrastructure),
            "numerics": _asdict(self.numerics),
        }
        with open(path, "w") as f:
            _yaml.safe_dump(data, f, sort_keys=False)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_legacy":
            object.__setattr__(self, name, value)
            return
        # Uppercase legacy write.
        if name.isupper():
            setattr(self._legacy, name, value)
            return
        # Wholesale layer replacement is intentionally not supported during
        # the cutover — layers are computed views, not stored state. Phase 6
        # introduces a self-owning _GlobalConfig where this works.
        if name in {"formalism", "infrastructure", "numerics"}:
            raise ConfigurationError(
                f"Cannot replace layer {name!r} on the layered facade during "
                "the P10 cutover. Use scoped override(), or set individual "
                "fields via flat lowercase access (config.precision = 6)."
            )
        # Lowercase layered write routes to the legacy uppercase storage.
        if name in FIELD_TO_LAYER:
            setattr(self._legacy, name.upper(), value)
            return
        raise ConfigurationError(
            f"Unknown config option: {name!r}. "
            "See changelog.d/p10-config-split.refactor.md for the rename map."
        )
