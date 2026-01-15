"""
Type stub file for pyphi.conf module.

This stub file provides type information for the Config descriptor pattern,
which allows type checkers to understand that config.OPTION_NAME returns
the actual value type (e.g., int, str, bool) rather than the Option descriptor.

When adding new config options to conf.py, remember to update this file as well.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from typing import Literal

class Option:
    """Configuration option descriptor."""
    def __init__(
        self,
        default: Any,
        values: list[Any] | None = None,
        type: type | tuple[type, ...] | None = None,
        on_change: Any = None,
        doc: str | None = None,
    ) -> None: ...

class Config:
    """Base configuration class."""
    def __init__(self, on_change: Any = None) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __getitem__(self, name: str) -> Any: ...
    def __eq__(self, other: object) -> bool: ...
    @classmethod
    def options(cls) -> dict[str, Option]: ...
    def defaults(self) -> dict[str, Any]: ...
    def load_dict(self, dct: dict[str, Any]) -> None: ...
    def load_file(self, filename: str) -> None: ...
    def to_yaml(self, filename: str) -> str: ...
    def snapshot(self) -> dict[str, Any]: ...
    def to_dict(self) -> dict[str, Any]: ...
    def override(self, **new_values: Any) -> Any: ...
    def diff(self, other: Config) -> tuple[dict[str, Any], dict[str, Any]]: ...

class PyphiConfig(Config):
    """PyPhi configuration object with all config options as typed attributes."""

    # Numeric configuration
    IIT_VERSION: float
    NUMBER_OF_CORES: int
    MAXIMUM_CACHE_MEMORY_PERCENTAGE: int
    PRECISION: int
    REPR_VERBOSITY: Literal[0, 1, 2]

    # String configuration
    REPERTOIRE_DISTANCE: str
    REPERTOIRE_DISTANCE_INFORMATION: str
    CES_DISTANCE: str
    ACTUAL_CAUSATION_MEASURE: str
    LOG_FILE: str | Path
    LOG_FILE_LEVEL: (
        Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] | None
    )
    LOG_STDOUT_LEVEL: (
        Literal["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"] | None
    )
    LABEL_SEPARATOR: str
    PARTITION_TYPE: str
    SYSTEM_PARTITION_TYPE: str
    DISTINCTION_PHI_NORMALIZATION: Literal["NONE", "NUM_CONNECTIONS_CUT"]
    RELATION_COMPUTATION: Literal["CONCRETE", "ANALYTICAL"]
    STATE_TIE_RESOLUTION: str
    PURVIEW_TIE_RESOLUTION: str
    SYSTEM_CUTS: Literal["3.0_STYLE", "CONCEPT_STYLE"]

    # Boolean configuration
    ASSUME_CUTS_CANNOT_CREATE_NEW_CONCEPTS: bool
    PARALLEL: bool
    CACHE_REPERTOIRES: bool
    CACHE_POTENTIAL_PURVIEWS: bool
    CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA: bool
    REDIS_CACHE: bool
    WELCOME_OFF: bool
    PROGRESS_BARS: bool
    VALIDATE_SUBSYSTEM_STATES: bool
    VALIDATE_CONDITIONAL_INDEPENDENCE: bool
    VALIDATE_JSON_VERSION: bool
    SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI: bool
    PRINT_FRACTIONS: bool
    SYSTEM_PARTITION_INCLUDE_COMPLETE: bool
    SHORTCIRCUIT_SIA: bool

    # Mapping/Dict configuration
    PARALLEL_COMPLEX_EVALUATION: Mapping[str, Any]
    PARALLEL_CUT_EVALUATION: Mapping[str, Any]
    PARALLEL_CONCEPT_EVALUATION: Mapping[str, Any]
    PARALLEL_PURVIEW_EVALUATION: Mapping[str, Any]
    PARALLEL_MECHANISM_PARTITION_EVALUATION: Mapping[str, Any]
    PARALLEL_RELATION_EVALUATION: Mapping[str, Any]
    RAY_CONFIG: dict[str, Any]
    REDIS_CONFIG: dict[str, Any]

    # List configuration
    MIP_TIE_RESOLUTION: list[str]

    def log(self) -> None: ...

# Module-level config instance
config: PyphiConfig

# Helper functions
def fallback(*args: Any) -> Any: ...
def parallel_kwargs(
    option_kwargs: Mapping[str, Any], **user_kwargs: Any
) -> dict[str, Any]: ...
