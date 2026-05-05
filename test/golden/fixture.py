"""Fixture loading, storing, and comparison.

Each fixture lives as a pair of files:

- ``test/data/golden/v1/<name>.json`` — structured data (config, network hashes,
  partitions, mechanisms, scalar values, array references)
- ``test/data/golden/v1/<name>.npz`` — array data (repertoires, partitioned
  repertoires) keyed by the references in the JSON

Arrays are referenced from the JSON via the ``"@npz:<key>"`` string sentinel.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np

from pyphi import config

GOLDEN_DATA_DIR = Path(__file__).parent.parent / "data" / "golden" / "v1"
SCHEMA_VERSION = 1


@dataclass
class GoldenFixture:
    """Specification of a single golden regression fixture.

    A fixture is the composition of (a) a way to construct the test inputs
    (network + state + node indices), (b) a config context, and (c) a captured
    set of expected output values to assert against.
    """

    name: str
    """Unique fixture name; becomes the file stem."""

    config_overrides: dict[str, Any]
    """Config settings applied during compute (overlaid on defaults)."""

    network_factory: Callable[[], Any]
    """Zero-arg callable returning a ``pyphi.Network``."""

    state: tuple[int, ...]
    """Network state to evaluate."""

    node_indices: tuple[int, ...] | None = None
    """Subsystem node indices (None = full network)."""

    description: str = ""
    """Human-readable description of what this fixture covers."""

    skip_layers: frozenset[str] = field(default_factory=frozenset)
    """Layers to skip in compute (e.g., ``{"phi_structure"}`` for IIT 3.0)."""

    # ============== Compute ==============

    def config_context(self) -> AbstractContextManager:
        """Return a context manager applying ``config_overrides``."""
        return config.override(**self.config_overrides)

    def build_network(self) -> Any:
        """Build the network."""
        return self.network_factory()

    # ============== Storage paths ==============

    @property
    def json_path(self) -> Path:
        return GOLDEN_DATA_DIR / f"{self.name}.json"

    @property
    def npz_path(self) -> Path:
        return GOLDEN_DATA_DIR / f"{self.name}.npz"

    @property
    def is_stored(self) -> bool:
        return self.json_path.exists() and self.npz_path.exists()


def store_fixture(
    fixture: GoldenFixture,
    structured: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    generated_from_commit: str = "",
) -> None:
    """Persist computed values to disk.

    Args:
        fixture: The fixture being captured.
        structured: JSON-serializable data (with ``"@npz:<key>"`` references for arrays).
        arrays: Mapping of npz keys to numpy arrays.
        generated_from_commit: Optional git commit hash for provenance.
    """
    GOLDEN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    header = {
        "fixture_name": fixture.name,
        "schema_version": SCHEMA_VERSION,
        "generated_from_commit": generated_from_commit,
        "description": fixture.description,
        "config_overrides": fixture.config_overrides,
        "state": list(fixture.state),
        "node_indices": list(fixture.node_indices) if fixture.node_indices else None,
    }

    payload = {**header, **structured}

    with fixture.json_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=False, default=_json_default)

    np.savez_compressed(fixture.npz_path, **arrays)


def load_fixture(fixture: GoldenFixture) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load expected values from disk.

    Returns ``(structured, arrays)`` where ``structured`` is the JSON content
    and ``arrays`` is the npz content as a dict.
    """
    with fixture.json_path.open() as f:
        structured = json.load(f)

    npz = np.load(fixture.npz_path)
    arrays = {key: npz[key] for key in npz.files}

    return structured, arrays


def _json_default(obj: Any) -> Any:
    """JSON encoder for numpy types and tuples."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, frozenset):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def network_hash(tpm: np.ndarray, cm: np.ndarray) -> dict[str, str]:
    """Stable hash of a network's TPM and connectivity matrix.

    Used as a sanity check during fixture loading: if a fixture's network
    changes shape or content, the hash diverges and the test fails loudly
    rather than producing mysteriously wrong values.
    """
    tpm_bytes = np.ascontiguousarray(tpm).tobytes()
    cm_bytes = np.ascontiguousarray(cm).tobytes()
    return {
        "tpm_shape": list(tpm.shape),
        "tpm_sha1": hashlib.sha1(tpm_bytes).hexdigest()[:16],
        "cm_shape": list(cm.shape),
        "cm_sha1": hashlib.sha1(cm_bytes).hexdigest()[:16],
    }


def array_ref(key: str) -> str:
    """Sentinel marking an array reference in structured JSON.

    The JSON contains ``"@npz:r0"`` where the actual array lives at key ``"r0"``
    in the sibling .npz file.
    """
    return f"@npz:{key}"


def is_array_ref(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("@npz:")


def deref_array_ref(value: str) -> str:
    """Extract the npz key from an array reference sentinel."""
    assert is_array_ref(value)
    return value[len("@npz:") :]
