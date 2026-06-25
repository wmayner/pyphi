"""One-time migration of the jsonify-format golden fixtures to pyphi.serialize.

For each golden the test suite consumes, this loads the old jsonify-format file
with the legacy serializer, asserts that ``pyphi.serialize`` round-trips the
resulting object to an equal object (the migration *is* the equivalence proof),
archives the old-format file under ``test/data/legacy/``, and writes the
new-format JSON at the original path.

Run once::

    uv run python scripts/migrate_jsonify_goldens.py

It is idempotent only against the original jsonify-format inputs; rerunning
after migration is a no-op-with-error because the originals are gone. The old
files are preserved under ``test/data/legacy/`` for provenance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyphi
from pyphi import jsonify
from pyphi import serialize

DATA = Path("test/data")
LEGACY = DATA / "legacy"

# The jsonify-format fixtures actually loaded by the test suite.
_SIA = [
    "s",
    "s_noised",
    "big_subsys_0_thru_3",
    "rule152_s",
    "big_subsys_all_complete",
    "macro_s",
    "micro_s",
]
_RELATIONS_NETWORKS = ["grid3", "basic", "xor", "rule110", "fig4"]
_PHI_STRUCTURE = [
    "basic",
    "basic_noisy_selfloop",
    "fig4",
    "fig5a",
    "fig5b",
    "grid3",
    "residue",
    "rule110",
    "rule154",
    "xor",
]

CONSUMED = (
    [f"sia/{name}" for name in _SIA]
    + [f"relations/ces_{name}" for name in _RELATIONS_NETWORKS]
    + [f"relations/relations_{name}" for name in _RELATIONS_NETWORKS]
    + [f"phi_structure/{name}" for name in _PHI_STRUCTURE]
)


def migrate_one(rel: str) -> int:
    """Migrate one fixture; return the new-format byte size."""
    path = DATA / f"{rel}.json"
    with open(path) as f:
        obj = jsonify.load(f)

    new_bytes = serialize.dumps(obj, format="json")
    # Equivalence proof: the new format reproduces the trusted object exactly,
    # in both wire formats.
    if serialize.loads(new_bytes, format="json") != obj:
        raise AssertionError(f"{rel}: JSON round-trip is not equal to the original")
    msgpack_bytes = serialize.dumps(obj, format="msgpack")
    if serialize.loads(msgpack_bytes, format="msgpack") != obj:
        raise AssertionError(f"{rel}: msgpack round-trip is not equal to the original")

    archive = LEGACY / f"{rel}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(path.read_bytes())
    path.write_bytes(new_bytes)
    return len(new_bytes)


def main() -> int:
    # Older fixtures were written by earlier PyPhi releases; the trusted
    # reference is the values, not the version pin.
    pyphi.config.validate_json_version = False
    migrated = 0
    for rel in CONSUMED:
        size = migrate_one(rel)
        migrated += 1
        print(f"  migrated {rel:42} -> {size:>8,} B")
    print(f"\nMigrated {migrated} fixtures; old format archived under {LEGACY}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
