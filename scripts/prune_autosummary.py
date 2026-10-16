"""Delete generated API stubs whose object no longer exists.

Sphinx's autosummary writes one ``.rst`` stub per documented name into
``docs/reference/_autosummary`` (gitignored) and regenerates the stubs of
names that still exist, but never removes the stub of a renamed or deleted
name. A leftover stub is a page outside every toctree, which fails the
``-W`` docs build. A module stub whose member tables still list a removed
name is deleted too, so Sphinx regenerates it instead of failing to import
the name. Run before ``sphinx-build``; stubs of names that still resolve
are left untouched, so incremental builds stay incremental.

Usage: uv run python scripts/prune_autosummary.py
"""

import importlib
import re
import sys
from pathlib import Path

STUBS = Path(__file__).resolve().parent.parent / "docs" / "reference" / "_autosummary"

# autosummary_filename_map in docs/conf.py adds a suffix such as "-class" to
# names that would otherwise collide on a case-insensitive file system.
SUFFIX = re.compile(r"-(class|function|attribute)$")


def resolves(dotted: str) -> bool:
    """Whether ``dotted`` names an importable module or an attribute of one."""
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except ImportError:
            continue
        try:
            for name in parts[split:]:
                obj = getattr(obj, name)
        except AttributeError:
            return False
        return True
    return False


def lists_a_missing_member(stub: Path) -> bool:
    """Whether a module stub's autosummary tables list a name that no longer
    resolves. Sphinx reads the stale table before regenerating the stub, so
    the removed name fails to import."""
    lines = stub.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].lstrip("\ufeff").strip():
        return False
    module = lines[0].lstrip("\ufeff").strip()
    if not any(line.strip() == f".. automodule:: {module}" for line in lines):
        return False
    in_table = False
    for line in lines:
        text = line.strip()
        if text.startswith(".. autosummary::"):
            in_table = True
        elif in_table and text and not text.startswith(":"):
            if text.startswith(".. "):
                in_table = False
            elif not resolves(f"{module}.{text}"):
                return True
    return False


def main() -> int:
    if not STUBS.is_dir():
        return 0
    removed = [
        stub
        for stub in sorted(STUBS.glob("*.rst"))
        if not resolves(SUFFIX.sub("", stub.stem)) or lists_a_missing_member(stub)
    ]
    for stub in removed:
        stub.unlink()
        print(f"removed stale stub {stub.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
