"""Delete generated API stubs whose object no longer exists.

Sphinx's autosummary writes one ``.rst`` stub per documented name into
``docs/reference/_autosummary`` (gitignored) and regenerates the stubs of
names that still exist, but never removes the stub of a renamed or deleted
name. A leftover stub is a page outside every toctree, which fails the
``-W`` docs build. Run before ``sphinx-build``; stubs of names that still
resolve are left untouched, so incremental builds stay incremental.

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


def main() -> int:
    if not STUBS.is_dir():
        return 0
    removed = [
        stub
        for stub in sorted(STUBS.glob("*.rst"))
        if not resolves(SUFFIX.sub("", stub.stem))
    ]
    for stub in removed:
        stub.unlink()
        print(f"removed stale stub {stub.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
