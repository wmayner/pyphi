"""Execute a notebook in place and record a hash of its code cells.

The stored outputs are what the docs render; the hash lets a fast test detect
code edits made without re-executing. Cells tagged ``skip-execution`` are
left untouched.
"""

import hashlib
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def code_hash(nb) -> str:
    """SHA-256 over the sources of the notebook's code cells, in order."""
    h = hashlib.sha256()
    for cell in nb.cells:
        if cell.cell_type == "code":
            h.update(cell.source.encode())
            h.update(b"\0")
    return h.hexdigest()


def main(path: str) -> None:
    file = Path(path)
    nb = nbformat.read(file, as_version=4)
    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(file.parent)}},
    )
    client.execute()
    nb.metadata.setdefault("pyphi", {})["code_hash"] = code_hash(nb)
    nbformat.write(nb, file)
    n = sum(c.cell_type == "code" for c in nb.cells)
    print(f"executed {n} code cells; hash {nb.metadata['pyphi']['code_hash'][:12]}")


if __name__ == "__main__":
    main(sys.argv[1])
