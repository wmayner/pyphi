"""The demo notebook's stored outputs must match its code and the library."""

import re
import sys
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "docs" / "examples" / "IIT_4.0_demo.ipynb"
sys.path.insert(0, str(ROOT / "scripts"))

from execute_notebook import code_hash  # noqa: E402


def test_stored_outputs_match_the_code():
    """Editing a code cell without re-executing leaves a stale hash."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert nb.metadata["pyphi"]["code_hash"] == code_hash(nb), (
        "the notebook's code changed since its outputs were stored; "
        "run `just notebook-outputs`"
    )


def _text_outputs(cell) -> list[str]:
    out = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream" and o.name == "stdout":
            out.append(o.text)
        elif o.output_type in ("execute_result", "display_data"):
            out.append(o.data.get("text/plain", ""))
    text = "\n".join(out)
    text = re.sub(r"<[^>]+ at 0x[0-9a-f]+>", "<obj>", text)  # object addresses
    timing = re.compile(r"\b\d+(\.\d+)?\s*(s|ms|it/s)\b")
    return [line for line in text.splitlines() if not timing.search(line)]


@pytest.mark.slow
def test_stored_outputs_match_a_fresh_execution():
    """Library changes that alter a printed value must be caught before release."""
    from nbclient import NotebookClient

    stored = nbformat.read(NOTEBOOK, as_version=4)
    fresh = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(
        fresh,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    ).execute()
    for i, (a, b) in enumerate(zip(stored.cells, fresh.cells, strict=True)):
        if a.cell_type != "code" or "skip-execution" in a.metadata.get("tags", []):
            continue
        assert _text_outputs(a) == _text_outputs(b), f"cell {i} output drifted"
