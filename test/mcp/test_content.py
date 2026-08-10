"""The bundled IIT reference documents.

These exercise :mod:`pyphi.mcp.content`, which does not import the optional
``mcp`` dependency, so they run on a base install.
"""

from pyphi.mcp import content


def test_reproducible_work_is_a_topic():
    assert "reproducible-work" in content.topics()


def test_reproducible_work_loads():
    text = content.load("reproducible-work")
    assert text.startswith("# ")
    assert "save_json" in text
    assert "default_rng" in text
