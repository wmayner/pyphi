"""The Selenized Pygments styles the docs register."""

import sys
from pathlib import Path

from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer
from pygments.token import Name

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "docs" / "_ext"))

from selenized import SelenizedDarkStyle
from selenized import SelenizedLightStyle


def test_plain_names_have_an_explicit_colour():
    """A block must not depend on the page's text colour for plain names."""
    for style in (SelenizedLightStyle, SelenizedDarkStyle):
        colour = style.styles[Name]
        assert colour.startswith("#")
        html = highlight(
            "pyphi.config", PythonLexer(), HtmlFormatter(style=style, noclasses=True)
        )
        assert colour.lower() in html.lower()
