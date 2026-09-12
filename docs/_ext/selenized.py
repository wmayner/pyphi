"""Pygments styles from the Selenized palettes (Jan Warchoł), light and dark.

Plain names carry the palette's foreground explicitly so a highlighted block
never inherits the page's text colour: a dark block on a light page (or the
reverse) stays readable.
"""

from pygments.style import Style
from pygments.token import Comment
from pygments.token import Keyword
from pygments.token import Name
from pygments.token import Number
from pygments.token import Operator
from pygments.token import Punctuation
from pygments.token import String
from pygments.token import Text


def _styles(fg, dim, green, yellow, blue, magenta, cyan, orange, violet):
    return {
        Text: fg,
        Name: fg,
        Comment: f"italic {dim}",
        Keyword: f"bold {green}",
        Keyword.Constant: yellow,
        Name.Builtin: blue,
        Name.Function: blue,
        Name.Class: f"bold {yellow}",
        Name.Decorator: violet,
        String: cyan,
        String.Interpol: orange,
        Number: magenta,
        Operator: fg,
        Operator.Word: f"bold {green}",
        Punctuation: fg,
    }


class SelenizedLightStyle(Style):
    """Selenized light: cream ground, muted ink."""

    background_color = "#fbf3db"
    styles = _styles(
        fg="#53676d",
        dim="#909995",
        green="#489100",
        yellow="#ad8900",
        blue="#0072d4",
        magenta="#ca4898",
        cyan="#009c8f",
        orange="#c25d1e",
        violet="#8762c6",
    )


class SelenizedDarkStyle(Style):
    """Selenized dark: blue-teal ground."""

    background_color = "#103c48"
    styles = _styles(
        fg="#adbcbc",
        dim="#72898f",
        green="#75b938",
        yellow="#dbb32d",
        blue="#4695f7",
        magenta="#f275be",
        cyan="#41c7b9",
        orange="#ed8649",
        violet="#af88eb",
    )
