"""Visual theme for the visualize renderers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Knobs read by the renderers; pass overrides via ``dataclasses.replace``."""

    colorscale: str = "Viridis"
    node_size_range: tuple[float, float] = (10.0, 36.0)
    edge_color: str = "rgba(60, 60, 60, 0.4)"
    edge_width: float = 2.0
    font_family: str = "Helvetica, Arial, sans-serif"
    background: str = "white"
    role_colors: tuple[tuple[str, str], ...] = (
        ("extended", "#e6b422"),
        ("includes", "#2f6fdb"),
        ("included", "#d85a46"),
        ("none", "#b0b0b0"),
    )
    cause_color: str = "#8D3D00"
    effect_color: str = "#006146"
    face_colorscale: str = "Blues"
    face_opacity: float = 0.2
    text_size: int = 12
    spoke_width: float = 1.5
    spoke_curvature: float = 0.0
    spoke_color: str = "rgba(100, 100, 100, 0.45)"
    hub_size_range: tuple[float, float] = (4.0, 22.0)
    relation_rgb: tuple[int, int, int] = (35, 45, 200)
    relation_alpha_range: tuple[float, float] = (0.06, 0.95)


DEFAULT_THEME = Theme()
