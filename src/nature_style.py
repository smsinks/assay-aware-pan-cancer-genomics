"""Shared plotting profile for Nature Communications-sized figures.

The existing project figures were designed on large canvases and embed Type 3 fonts in
PDF output.  This module centralises a journal-width export profile: colourblind-safe
colours, TrueType text in PDF, restrained line weights, and panel labels that remain
legible after reduction to a 180-mm double-column page width.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.font_manager import FontProperties, findfont


MM_TO_IN = 1 / 25.4
SINGLE_COLUMN_MM = 88
DOUBLE_COLUMN_MM = 180

# Okabe-Ito palette, supplemented with neutral greys.  These colours remain separable
# under the common red-green colour-vision deficiencies.
COLORS = {
    "blue": "#0072B2",
    "sky": "#56B4E9",
    "green": "#009E73",
    "orange": "#E69F00",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#222222",
    "grey": "#7A7A7A",
    "light_grey": "#D9D9D9",
    "very_light_grey": "#F2F2F2",
}

# Resolve a real bold face at runtime. This keeps the repository portable while
# preventing PDF and PNG exports from silently substituting a regular face for panel
# letters. Arial is preferred; Matplotlib's bundled bold sans-serif is the fallback.
_PANEL_REQUEST = FontProperties(family="Arial", weight="bold")
PANEL_LABEL_FONT = FontProperties(fname=findfont(_PANEL_REQUEST, fallback_to_default=True))
PANEL_LABEL_FONT.set_weight("bold")


def apply() -> None:
    """Apply the shared publication profile to Matplotlib."""
    mpl.rcParams.update(
        {
            # Exported figures use Helvetica throughout. The font is
            # available as a system TrueType collection in the build environment and
            # remains embedded as Type 42 text in PDF output.
            "font.family": "Helvetica",
            "font.sans-serif": ["Helvetica"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Helvetica",
            "mathtext.it": "Helvetica:italic",
            "mathtext.bf": "Helvetica:bold",
            "mathtext.sf": "Helvetica",
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.5,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            # Preserve the declared physical canvas.  A tight bounding box makes the
            # final PDF width depend on panel contents and therefore silently changes
            # the publication-scale typography from figure to figure.
            "savefig.bbox": None,
            "savefig.pad_inches": 0.02,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def figsize(width_mm: float = DOUBLE_COLUMN_MM, height_mm: float = 150) -> tuple[float, float]:
    """Return a Matplotlib figure size in inches from journal dimensions in mm."""
    return width_mm * MM_TO_IN, height_mm * MM_TO_IN


def panel_label(ax, label: str, *, x: float = -0.12, y: float = 1.05) -> None:
    """Place a bold lower-case Nature-style panel label relative to an axes.

    Lower-casing centrally keeps every figure consistent even when a builder
    supplies a capital literal or generates labels with ``chr(...)``.
    """
    ax.text(
        x,
        y,
        str(label).lower(),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontproperties=PANEL_LABEL_FONT,
        fontweight="bold",
        color=COLORS["black"],
        zorder=1000,
        clip_on=False,
    )


def figure_panel_label(
    fig,
    label: str,
    *,
    x: float,
    y: float,
    ha: str = "left",
    va: str = "top",
) -> None:
    """Place a bold lower-case panel label in figure coordinates.

    This companion is for layouts in which an equal-aspect axes does not occupy its
    full GridSpec cell.  It uses the same typography as :func:`panel_label` and keeps
    figure-coordinate labels subject to the same lower-case contract.
    """
    fig.text(
        x,
        y,
        str(label).lower(),
        ha=ha,
        va=va,
        fontsize=9.5,
        fontproperties=PANEL_LABEL_FONT,
        fontweight="bold",
        color=COLORS["black"],
        zorder=1000,
    )


def aligned_panel_labels(
    fig,
    rows,
    *,
    x_pad_points: float = 14.0,
    y_pad_points: float = 11.0,
) -> None:
    """Place true-bold panel letters at fixed physical offsets and align each row."""
    fig.canvas.draw()
    width_inches, height_inches = fig.get_size_inches()
    x_pad = x_pad_points / (72.0 * width_inches)
    y_pad = y_pad_points / (72.0 * height_inches)
    for row in rows:
        row = tuple(row)
        if not row:
            continue
        label_y = min(0.997, max(ax.get_position().y1 for _, ax in row) + y_pad)
        for letter, ax in row:
            figure_panel_label(
                fig,
                letter,
                x=max(0.004, ax.get_position().x0 - x_pad),
                y=label_y,
            )


def save_figure(fig, stem: Path) -> None:
    """Save vector PDF/SVG and review-ready PNG from one figure object."""
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(stem.with_suffix(".png"), dpi=600)
