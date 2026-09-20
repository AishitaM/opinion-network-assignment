"""Shared plotting style: one validated palette, one set of chrome rules.

The report is print media, so the figures commit to the light surface only.
Colour jobs used here:
  categorical - topic identity, capped at four validated hues
  sequential  - single blue ramp for magnitude
  diverging   - blue<->red with a grey midpoint for signed opinion (-2..+2)
Community identity (8 groups) is deliberately NOT carried by hue: eight
categorical slots fail the all-pairs colour-blind separation gate, so
communities are faceted and direct-labelled instead.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------- surfaces
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# ----------------------------------------------------------- categorical
# blue / orange / aqua / violet - validated all-pairs, light mode.
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
TOPIC_COLOR = {"T": CAT[0], "E": CAT[1], "S": CAT[2], "V": CAT[3]}

# ------------------------------------------------------------ sequential
BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE)

# ------------------------------------------------------------- diverging
DIV = LinearSegmentedColormap.from_list(
    "div_br", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f3a3a2", "#e34948", "#a32424"]
)

HIGHLIGHT = CAT[0]
NEUTRAL = "#d6d5cf"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 18,
            "axes.labelsize": 8,
            "axes.labelcolor": INK2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.size": 0,
            "ytick.major.size": 0,
            "legend.frameon": False,
            "legend.fontsize": 7.5,
            "lines.linewidth": 2.0,
            "patch.linewidth": 0,
            "text.color": INK,
        }
    )


def title(ax, text: str, sub: str | None = None) -> None:
    """Left-aligned title with an optional secondary-ink subtitle beneath it.

    The subtitle is drawn just above the axes and the title is pushed further
    up by `axes.titlepad`, so the two never overlap.
    """
    ax.set_title(text, color=INK, loc="left", pad=18 if sub else 8)
    if sub:
        ax.text(
            0.0, 1.012, sub, transform=ax.transAxes, fontsize=7.5,
            color=INK2, va="bottom", ha="left",
        )


def bare(ax) -> None:
    """Strip an axes down for node-link diagrams."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
