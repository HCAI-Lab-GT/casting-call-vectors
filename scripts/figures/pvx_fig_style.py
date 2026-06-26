"""Shared matplotlib style for camera-ready figures.

Figures are designed at their true printed size for the COLM single-column
layout so that font points in generated PDFs are font points on the page.
Nothing here should be scaled substantially at include time.
"""

from pathlib import Path
import sys

import matplotlib as mpl

# COLM 2026 single-column geometry.
#
# The paper includes most generated PDFs at \linewidth in a single-column
# layout. Generate figures at that true display width so LaTeX does not scale
# 8 pt axis labels into poster-sized text.
TEXT_W_IN = 5.55
COLUMN_W_IN = TEXT_W_IN
HALF_PANEL_W_IN = 0.48 * TEXT_W_IN

PAPER_DIR = Path(__file__).resolve().parents[2] / "paper"
if (PAPER_DIR / "palette_mpl.py").exists():
    sys.path.insert(0, str(PAPER_DIR))

try:
    from palette_mpl import ROLES, apply_rcparams
except ImportError:  # pragma: no cover - fallback for pre-palette checkouts
    ROLES = {
        "role": "#2F6FA3",
        "assistant": "#A84F2A",
        "screen": "#1C7A6A",
        "saturation": "#9A6200",
        "geometry": "#71579B",
        "neutral": "#5F6368",
    }

    def apply_rcparams(rc=None):
        return None


# Semantic colors, shared with paper/palette.toml.
ROLE_STEERING = ROLES["role"]
ASSISTANT_AXIS = ROLES["assistant"]
FIDELITY_SCREEN = ROLES["screen"]
SATURATION = ROLES["saturation"]
GEOMETRY = ROLES["geometry"]
NEUTRAL = ROLES["neutral"]

# Legacy aliases retained so older figure scripts keep working.
BLUE = ROLE_STEERING
VERMILLION = ASSISTANT_AXIS
GREEN = FIDELITY_SCREEN
ORANGE = SATURATION
SKY = "#6BAED6"
PURPLE = GEOMETRY
YELLOW = "#D6B656"
GREY = NEUTRAL

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,          # embed TrueType, ICML-safe
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 7.8,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.45,
    "legend.columnspacing": 0.9,
    "legend.borderaxespad": 0.2,
    "figure.constrained_layout.use": True,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}


def apply():
    mpl.rcParams.update(RC)
    apply_rcparams(mpl.rcParams)
