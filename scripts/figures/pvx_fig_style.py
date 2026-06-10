"""Shared matplotlib style for camera-ready figures.

Figures are designed at their TRUE printed size (ICML two-column:
\\columnwidth = 234.8775pt = 3.25in) so that font points in the PDF are
font points on the page. Nothing here may be scaled down at include time
by more than ~10% without violating the >=7pt effective-font rule.
"""

import matplotlib as mpl

# ICML 2026 geometry
COLUMN_W_IN = 3.25          # \columnwidth
TEXT_W_IN = 6.75            # \textwidth (full two-column span)
HALF_PANEL_W_IN = 1.56      # 0.48\columnwidth minipage

# Okabe-Ito (colorblind-safe)
BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
ORANGE = "#E69F00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
YELLOW = "#F0E442"
GREY = "#999999"

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,          # embed TrueType, ICML-safe
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
}


def apply():
    mpl.rcParams.update(RC)
