"""Shared plotting style for all empirical analyses."""
from __future__ import annotations
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

# ── Palette (ColorBrewer + Okabe-Ito, colorblind-safe) ───────────────────────
STEERED  = "#2166AC"   # deep blue  — proposed method
AA       = "#B2182B"   # deep red   — assistant axis baseline
BASELINE = "#4D4D4D"   # dark gray  — gold-standard reference lines
WIN      = "#1A9850"   # deep green — win-rate / positive bars
ACCENT   = "#E08214"   # amber      — advantage / residual plots
REPR     = "#762A83"   # purple     — representational geometry

ALPHA_COLORS = {1.0: "#C6DBEF", 1.5: "#6BAED6", 2.0: "#2171B5", 2.5: "#08306B"}

_STYLE = {
    "font.family":          "serif",
    "font.serif":           ["Computer Modern Roman", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset":     "cm",
    "font.size":            10,
    "axes.titlesize":       11,
    "axes.labelsize":       10,
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "legend.fontsize":      9,
    "legend.frameon":       False,
    "legend.borderaxespad": 0.0,
    "figure.dpi":           150,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "axes.grid":            True,
    "grid.alpha":           0.25,
    "grid.linewidth":       0.5,
    "grid.color":           "#cccccc",
    "grid.linestyle":       "-",
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "lines.linewidth":      1.8,
    "lines.markersize":     6,
    "patch.linewidth":      0.6,
    "figure.figsize":       (5.5, 4.0),
    "axes.prop_cycle": matplotlib.cycler("color", ["#2166AC", "#B2182B", "#1A9850", "#E08214", "#762A83"]),
}


def apply_style() -> None:
    """Apply shared LaTeX-like style. Call once at module level."""
    plt.rcParams.update(_STYLE)


def legend_above(ax, ncol: int | None = None, **kwargs) -> None:
    """Place legend centered above axes, no frame."""
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    if ncol is None:
        ncol = len(handles)
    ax.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=ncol,
        frameon=False,
        **kwargs,
    )


def save_fig(fig, out_dir: Path, stem: str) -> None:
    """Save as PDF and PNG, then close."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(out_dir / f"{stem}{ext}")
    plt.close(fig)
    print(f"Saved: {out_dir / stem}.pdf/.png")
