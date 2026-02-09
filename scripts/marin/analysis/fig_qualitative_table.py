#!/usr/bin/env python
"""
Generate a figure showing qualitative steering differences between
residual personality vectors.

Creates a text table figure showing responses to key questions under
different residual steering conditions.
"""

import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIT_COLORS = {
    "artistic":      "#009E73",
    "conventional":  "#56B4E9",
    "investigative": "#0072B2",
    "social":        "#CC79A7",
}


def fig8_qualitative():
    data_path = Path("outputs/qualitative/meta-llama__Llama-3.2-1B-Instruct_qualitative_steering.json")
    if not data_path.exists():
        print("Skipping fig8: no qualitative data")
        return

    with open(data_path) as f:
        data = json.load(f)

    conditions = ["baseline", "residual_artistic", "residual_conventional",
                   "residual_investigative", "residual_social"]
    cond_labels = ["Baseline", "Residual: Artistic", "Residual: Conventional",
                   "Residual: Investigative", "Residual: Social"]
    colors = ["#888888", "#009E73", "#56B4E9", "#0072B2", "#CC79A7"]

    # Use questions 0 and 3 (most distinctive)
    q_indices = [0, 3]

    fig, axes = plt.subplots(len(q_indices), 1, figsize=(7.5, 6.0))
    fig.subplots_adjust(hspace=0.25, top=0.94, bottom=0.02, left=0.02, right=0.98)

    for ax_idx, q_idx in enumerate(q_indices):
        ax = axes[ax_idx]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        question = data["questions"][q_idx]
        ax.text(0.01, 0.98, f"Q: {question}",
                fontsize=8, fontweight="bold", va="top", ha="left",
                transform=ax.transAxes)

        y = 0.88
        for cond, label, color in zip(conditions, cond_labels, colors):
            if cond in data["conditions"]:
                resp = data["conditions"][cond][q_idx]["response"]
                # Truncate to ~120 chars
                resp = resp[:150].replace("\n", " ").strip()
                if len(data["conditions"][cond][q_idx]["response"]) > 150:
                    resp += "..."

                wrapped = textwrap.fill(resp, width=100)

                ax.text(0.01, y, f"{label}:", fontsize=7, fontweight="bold",
                        color=color, va="top", ha="left", transform=ax.transAxes)
                ax.text(0.19, y, wrapped, fontsize=6.5, va="top", ha="left",
                        transform=ax.transAxes, fontstyle="italic",
                        color="#333333", wrap=True)
                y -= 0.19

    fig.suptitle("Residual steering: trait-specific effects in open-ended generation",
                 fontsize=9, fontweight="bold", y=0.98)

    fig.savefig(OUTPUT_DIR / "fig8_qualitative.pdf", dpi=300)
    fig.savefig(OUTPUT_DIR / "fig8_qualitative.png", dpi=300)
    plt.close(fig)
    print("Saved fig8_qualitative")


if __name__ == "__main__":
    fig8_qualitative()
