"""Regenerate the Figure 2 headline panels from committed CSVs.

Outputs (vector PDF, true printed size):
  out/fig_headline_scores.pdf     -> fig:headline-empirical (left)
  out/fig_headline_coherence.pdf  -> fig:headline-empirical (right)

Left panel: mean judge score vs alpha for our role vectors, the per-role
assistant-axis baseline, and the prompted reference. Gated against the
per-alpha means of appendix Tables 12-13 (dedup convention).

Right panel: unique-bigram ratio (coherence proxy) of the raw generation
texts vs alpha, ours vs assistant-axis. Computed fresh from the text
columns; the paper makes only the qualitative claim (ours stable through
alpha=2.5, assistant-axis degrades) which is asserted as ratio ordering.

Texts are processed per-file (275 CSVs, ~1k rows each after dedup) so the
full ~250k x 3 text columns never sit in memory at once.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pvx_fig_style as style

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "experiment_data" / "gold_prompt_experiments"
OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

ALPHAS = [1.0, 1.5, 2.0, 2.5]

# Table 12 / Table 13 per-alpha means (dedup convention) + prompted ref
WANT_STEERED = {1.0: 55.6, 1.5: 59.5, 2.0: 65.9, 2.5: 71.9}
WANT_AA = {1.0: 51.6, 1.5: 53.4, 2.0: 42.1, 2.5: 17.3}
WANT_BASELINE = 89.2


def bigram_ratio(text: str) -> float:
    toks = str(text).lower().split()
    if len(toks) < 2:
        return np.nan
    bigrams = list(zip(toks, toks[1:]))
    return len(set(bigrams)) / len(bigrams)


def accumulate():
    """Per-alpha sums/counts for scores and bigram ratios, streamed per file."""
    cols = ["role", "alpha", "question",
            "steered_score", "assistant_axis_score", "baseline_score",
            "steered", "assistant_axis"]
    sums = {k: {a: 0.0 for a in ALPHAS} for k in
            ("s_score", "a_score", "b_score", "s_coh", "a_coh")}
    counts = {k: {a: 0 for a in ALPHAS} for k in sums}

    def add(key, alpha_groups):
        for a, vals in alpha_groups:
            v = vals.dropna()
            sums[key][a] += float(v.sum())
            counts[key][a] += int(len(v))

    for fp in sorted(DATA.glob("Comparison_GoldStandard_*.csv")):
        df = pd.read_csv(fp, usecols=cols)
        df = df.drop_duplicates(subset=["role", "alpha", "question"],
                                keep="first")
        df = df[df["alpha"].isin(ALPHAS)]
        g = df.groupby("alpha")
        add("s_score", g["steered_score"])
        add("a_score", g["assistant_axis_score"])
        add("b_score", g["baseline_score"])
        for key, col in (("s_coh", "steered"), ("a_coh", "assistant_axis")):
            coh = df[col].map(bigram_ratio)
            add(key, coh.groupby(df["alpha"]))

    means = {k: {a: sums[k][a] / counts[k][a] for a in ALPHAS} for k in sums}
    return means


def verify(name, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {name}: got {got:.3f}, paper {want}")
    if not ok:
        raise SystemExit(f"verification failed: {name}")


def fig_scores(m):
    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.45))
    ref = np.mean([m["b_score"][a] for a in ALPHAS])
    ax.axhline(ref, color=style.GREY, lw=0.9, ls="--")
    ax.text(1.02, ref - 2.0, "prompted reference", fontsize=5.5,
            color=style.GREY, va="top")
    ax.plot(ALPHAS, [m["s_score"][a] for a in ALPHAS], color=style.BLUE,
            marker="o", ms=2.5)
    ax.text(1.0, 66, "role vectors (ours)",
            color=style.BLUE, fontsize=5.5, ha="left", va="bottom")
    ax.plot(ALPHAS, [m["a_score"][a] for a in ALPHAS],
            color=style.VERMILLION, marker="s", ms=2.5)
    ax.text(1.62, m["a_score"][1.5] - 4.0, "assistant axis",
            color=style.VERMILLION, fontsize=5.5, ha="left", va="top")
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("mean judge score")
    ax.set_xticks(ALPHAS)
    ax.set_ylim(0, 100)
    fig.savefig(OUT / "fig_headline_scores.pdf")
    plt.close(fig)


def fig_coherence(m):
    fig, ax = plt.subplots(figsize=(style.HALF_PANEL_W_IN, 1.45))
    ax.plot(ALPHAS, [m["s_coh"][a] for a in ALPHAS], color=style.BLUE,
            marker="o", ms=2.5)
    ax.text(2.42, m["s_coh"][2.5] - 0.035, "role vectors (ours)",
            color=style.BLUE, fontsize=5.5, ha="right", va="top")
    ax.plot(ALPHAS, [m["a_coh"][a] for a in ALPHAS],
            color=style.VERMILLION, marker="s", ms=2.5)
    ax.text(1.0, m["a_coh"][2.0] - 0.04, "assistant axis",
            color=style.VERMILLION, fontsize=5.5, ha="left", va="top")
    ax.set_xlabel(r"steering coefficient $\alpha$")
    ax.set_ylabel("unique-bigram\nratio")
    ax.set_xticks(ALPHAS)
    fig.savefig(OUT / "fig_headline_coherence.pdf")
    plt.close(fig)


def main():
    style.apply()
    m = accumulate()

    print("verification gate (Tables 12-13, dedup convention):")
    for a in ALPHAS:
        verify(f"steered mean @ alpha={a}", m["s_score"][a],
               WANT_STEERED[a], 0.1)
    for a in ALPHAS:
        verify(f"assistant-axis mean @ alpha={a}", m["a_score"][a],
               WANT_AA[a], 0.1)
    b_all = np.mean([m["b_score"][a] for a in ALPHAS])
    verify("prompted-reference grand mean", float(b_all), WANT_BASELINE, 0.1)

    print("coherence (unique-bigram ratio) per alpha:")
    for key, label in (("s_coh", "ours"), ("a_coh", "assistant-axis")):
        vals = [m[key][a] for a in ALPHAS]
        print(f"  {label}: " + ", ".join(f"{a}:{v:.3f}"
                                         for a, v in zip(ALPHAS, vals)))
    # qualitative claim: ours stable through 2.5; assistant-axis degrades
    s_drop = m["s_coh"][1.0] - m["s_coh"][2.5]
    a_drop = m["a_coh"][1.0] - m["a_coh"][2.5]
    print(f"  drop 1.0->2.5: ours {s_drop:.3f}, assistant-axis {a_drop:.3f}")
    if not a_drop > s_drop:
        raise SystemExit("qualitative coherence claim does not hold")

    fig_scores(m)
    fig_coherence(m)
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
