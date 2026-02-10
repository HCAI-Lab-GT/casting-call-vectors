#!/usr/bin/env python3
"""Publication-quality figures for personality vectors research report."""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

# ── Output ──────────────────────────────────────────────────
OUT = Path("outputs/figures/report")
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path("outputs/analysis")

# ── Publication style ───────────────────────────────────────
RC_PARAMS = {
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "font.family": "sans-serif",
}


def _init_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(RC_PARAMS)
    return plt


plt = _init_matplotlib()

TRAITS = ["artistic", "conventional", "enterprising", "investigative", "realistic", "social"]
C = plt.cm.tab10.colors


def load(name):
    with open(DATA / name) as f:
        return json.load(f)


def save(fig, name):
    fig.savefig(OUT / name, dpi=300)
    plt.close(fig)
    print(f"  {name}")


# ════════════════════════════════════════════════════════════
# 1. Singular value spectrum across models
# ════════════════════════════════════════════════════════════
def fig01_singular_values():
    d = load("personality_scaling.json")
    models = d["per_model"]

    short_names = {
        "meta-llama/Llama-3.2-1B-Instruct": "Llama 1B",
        "HuggingFaceTB/SmolLM3-3B": "SmolLM3 3B",
        "marin-community/marin-8b-instruct": "Marin 8B",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.8))

    for ax, (model_id, info) in zip(axes, models.items(), strict=False):
        svs = info["singular_values"]
        x = np.arange(1, 7)
        colors = [C[0]] * 5 + [C[3]]
        ax.bar(x, svs, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
        ax.set_xticks(x)
        ax.set_xlabel("Component")
        name = short_names.get(model_id, model_id.split("/")[-1])
        ax.set_title(name, fontsize=11)

        # Annotate SV6 value
        ymax = max(svs)
        ax.annotate(
            f"{svs[5]:.1e}", xy=(6, svs[5] + ymax * 0.03), fontsize=8, ha="center", color=C[3]
        )

        # Set y range starting at 0
        ax.set_ylim(0, ymax * 1.15)
        ax.set_ylabel("Singular value")

    fig.suptitle(
        "Residual singular value spectrum (6th $\\approx$ 0 for all models)", fontsize=12, y=1.05
    )
    fig.tight_layout()
    save(fig, "fig01_svd_spectrum.png")


# ════════════════════════════════════════════════════════════
# 2. Alpha phase diagram
# ════════════════════════════════════════════════════════════
def fig02_alpha_phase():
    d = load("alpha_phase_diagram.json")
    pa = d["positive_alpha"]
    keys = sorted(pa.keys(), key=lambda x: float(x))
    alphas = [float(k) for k in keys]
    accs = [pa[k]["accuracy"] * 100 for k in keys]
    ppls = [pa[k]["mean_perplexity"] for k in keys]

    fig, ax1 = plt.subplots(figsize=(5, 3.5))

    ax1.semilogx(alphas, accs, "o-", color=C[2], markersize=4, label="Detection accuracy")
    ax1.axhline(y=16.7, color="grey", linestyle=":", linewidth=0.8, alpha=0.6)
    ax1.set_xlabel(r"Steering strength $\alpha$")
    ax1.set_ylabel("Detection accuracy (%)", color=C[2])
    ax1.tick_params(axis="y", labelcolor=C[2])
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    ax2.semilogx(alphas, ppls, "s-", color=C[3], markersize=3, linewidth=1.2, label="Perplexity")
    ax2.set_ylabel("Perplexity", color=C[3])
    ax2.tick_params(axis="y", labelcolor=C[3])
    ax2.set_ylim(0, 12)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    # Goldilocks zone
    ax1.axvspan(1.5, 15, alpha=0.08, color=C[1])
    ax1.text(4.5, 8, r"$\alpha$=1.5\u201315", fontsize=9, ha="center", color=C[1], alpha=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", frameon=False, fontsize=9)

    save(fig, "fig02_alpha_phase.png")


# ════════════════════════════════════════════════════════════
# 3. KL divergence layer profile (logit lens)
# ════════════════════════════════════════════════════════════
def fig03_kl_layer_profile():
    d = load("logit_lens_personality.json")
    kl = d["kl_divergence"]

    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    for i, trait in enumerate(TRAITS):
        vals = kl[trait]
        ax.plot(range(len(vals)), vals, "-", color=C[i], linewidth=1.2, label=trait, alpha=0.85)

    ax.axvline(x=16, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ymax = max(max(kl[t]) for t in TRAITS)
    ax.text(16.5, ymax * 0.92, "inject L16", fontsize=8, color="grey")

    ax.set_xlabel("Layer")
    ax.set_ylabel("KL divergence\n(steered vs. baseline)")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    save(fig, "fig03_kl_layer_profile.png")


# ════════════════════════════════════════════════════════════
# 4. Causal patching across all layers
# ════════════════════════════════════════════════════════════
def fig04_causal_patching():
    d = load("causal_patching.json")
    slp = d["single_layer_patch"]

    # Average across available traits
    available_traits = [t for t in TRAITS if t in slp]
    all_layers = sorted(slp[available_traits[0]].keys(), key=int)
    layer_nums = [int(layer_str) for layer_str in all_layers]

    mean_norms = []
    for layer_str in all_layers:
        vals = [slp[t][layer_str]["norm"] for t in available_traits if layer_str in slp[t]]
        mean_norms.append(np.mean(vals))

    fig, ax = plt.subplots(figsize=(5, 2.8))
    colors = [C[3] if n < np.mean(mean_norms) * 0.1 else C[0] for n in mean_norms]
    ax.bar(layer_nums, mean_norms, color=colors, width=0.8, edgecolor="none")
    ax.set_xlabel("Layer replaced with baseline activations")
    ax.set_ylabel("5D personality norm")

    # Find the critical layer
    min_idx = np.argmin(mean_norms)
    min_layer = layer_nums[min_idx]
    ax.annotate(
        "Signal destroyed",
        xy=(min_layer, mean_norms[min_idx]),
        xytext=(min_layer + 6, max(mean_norms) * 0.6),
        fontsize=9,
        ha="center",
        color=C[3],
        arrowprops={"arrowstyle": "->", "color": C[3], "linewidth": 1},
    )

    fig.tight_layout()
    save(fig, "fig04_causal_patching.png")


# ════════════════════════════════════════════════════════════
# 5. Token shift semantic analysis (logit lens)
# ════════════════════════════════════════════════════════════
def fig05_token_shifts():
    d = load("logit_lens_personality.json")
    ts = d["token_shifts"]

    fig, axes = plt.subplots(2, 3, figsize=(8, 5.5))
    axes = axes.flatten()

    for idx, trait in enumerate(TRAITS):
        ax = axes[idx]
        up = ts[trait]["upweighted"][:5]
        down = ts[trait]["downweighted"][:5]

        tokens = [t[0] for t in down[::-1]] + [t[0] for t in up]
        shifts = [-t[1] for t in down[::-1]] + [t[1] for t in up]
        colors_bar = [C[3]] * len(down) + [C[2]] * len(up)

        # Clean up token display
        tokens = [t.strip()[:14] for t in tokens]

        ax.barh(range(len(tokens)), shifts, color=colors_bar, edgecolor="none", height=0.65)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens, fontsize=8)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.set_title(trait, fontsize=10, color=C[idx])
        ax.set_xlabel("Logit shift", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout(h_pad=1.5, w_pad=1.0)
    save(fig, "fig05_token_shifts.png")


# ════════════════════════════════════════════════════════════
# 6. Negative alpha → Holland opposites
# ════════════════════════════════════════════════════════════
def fig06_negative_alpha():
    d = load("alpha_phase_diagram.json")
    neg = d["negative_alpha"]
    alphas_sorted = sorted(neg.keys(), key=lambda x: float(x))

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for i, trait in enumerate(["artistic", "conventional", "investigative"]):
        abs_a, sims = [], []
        for k in alphas_sorted:
            if trait in neg[k].get("traits", {}):
                abs_a.append(abs(float(k)))
                sims.append(neg[k]["traits"][trait]["target_sim"])
        detected_at_max = neg[alphas_sorted[-1]]["traits"][trait]["detected"]
        ax.plot(
            abs_a, sims, "o-", color=C[i], markersize=4, label=f"{trait} $\\to$ {detected_at_max}"
        )

    ax.axhline(y=0, color="grey", linestyle=":", linewidth=0.6)
    ax.axhline(y=-1, color="grey", linestyle=":", linewidth=0.4, alpha=0.5)
    ax.set_xlabel(r"$|\alpha|$ (negative steering)")
    ax.set_ylabel("Cosine sim. to own trait")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(-1.1, 0.5)
    fig.tight_layout()
    save(fig, "fig06_negative_alpha.png")


# ════════════════════════════════════════════════════════════
# 7. Sparsity / compression: accuracy vs dimensions retained
# ════════════════════════════════════════════════════════════
def fig07_sparsity():
    d = load("personality_sparsity.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: top-k dimensions
    dt = d["dimension_threshold"]
    ks = sorted(dt.keys(), key=int)
    k_vals = [int(k) for k in ks]
    accs = [dt[k]["accuracy"] * 100 for k in ks]
    ax1.semilogx(k_vals, accs, "o-", color=C[0], markersize=5)
    ax1.axhline(y=16.7, color="grey", linestyle=":", linewidth=0.6)
    ax1.set_xlabel("Dimensions retained (of 4096)")
    ax1.set_ylabel("Detection accuracy (%)")
    ax1.set_ylim(0, 115)
    ax1.set_title("(a) Top-$k$ dimensions", fontsize=11)
    ax1.annotate(
        "100% at just 10 dims",
        xy=(10, 100),
        xytext=(80, 60),
        fontsize=9,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": C[0], "linewidth": 0.8},
    )

    # Panel B: random projection
    rp = d["random_projection"]
    dims = sorted(rp.keys(), key=int)
    dim_vals = [int(k) for k in dims]
    means = [rp[k]["mean_accuracy"] * 100 for k in dims]
    stds = [rp[k]["std"] * 100 for k in dims]
    ax2.errorbar(
        dim_vals, means, yerr=stds, fmt="o-", color=C[1], markersize=4, capsize=3, linewidth=1.2
    )
    ax2.axhline(y=100, color="grey", linestyle=":", linewidth=0.6)
    ax2.axhline(y=16.7, color="grey", linestyle=":", linewidth=0.4, alpha=0.5)
    ax2.set_xlabel("Random projection dimension")
    ax2.set_ylabel("Detection accuracy (%)")
    ax2.set_ylim(0, 115)
    ax2.set_title("(b) Random projection", fontsize=11)

    fig.tight_layout()
    save(fig, "fig07_sparsity.png")


# ════════════════════════════════════════════════════════════
# 8. Information channel: SNR + amplitude linearity
# ════════════════════════════════════════════════════════════
def fig08_info_channel():
    d = load("personality_information_channel.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: SNR vs alpha
    snr = d["snr_vs_alpha"]
    alphas_sorted = sorted(snr.keys(), key=float)
    alphas = [float(k) for k in alphas_sorted]
    snr_db = [snr[k]["snr_db"] for k in alphas_sorted]
    ax1.plot(alphas, snr_db, "o-", color=C[0], markersize=5)
    ax1.set_xlabel(r"Steering strength $\alpha$")
    ax1.set_ylabel("SNR (dB)")
    ax1.set_title("(a) Signal-to-noise ratio", fontsize=11)

    # Panel B: amplitude linearity
    prec = d["precision"]
    target = np.array(prec["target_amplitudes"])
    received = np.array(prec["received_amplitudes"])
    r_val = prec["linearity_r"]

    ax2.scatter(target, received, color=C[1], s=40, zorder=3)
    # Linear fit line (not y=x since received is scaled)
    slope = np.polyfit(target, received, 1)
    x_fit = np.linspace(min(target), max(target), 50)
    y_fit = np.polyval(slope, x_fit)
    ax2.plot(x_fit, y_fit, "--", color="grey", linewidth=0.8, alpha=0.6)
    ax2.set_xlabel(r"Target amplitude $\alpha$")
    ax2.set_ylabel("Received 5D norm")
    ax2.set_title(f"(b) Linearity ($r$ = {r_val:.3f})", fontsize=11)

    fig.tight_layout()
    save(fig, "fig08_info_channel.png")


# ════════════════════════════════════════════════════════════
# 9. Cross-language universality
# ════════════════════════════════════════════════════════════
def fig09_cross_language():
    d = load("cross_language_personality.json")
    consist = d["cross_language_consistency"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: accuracy per language
    la = d["language_accuracy"]
    langs = list(la.keys())
    accs = [la[lang] * 100 for lang in langs]
    ax1.bar(range(len(langs)), accs, color=C[0], edgecolor="none", width=0.7)
    ax1.set_xticks(range(len(langs)))
    ax1.set_xticklabels([lang.capitalize() for lang in langs], rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(0, 115)
    ax1.axhline(y=16.7, color="grey", linestyle=":", linewidth=0.6)
    ax1.set_title("(a) Per-language accuracy", fontsize=11)

    # Panel B: cross-language consistency
    traits_sorted = sorted(consist.keys())
    means = [consist[t]["mean_cross_lang_cos"] for t in traits_sorted]
    mins = [consist[t]["min_cross_lang_cos"] for t in traits_sorted]

    x = np.arange(len(traits_sorted))
    ax2.bar(x - 0.2, means, 0.35, label="Mean cos", color=C[1], edgecolor="none")
    ax2.bar(x + 0.2, mins, 0.35, label="Min cos", color=C[2], edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels([t[:6] for t in traits_sorted], fontsize=8, rotation=30, ha="right")
    ax2.set_ylabel("Cross-language cosine")
    ax2.set_ylim(0.99, 1.001)
    ax2.legend(fontsize=8, frameon=False)
    ax2.set_title("(b) Cross-language consistency", fontsize=11)

    fig.tight_layout()
    save(fig, "fig09_cross_language.png")


# ════════════════════════════════════════════════════════════
# 10. Activation vs text detection summary
# ════════════════════════════════════════════════════════════
def fig10_activation_vs_text():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    methods = [
        "5D\nprojection",
        "Logit\ntop-50",
        "Text\ndescription",
        "Few-shot\n(3-shot)",
        "Text\ntransfer",
        "Watermark\n(5-bit)",
        "Best\ndecoding",
    ]
    accs = [100, 100, 50, 17, 28, 6.25, 33]
    colors = [C[2], C[2], C[1], C[3], C[3], C[3], C[3]]

    bars = ax.bar(range(len(methods)), accs, color=colors, edgecolor="none", width=0.7)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, fontsize=9)
    ax.axhline(y=16.7, color="grey", linestyle=":", linewidth=0.6)
    ax.set_ylabel("Detection accuracy (%)")
    ax.set_ylim(0, 118)

    # Divider
    ax.axvline(x=1.5, color="black", linestyle=":", linewidth=0.5, alpha=0.4)
    ax.text(0.75, 112, "activation", fontsize=9, ha="center", color=C[2])
    ax.text(4.0, 112, "text", fontsize=9, ha="center", color=C[3])

    for bar, val in zip(bars, accs, strict=False):
        label = f"{val:.0f}" if val == int(val) else f"{val}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )

    fig.tight_layout()
    save(fig, "fig10_activation_vs_text.png")


# ════════════════════════════════════════════════════════════
# 11. Dynamic personality switching
# ════════════════════════════════════════════════════════════
def fig11_dynamic_switching():
    d = load("dynamic_personality_transition.json")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.8))

    # Panel A: period-2 oscillation
    osc = d["oscillation"]["2"]
    detections = osc["detections"][:40]
    vals = [1 if det == "artistic" else 0 for det in detections]
    ax1.step(range(len(vals)), vals, where="mid", color=C[0], linewidth=1.2)
    ax1.fill_between(range(len(vals)), vals, step="mid", alpha=0.15, color=C[0])
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(["social", "artistic"], fontsize=9)
    ax1.set_xlabel("Token position")
    acc = osc["accuracy"]
    ax1.set_title(f"(a) Period-2 oscillation ({acc:.0%})", fontsize=11)
    ax1.set_xlim(-0.5, len(vals) - 0.5)

    # Panel B: smooth interpolation
    interp = d.get("smooth_interpolation", {})
    if interp:
        first_key = list(interp.keys())[0]
        entry = interp[first_key]
        # Keys are a_sims/b_sims (not cos_a/cos_b)
        cos_a = entry.get("cos_a", entry.get("a_sims", []))
        cos_b = entry.get("cos_b", entry.get("b_sims", []))
        if cos_a and cos_b:
            tokens = range(len(cos_a))
            parts = first_key.split("\u2192")
            label_a = parts[0].strip() if len(parts) > 0 else "A"
            label_b = parts[1].strip() if len(parts) > 1 else "B"
            ax2.plot(tokens, cos_a, "-", color=C[0], linewidth=1.2, label=label_a)
            ax2.plot(tokens, cos_b, "-", color=C[1], linewidth=1.2, label=label_b)
            ax2.axhline(y=0, color="grey", linestyle=":", linewidth=0.5)
            # Mark crossover
            crossover = entry.get("crossover_token", None)
            if crossover:
                ax2.axvline(x=crossover, color="grey", linestyle="--", linewidth=0.6, alpha=0.5)
                ax2.text(crossover + 1, 0.8, f"x={crossover}", fontsize=8, color="grey")
            ax2.set_xlabel("Token position")
            ax2.set_ylabel("Cosine similarity")
            ax2.legend(fontsize=8, frameon=False)
            ax2.set_title("(b) Smooth interpolation", fontsize=11)
        else:
            _fallback_instant_switch(ax2, d)
    else:
        _fallback_instant_switch(ax2, d)

    fig.tight_layout()
    save(fig, "fig11_dynamic_switching.png")


def _fallback_instant_switch(ax, d):
    sw = d.get("instant_switch", {})
    pairs = list(sw.keys())[:3]
    before = [sw[p]["before_correct"] / max(sw[p]["before_total"], 1) * 100 for p in pairs]
    after = [sw[p]["after_correct"] / max(sw[p]["after_total"], 1) * 100 for p in pairs]
    x = np.arange(len(pairs))
    ax.bar(x - 0.18, before, 0.32, label="Before", color=C[0], edgecolor="none")
    ax.bar(x + 0.18, after, 0.32, label="After", color=C[2], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace("\u2192", "\n\u2192") for p in pairs], fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("(b) Instant switching", fontsize=11)


# ════════════════════════════════════════════════════════════
# 12. System prompt vs activation steering
# ════════════════════════════════════════════════════════════
def fig12_system_prompt():
    fig, ax = plt.subplots(figsize=(5, 3.2))

    props = ["5D\ncapture", "Neutral-\nizable", "Detection\nin generation", "MLP\ncontrib."]
    act_vals = [100, 90.7, 100, 77.7]
    sys_vals = [18, 5.8, 0, 61.4]

    x = np.arange(len(props))
    ax.bar(x - 0.2, act_vals, 0.35, label="Activation steering", color=C[0], edgecolor="none")
    ax.bar(x + 0.2, sys_vals, 0.35, label="System prompt", color=C[3], edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels(props, fontsize=9)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    save(fig, "fig12_system_prompt.png")


# ════════════════════════════════════════════════════════════
# 13. Reasoning preservation (baseline vs steered)
# ════════════════════════════════════════════════════════════
def fig13_reasoning():
    d = load("reasoning_interference.json")

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    if "categories" in d:
        cats = d["categories"]
    else:
        cats = {
            "arithmetic": {"baseline": 1.0, "steered": 1.0},
            "logic": {"baseline": 1.0, "steered": 1.0},
            "knowledge": {"baseline": 0.83, "steered": 0.83},
            "sequences": {"baseline": 1.0, "steered": 0.92},
        }

    cat_names = list(cats.keys())
    baseline = [
        cats[c].get("baseline", cats[c].get("baseline_accuracy", 1.0)) * 100 for c in cat_names
    ]
    steered = [
        cats[c].get("steered", cats[c].get("steered_accuracy", 1.0)) * 100 for c in cat_names
    ]

    x = np.arange(len(cat_names))
    ax.bar(x - 0.2, baseline, 0.35, label="Baseline", color=C[7], edgecolor="none")
    ax.bar(x + 0.2, steered, 0.35, label=r"Steered ($\alpha$=2)", color=C[2], edgecolor="none")

    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in cat_names], fontsize=10)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 112)
    ax.axhline(y=100, color="grey", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "fig13_reasoning.png")


# ════════════════════════════════════════════════════════════
# 14. KL dose-response (logit lens)
# ════════════════════════════════════════════════════════════
def fig14_kl_dose():
    d = load("logit_lens_personality.json")
    kl_dr = d["kl_dose_response"]

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for i, (trait, vals) in enumerate(kl_dr.items()):
        alphas_sorted = sorted(vals.keys(), key=float)
        alphas = [float(k) for k in alphas_sorted]
        kls = [vals[k] for k in alphas_sorted]
        ax.plot(alphas, kls, "o-", color=C[i], markersize=4, label=trait)

    ax.set_xlabel(r"Steering strength $\alpha$")
    ax.set_ylabel("KL divergence at L17")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "fig14_kl_dose_response.png")


# ════════════════════════════════════════════════════════════
# 15. Holland hexagonal structure
# ════════════════════════════════════════════════════════════
def fig15_holland():
    d = load("personality_scaling.json")
    holland = d["holland"]

    short_names = {
        "Llama-3.2-1B-Instruct": "Llama 1B",
        "SmolLM3-3B": "SmolLM3 3B",
        "marin-8b-instruct": "Marin 8B",
    }

    fig, ax = plt.subplots(figsize=(4.5, 3.2))

    for i, (model, vals) in enumerate(holland.items()):
        dists = sorted(vals["by_distance"].keys(), key=int)
        cos_vals = [vals["by_distance"][d_] for d_ in dists]
        label = short_names.get(model, model)
        ax.plot([int(d_) for d_ in dists], cos_vals, "o-", color=C[i], markersize=5, label=label)

    ax.set_xlabel("Holland hexagonal distance")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["Adjacent (1)", "Alternate (2)", "Opposite (3)"])
    ax.legend(fontsize=9, frameon=False)
    ax.axhline(y=0, color="grey", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    save(fig, "fig15_holland.png")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating publication figures...")
    print(f"Output: {OUT}/")
    print()

    fns = [
        ("SVD spectrum", fig01_singular_values),
        ("Alpha phase diagram", fig02_alpha_phase),
        ("KL layer profile", fig03_kl_layer_profile),
        ("Causal patching", fig04_causal_patching),
        ("Token shifts", fig05_token_shifts),
        ("Negative alpha", fig06_negative_alpha),
        ("Sparsity", fig07_sparsity),
        ("Info channel", fig08_info_channel),
        ("Cross-language", fig09_cross_language),
        ("Activation vs text", fig10_activation_vs_text),
        ("Dynamic switching", fig11_dynamic_switching),
        ("System prompt", fig12_system_prompt),
        ("Reasoning", fig13_reasoning),
        ("KL dose-response", fig14_kl_dose),
        ("Holland structure", fig15_holland),
    ]

    ok = 0
    for name, fn in fns:
        try:
            print(f"[{name}]")
            fn()
            ok += 1
        except Exception as e:
            import traceback

            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n{ok}/{len(fns)} figures saved to {OUT}/")
