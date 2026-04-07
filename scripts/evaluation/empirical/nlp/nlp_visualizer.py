#!/usr/bin/env python3
"""
NLP Evaluation Visualizer
=========================
Generates visualizations from NLP evaluation results JSON.

Can be called standalone to visualize/tweak plots without re-running
the full evaluation pipeline, or invoked programmatically from nlp_evaluator.

Usage:
    # Visualize from a results.json file
    python -m nlp.nlp_visualizer --input ./analysis/empirical/nlp_experiments/results.json

    # Specify output directory
    python -m nlp.nlp_visualizer --input results.json --output ./plots/

    # Use box plots instead of bar charts
    python -m nlp.nlp_visualizer --input results.json --boxplot

    # Visualize a specific subset of roles
    python -m nlp.nlp_visualizer --input results.json --roles accountant teacher doctor

    # Visualize specific alpha values only
    python -m nlp.nlp_visualizer --input results.json --alphas 1.0 2.0
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional


def _collect_per_alpha_raw(results: Dict, role_filter: Optional[List[str]] = None,
                           alpha_filter: Optional[List[float]] = None) -> Dict:
    """Collect raw per-alpha metrics from results, optionally filtering by role/alpha."""
    per_alpha_raw = defaultdict(lambda: defaultdict(list))
    is_folder = 'per_role_results' in results

    if is_folder:
        for role, role_result in results.get('per_role_results', {}).items():
            if role_filter and role not in role_filter:
                continue
            for alpha_str, alpha_data in role_result.get('per_alpha', {}).items():
                alpha_val = float(alpha_str)
                if alpha_filter and not any(abs(alpha_val - a) < 0.001 for a in alpha_filter):
                    continue
                for col, metrics in alpha_data['raw_metrics'].items():
                    per_alpha_raw[alpha_val][col].extend(metrics)
    else:
        for alpha_str, alpha_data in results.get('per_alpha', {}).items():
            alpha_val = float(alpha_str)
            if alpha_filter and not any(abs(alpha_val - a) < 0.001 for a in alpha_filter):
                continue
            for col, metrics in alpha_data['raw_metrics'].items():
                per_alpha_raw[alpha_val][col].extend(metrics)

    return per_alpha_raw


def generate_visualizations(
    results: Dict,
    output_dir: str,
    is_folder: bool = False,
    boxplot: bool = False,
    role_filter: Optional[List[str]] = None,
    alpha_filter: Optional[List[float]] = None,
):
    """Generate per-alpha visualizations as bar charts (default) or box-and-whisker plots.

    Parameters
    ----------
    results : dict
        Full results dict (as produced by nlp_evaluator or loaded from results.json).
    output_dir : str
        Directory to write plots into (a ``visualizations/`` subfolder is created).
    is_folder : bool
        Whether *results* came from folder-mode analysis.
    boxplot : bool
        Use box-and-whisker plots instead of grouped bar charts.
    role_filter : list[str] | None
        If provided, only include these roles when collecting raw metrics.
    alpha_filter : list[float] | None
        If provided, only include these alpha values.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.patches as mpatches
    except ImportError:
        print("Warning: matplotlib not available, skipping visualizations")
        return

    viz_dir = Path(output_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    per_alpha_raw = _collect_per_alpha_raw(results, role_filter=role_filter, alpha_filter=alpha_filter)

    if not per_alpha_raw:
        print("  No per-alpha data available for visualization")
        return

    alphas = sorted(per_alpha_raw.keys())
    methods = list(next(iter(per_alpha_raw.values())).keys())
    display_names = {'baseline': 'reference', 'steered': 'proposed'}
    colors = {'assistant_axis': '#e74c3c', 'baseline': '#27ae60', 'steered': '#3498db'}

    coherence_metrics = [
        ('unique_bigram_ratio', 'Unique Bigram Ratio'),
        ('word_count', 'Word Count'),
        ('compression_ratio', 'Compression Ratio'),
        ('bleu', 'BLEU vs Reference'),
    ]

    style_metrics = [
        ('first_person_rate', 'First-Person Rate (%)'),
        ('modal_verb_rate', 'Verb Rate (%)'),
        ('hedge_rate', 'Hedge Rate (%)'),
        ('question_rate', 'Question Rate (%)'),
    ]

    reference_methods = ['baseline']
    bar_methods = [m for m in methods if m not in reference_methods]
    n_bar = len(bar_methods)
    n_alphas = len(alphas)
    bar_width = 0.7 / max(n_bar, 1)

    def _plot_metric_group(metrics_to_plot, title, filename):
        n_cols = 2
        n_rows = int(np.ceil(len(metrics_to_plot) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for ax in axes.flat[len(metrics_to_plot):]:
            ax.set_visible(False)

        for idx, (attr, label) in enumerate(metrics_to_plot):
            ax = axes[idx // n_cols, idx % n_cols]

            # Draw horizontal reference lines for baseline/assistant_axis
            for ref_method in reference_methods:
                if attr == 'bleu' and ref_method == 'baseline':
                    continue
                all_values = []
                for alpha in alphas:
                    raw = per_alpha_raw[alpha].get(ref_method, [])
                    all_values.extend([m[attr] for m in raw if attr in m])
                if all_values:
                    ref_mean = np.mean(all_values)
                    ref_color = colors.get(ref_method, '#888888')
                    ax.axhline(y=ref_mean, color=ref_color, linestyle='--', linewidth=1.5, alpha=0.8)
                    ax.annotate(f'{ref_mean:.2f}', xy=(0, ref_mean), xytext=(-8, 4),
                                textcoords='offset points', fontsize=7, color=ref_color,
                                fontweight='bold', ha='right', va='bottom')

            # Plot steered methods as bars or boxes
            for m_idx, method in enumerate(bar_methods):
                color = colors.get(method, '#3498db')
                offset = (m_idx - n_bar / 2 + 0.5) * bar_width

                if boxplot:
                    positions = []
                    data = []
                    for a_idx, alpha in enumerate(alphas):
                        raw = per_alpha_raw[alpha].get(method, [])
                        values = [m[attr] for m in raw if attr in m]
                        if values:
                            data.append(values)
                            positions.append(a_idx + offset)
                    if data:
                        bp = ax.boxplot(
                            data, positions=positions, widths=bar_width * 0.85,
                            patch_artist=True, showfliers=False,
                            medianprops={'color': 'black', 'linewidth': 1.2},
                        )
                        for patch in bp['boxes']:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                else:
                    means = []
                    for alpha in alphas:
                        raw = per_alpha_raw[alpha].get(method, [])
                        values = [m[attr] for m in raw if attr in m]
                        means.append(np.mean(values) if values else 0)
                    x = np.arange(n_alphas) + offset
                    bars = ax.bar(x, means, bar_width, color=color, alpha=0.8)
                    for bar, val in zip(bars, means):
                        if val != 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                                f'{val:.2f}', ha='center', va='bottom', fontsize=6,
                            )

            ax.set_title(label, fontsize=10)
            ax.set_xticks(range(n_alphas))
            ax.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=8)
            ax.set_xlabel('alpha', fontsize=8)

        # Legend: dashed lines for references, patches for steered
        legend_handles = []
        for ref_method in reference_methods:
            legend_handles.append(
                plt.Line2D([0], [0], color=colors.get(ref_method, '#888888'), linestyle='--',
                           linewidth=1.5, label=display_names.get(ref_method, ref_method))
            )
        for method in bar_methods:
            legend_handles.append(
                mpatches.Patch(color=colors.get(method, '#3498db'), alpha=0.7,
                               label=display_names.get(method, method))
            )
        fig.legend(handles=legend_handles, loc='lower center', ncol=len(legend_handles), fontsize=9)
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.savefig(viz_dir / filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {viz_dir / filename}")

    _plot_metric_group(coherence_metrics, 'Coherence Metrics by Alpha', 'coherence_comparison.png')
    _plot_metric_group(style_metrics, 'Style Metrics by Alpha', 'style_comparison.png')


def main():
    parser = argparse.ArgumentParser(
        description='Visualize NLP evaluation results (standalone)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m nlp.nlp_visualizer --input ./analysis/empirical/nlp_experiments/results.json
  python -m nlp.nlp_visualizer --input results.json --output ./plots/ --boxplot
  python -m nlp.nlp_visualizer --input results.json --roles accountant teacher
  python -m nlp.nlp_visualizer --input results.json --alphas 1.0 2.5
        """
    )

    parser.add_argument('--input', '-i', type=str, default="analysis/empirical/nlp_experiments/results.json",
                        help='Path to results.json from nlp_evaluator')
    parser.add_argument('--output', '-o', type=str, default="analysis/empirical/nlp_experiments/visualizations",
                        help='Output directory (default: same directory as input)')
    parser.add_argument('--boxplot', action='store_true',
                        help='Use box-and-whisker plots instead of bar charts')
    parser.add_argument('--roles', type=str, nargs='+', default=None,
                        help='Only visualize specific roles')
    parser.add_argument('--alphas', type=float, nargs='+', default=None,
                        help='Only visualize specific alpha values')

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    print(f"Loading results from {input_path}...")
    with open(input_path) as f:
        results = json.load(f)

    output_dir = args.output if args.output else str(input_path.parent)
    is_folder = 'per_role_results' in results

    print(f"Output: {output_dir}")
    if args.roles:
        print(f"Role filter: {args.roles}")
    if args.alphas:
        print(f"Alpha filter: {args.alphas}")

    generate_visualizations(
        results, output_dir, is_folder=is_folder, boxplot=args.boxplot,
        role_filter=args.roles, alpha_filter=args.alphas,
    )
    print("Done!")


if __name__ == "__main__":
    main()
