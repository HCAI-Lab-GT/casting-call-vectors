import argparse
import csv
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
from pvx import setup_logging

logger = setup_logging(name="plot-gold-prompt-role-graphs")
FILTER_FIELDS = ("layer", "sample_count", "alpha", "temperature")
JUDGE_FIELDS = (("Assistant Axis", "assistant_axis_score"), ("Steered", "steered_score"), ("Baseline", "baseline_score"))
METRICS = (
    ("Emotional Register", "assistant_axis_cmp_emotional_register", "cmp_emotional_register"),
    ("Vocab Choice", "assistant_axis_cmp_vocab_choice", "cmp_vocab_choice"),
    ("Social Dynamic", "assistant_axis_cmp_social_dynamic", "cmp_social_dynamic"),
    ("Motivation", "assistant_axis_cmp_motivation", "cmp_motivation"),
    ("Worldview Alignment", "assistant_axis_cmp_worldview_alignment", "cmp_worldview_alignment"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create one comparison graph per role")
    parser.add_argument("--input_dir", default="./experiment_data/gold_prompt_experiments")
    parser.add_argument("--output_dir", default="./analysis/empirical/gold_prompt_role_graphs")
    parser.add_argument("--roles", nargs="+", default=None)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--sample_count", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    return parser.parse_args()


def discover_csv_paths(input_dir: Path, roles: list[str] | None) -> list[Path]:
    paths = sorted(input_dir.glob("Comparison_GoldStandard_*.csv"))
    if not roles:
        return paths
    allowed = {role.replace("/", "_") for role in roles}
    return [path for path in paths if path.stem.removeprefix("Comparison_GoldStandard_") in allowed]


def parse_number(value: str) -> float | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def average_column(rows: list[dict[str, str]], column: str) -> float | None:
    values = [value for value in (parse_number(row.get(column, "")) for row in rows) if value is not None]
    return mean(values) if values else None


def row_matches_filters(row: dict[str, str], args: argparse.Namespace) -> bool:
    for key in FILTER_FIELDS:
        expected = getattr(args, key)
        if expected is None:
            continue
        actual = parse_number(row.get(key, ""))
        if actual is None or abs(actual - float(expected)) > 1e-9:
            return False
    return True


def load_rows(csv_path: Path, args: argparse.Namespace) -> list[dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return [row for row in csv.DictReader(handle) if row_matches_filters(row, args)]


def summarize_metrics(rows: list[dict[str, str]]) -> list[tuple[str, float, float]]:
    summary = []
    for label, aa_column, steered_column in METRICS:
        assistant_axis_score = average_column(rows, aa_column)
        steered_score = average_column(rows, steered_column)
        if assistant_axis_score is None or steered_score is None:
            continue
        summary.append((label, (assistant_axis_score), (steered_score)))
    return summary


def build_caption(row_count: int, args: argparse.Namespace) -> str:
    parts = [f"rows={row_count}"]
    for key in FILTER_FIELDS:
        value = getattr(args, key)
        if value is not None:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def build_judge_box(rows: list[dict[str, str]]) -> str:
    lines = []
    for label, column in JUDGE_FIELDS:
        score = average_column(rows, column)
        if score is not None:
            lines.append(f"{label}: {score:.2f}")
    return f"Avg Judge\n{"\n".join(lines)}" if lines else ""


def plot_role_summary(role: str, summary: list[tuple[str, float, float]], output_path: Path, caption: str, judge_box: str) -> None:
    labels = [item[0] for item in summary]
    assistant_axis_values = [item[1] for item in summary]
    steered_values = [item[2] for item in summary]
    width = 0.36
    positions = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([position - width / 2 for position in positions], assistant_axis_values, width, label="Assistant Axis")
    ax.bar([position + width / 2 for position in positions], steered_values, width, label="Steered")
    ax.set_xticks(positions, labels, rotation=24, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean gold comparison score (0-100)")
    ax.set_title(f"{role}: Assistant Axis vs Steered")
    ax.legend()
    if judge_box:
        ax.text(0.02, 0.98, judge_box, transform=ax.transAxes, va="top", ha="left", fontsize=9, bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9})
    fig.text(0.01, 0.01, caption, fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_paths = discover_csv_paths(Path(args.input_dir), args.roles)
    if not csv_paths:
        logger.info("No matching role CSVs found in %s", args.input_dir)
        return
    output_dir = Path(args.output_dir)
    for csv_path in csv_paths:
        role = csv_path.stem.removeprefix("Comparison_GoldStandard_")
        rows = load_rows(csv_path, args)
        if not rows:
            logger.warning("Skipping %s because no rows matched the current filters", role)
            continue
        summary = summarize_metrics(rows)
        if not summary:
            logger.warning("Skipping %s because the assistant-axis comparison columns are empty", role)
            continue
        output_path = output_dir / f"{role}_sample_count_{args.sample_count if args.sample_count is not None else 'mixed'}_alpha_{args.alpha if args.alpha is not None else 'mixed'}.png"
        plot_role_summary(role, summary, output_path, build_caption(len(rows), args), build_judge_box(rows))
        logger.info("Saved %s", output_path)


if __name__ == "__main__":
    main()