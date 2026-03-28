"""
Pairwise judge experiment: compares steered model responses (our method) against
prompted-baseline responses using SOTA judge prompting techniques.

SOTA techniques applied:
  - Position-swap debiasing          (MT-Bench / FairEval, NeurIPS 2023 / ACL 2023)
  - Justification-before-score       (Calibrating LLM-Based Evaluator, EMNLP 2023)
  - Gold-label reference anchoring   (Prometheus, ICLR 2024)

Input:  CSV output from gold_prompt_experiments.py
          (columns: role, layer, sample_count, alpha, temperature, question,
                    nonsteered, baseline, steered, scores, cmp_*)
Output: Augmented CSV with pairwise judge columns appended.
"""

import argparse
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pvx import setup_logging
from pvx.implementations.judges.llm_as_judge import LLMJudge, PAIRWISE_PROMPT_TEMPLATE

logger = setup_logging(name="pairwise-judge-experiments")


def _parse_pairwise_response(response: str) -> dict:
    """
    Parse the pairwise JSON response from the judge.

    Returns a dict with keys: reasoning, score_a, score_b, winner.
    Falls back to neutral values on parse failure.
    """
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response.strip())
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, AttributeError):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Pairwise response parsing failed; returning neutral scores. Raw: %r", response)
    return {"reasoning": "", "score_a": 50, "score_b": 50, "winner": "TIE"}


def _debiased_verdict(score_steered: float, score_baseline: float, margin: float = 10.0) -> str:
    """
    Return 'steered', 'baseline', or 'tie' based on debiased scores.
    Uses a margin to avoid declaring wins on negligible differences.
    10.0 chosen per LLM-judge literature: Zheng et al. (NeurIPS 2023) and Chatbot Arena
    work recommend 15-20% of scale range for subjective tasks (~15-20 pts on 0-100).
    Position-swap averaging reduces noise, making 10 pts a conservative but defensible
    threshold. The 'pw_steered_advantage' column is saved for post-hoc sensitivity analysis.
    """
    diff = score_steered - score_baseline
    if diff > margin:
        return "steered"
    elif diff < -margin:
        return "baseline"
    return "tie"


class PairwiseJudgeExperiments:
    """
    Loads a gold_prompt_experiments CSV and runs a position-swap-debiased
    pairwise judge between:
      - steered responses  (our contrastive activation method)
      - baseline responses (prompted baseline method)

    SOTA techniques applied:
      - Position-swap debiasing (MT-Bench / FairEval)
      - Justification-before-score (Calibrating LLM-Based Evaluator, EMNLP 2023)
      - Gold-label reference anchoring (Prometheus, ICLR 2024)

    Output columns added (all prefixed pw_):
      pw_reasoning_ab            judge reasoning when steered=A, baseline=B
      pw_score_steered_ab        steered score in AB ordering
      pw_score_baseline_ab       baseline score in AB ordering
      pw_winner_ab               raw winner in AB ordering
      pw_reasoning_ba            judge reasoning when baseline=A, steered=B
      pw_score_steered_ba        steered score in BA ordering
      pw_score_baseline_ba       baseline score in BA ordering
      pw_winner_ba               raw winner in BA ordering
      pw_debiased_steered_score  average steered score across both orderings
      pw_debiased_baseline_score average baseline score across both orderings
      pw_steered_advantage       debiased steered score minus debiased baseline score
      pw_debiased_winner         final verdict: steered / baseline / tie
    """

    def __init__(
        self,
        input_csv: str | None = None,
        gold_prompts_dir: str = "./persona_data/gold_labels_prompts_dataset",
        judge_model_id: str = "openai/gpt-4.1-mini",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
        save_path: str = "./experiment_data/pairwise_judge_experiments/pairwise_results.csv",
        col_a: str = "steered",
        col_b: str = "assistant_axis",
    ):
        self.input_csv = Path(input_csv) if input_csv else None
        self.gold_prompts_dir = Path(gold_prompts_dir)
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Caches to avoid re-reading JSON files per row (populated before threading starts)
        self._gold_role_descriptions: dict[str, str] = {}
        self._gold_label_responses: dict[str, str] = {}

        self.col_a = col_a
        self.col_b = col_b

        self.judge = LLMJudge(
            model=judge_model_id,
            base_url=base_url,
            api_key_env=api_key_env,
            prompt_template=PAIRWISE_PROMPT_TEMPLATE,
        )

    def _load_gold_label(self, role: str) -> tuple[str, str]:
        """
        Load (role_description, gold_label_response) for a role from its JSON file.

        role_description:    text describing the role's traits and perspective.
        gold_label_response: ideal response to anchor the judge (Prometheus-style).
                             Falls back to empty string if not present in the JSON,
                             in which case the judge still uses role_description as context.

        Returns (role_description, gold_label_response), both possibly empty strings.
        """
        if role in self._gold_role_descriptions:
            return self._gold_role_descriptions[role], self._gold_label_responses[role]

        gold_path = self.gold_prompts_dir / f"{role.replace('/', '_')}_gold_label.json"
        if not gold_path.exists():
            logger.warning("Gold label file not found for role '%s': %s", role, gold_path)
            self._gold_role_descriptions[role] = ""
            self._gold_label_responses[role] = ""
            return "", ""

        with open(gold_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        role_description = str(payload.get("role_description", "")).strip()
        gold_response = str(payload.get("gold_response", "")).strip()

        self._gold_role_descriptions[role] = role_description
        self._gold_label_responses[role] = gold_response
        return role_description, gold_response

    def _run_pairwise(
        self,
        role: str,
        role_description: str,
        gold_label: str,
        question: str,
        response_a: str,
        response_b: str,
    ) -> dict:
        """
        Single pairwise judge call.

        Returns parsed dict with keys: reasoning, score_a, score_b, winner.
        """
        gold_label_text = gold_label if gold_label else "(No gold reference available — score relative to role description only)"
        messages = [
            {
                "role": "user",
                "content": PAIRWISE_PROMPT_TEMPLATE.format(
                    role=role,
                    role_description=role_description if role_description else role,
                    gold_label=gold_label_text,
                    question=question,
                    response_a=response_a,
                    response_b=response_b,
                ),
            }
        ]
        _, raw_response = self.judge._inference_with_client(messages=messages)
        return _parse_pairwise_response(raw_response)

    def _evaluate_row(
        self,
        role: str,
        role_description: str,
        gold_label: str,
        question: str,
        steered: str,
        baseline: str,
    ) -> dict:
        """
        Run position-swap-debiased pairwise evaluation for a single row.

        Runs the judge twice:
          AB: steered=A, baseline=B
          BA: baseline=A, steered=B  (position-swapped)

        Debiased score = average of each model's score across both orderings.
        This eliminates positional bias per MT-Bench / FairEval methodology.

        Returns dict of all pw_* columns.
        """
        def _clamp(v: float) -> float:
            return max(0.0, min(100.0, v))

        # Run 1: steered = A, baseline = B
        ab = self._run_pairwise(role, role_description, gold_label, question,
                                response_a=steered, response_b=baseline)
        score_steered_ab  = _clamp(float(ab.get("score_a", 50)))
        score_baseline_ab = _clamp(float(ab.get("score_b", 50)))

        logger.info("AB | steered_score=%s baseline_score=%s winner=%s",
                    score_steered_ab, score_baseline_ab, ab.get("winner"))

        # Run 2: baseline = A, steered = B  (position-swapped)
        ba = self._run_pairwise(role, role_description, gold_label, question,
                                response_a=baseline, response_b=steered)
        score_baseline_ba = _clamp(float(ba.get("score_a", 50)))
        score_steered_ba  = _clamp(float(ba.get("score_b", 50)))

        logger.info("BA | steered_score=%s baseline_score=%s winner=%s",
                    score_steered_ba, score_baseline_ba, ba.get("winner"))

        # Debiased scores: average each model's score across both orderings
        debiased_steered  = (score_steered_ab + score_steered_ba) / 2
        debiased_baseline = (score_baseline_ab + score_baseline_ba) / 2
        advantage = debiased_steered - debiased_baseline

        logger.info("Debiased | steered=%.2f baseline=%.2f advantage=%.2f verdict=%s",
                    debiased_steered, debiased_baseline, advantage,
                    _debiased_verdict(debiased_steered, debiased_baseline))

        return {
            "pw_reasoning_ab":            ab.get("reasoning", ""),
            "pw_score_steered_ab":        score_steered_ab,
            "pw_score_baseline_ab":       score_baseline_ab,
            "pw_winner_ab":               ab.get("winner", "TIE"),
            "pw_reasoning_ba":            ba.get("reasoning", ""),
            "pw_score_steered_ba":        score_steered_ba,
            "pw_score_baseline_ba":       score_baseline_ba,
            "pw_winner_ba":               ba.get("winner", "TIE"),
            "pw_debiased_steered_score":  round(debiased_steered, 2),
            "pw_debiased_baseline_score": round(debiased_baseline, 2),
            "pw_steered_advantage":       round(advantage, 2),
            "pw_debiased_winner":         _debiased_verdict(debiased_steered, debiased_baseline),
        }

    def _load_existing_results(self) -> pd.DataFrame:
        if not self.save_path.exists() or self.save_path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            df = pd.read_csv(self.save_path)
            return df if not df.empty else pd.DataFrame()
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def evaluate_pairwise(
        self,
        roles: list[str] | None = None,
        layers: list[int] | None = None,
        sample_counts: list[int] | None = None,
        save_each: bool = True,
        workers: int = 1,
    ) -> pd.DataFrame:
        """
        Run pairwise evaluation over rows in the input CSV.

        Args:
            roles:         If provided, only evaluate rows matching these roles.
            layers:        If provided, only evaluate rows matching these layers.
            sample_counts: If provided, only evaluate rows matching these sample counts.
            save_each:     If True, saves progress to CSV after each completed row.
            workers:       Number of concurrent threads for parallel judge calls.
                           Set >1 to fan out across a vLLM server. Default: 1 (sequential).

        Returns:
            DataFrame with all original columns plus pw_* pairwise columns.
        """
        if not self.input_csv.exists():
            raise FileNotFoundError(f"Input CSV not found: {self.input_csv}")

        input_df = pd.read_csv(self.input_csv)
        logger.info("Loaded %d rows from %s", len(input_df), self.input_csv)

        # Filter to requested subset if provided
        if roles:
            input_df = input_df[input_df["role"].isin(roles)]
        if layers:
            input_df = input_df[input_df["layer"].isin(layers)]
        if sample_counts:
            input_df = input_df[input_df["sample_count"].isin(sample_counts)]

        logger.info("Evaluating %d rows after filtering (roles=%s layers=%s sample_counts=%s)",
                    len(input_df), roles, layers, sample_counts)

        existing_df = self._load_existing_results()

        # Build O(1) lookup set of already-evaluated row keys for resume support
        evaluated_keys: set[tuple] = set()
        if not existing_df.empty:
            for _, r in existing_df.iterrows():
                evaluated_keys.add((
                    str(r["role"]), r["layer"], r["sample_count"],
                    r["alpha"], r["temperature"], str(r["question"]),
                ))

        # Pre-load all gold labels before threading to avoid file I/O contention
        for role in input_df["role"].unique():
            self._load_gold_label(role)

        completed_rows: list[dict] = []
        lock = threading.Lock()

        def _process_row(item: tuple) -> dict | None:
            idx, row = item
            key = (
                str(row["role"]), row["layer"], row["sample_count"],
                row["alpha"], row["temperature"], str(row["question"]),
            )

            with lock:
                if key in evaluated_keys:
                    logger.info("Skipping already evaluated row %d (role=%s)", idx, key[0])
                    return None

            role      = str(row["role"])
            question  = str(row["question"])
            steered   = str(row[self.col_a])
            baseline  = str(row[self.col_b])

            role_description = self._gold_role_descriptions.get(role, "")
            gold_label       = self._gold_label_responses.get(role, "")

            logger.info(
                "=== Row %d | role=%s layer=%s sample_count=%s alpha=%s ===",
                idx, role, row["layer"], row["sample_count"], row["alpha"],
            )
            logger.info(
                "Response lengths | %s=%d words  %s=%d words",
                self.col_a, len(steered.split()), self.col_b, len(baseline.split()),
            )

            pw_scores = self._evaluate_row(
                role=role,
                role_description=role_description,
                gold_label=gold_label,
                question=question,
                steered=steered,
                baseline=baseline,
            )
            return {**row.to_dict(), **pw_scores}

        rows_to_process = list(input_df.iterrows())

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_process_row, item) for item in rows_to_process]
            for future in as_completed(futures):
                result = future.result()
                if result is None:
                    continue

                with lock:
                    evaluated_keys.add((
                        str(result["role"]), result["layer"], result["sample_count"],
                        result["alpha"], result["temperature"], str(result["question"]),
                    ))
                    completed_rows.append(result)

                    if save_each:
                        all_rows = (
                            pd.concat([existing_df, pd.DataFrame(completed_rows)], ignore_index=True)
                            if not existing_df.empty
                            else pd.DataFrame(completed_rows)
                        )
                        all_rows.sort_values(
                            by=["role", "layer", "sample_count", "alpha", "temperature"],
                            inplace=True,
                        )
                        all_rows.to_csv(self.save_path, index=False)
                        logger.info("Saved progress (%d completed) to %s",
                                    len(completed_rows), self.save_path)

        # Final save
        final_df = (
            pd.concat([existing_df, pd.DataFrame(completed_rows)], ignore_index=True)
            if completed_rows and not existing_df.empty
            else pd.DataFrame(completed_rows) if completed_rows
            else existing_df
        )
        if not final_df.empty:
            final_df.sort_values(
                by=["role", "layer", "sample_count", "alpha", "temperature"],
                inplace=True,
            )
            final_df.to_csv(self.save_path, index=False)
            logger.info("Pairwise evaluation complete. Results saved to %s", self.save_path)
        else:
            logger.info("No rows to evaluate for %s — skipping save.", self.input_csv)
        return final_df

    def evaluate_pairwise_dir(
        self,
        input_dir: str,
        output_dir: str,
        glob_pattern: str = "Comparison_GoldStandard_*.csv",
        save_each: bool = True,
        workers: int = 1,
    ) -> None:
        """
        Run pairwise evaluation over every per-role CSV in input_dir.

        Reads each matching CSV, runs evaluate_pairwise, and writes per-role
        output files to output_dir. The original CSVs are never modified.

        Args:
            input_dir:    Directory containing per-role input CSVs.
            output_dir:   Directory to write per-role pairwise output CSVs.
            glob_pattern: Glob to select input files (default: Comparison_GoldStandard_*.csv).
            save_each:    If True, saves progress after each completed row.
            workers:      Number of concurrent judge threads.
        """
        input_dir_path  = Path(input_dir)
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(input_dir_path.glob(glob_pattern))
        if not csv_files:
            raise FileNotFoundError(
                f"No files matching '{glob_pattern}' found in {input_dir_path}"
            )

        logger.info("Found %d input CSVs in %s", len(csv_files), input_dir_path)

        for csv_file in csv_files:
            role = csv_file.stem.replace("Comparison_GoldStandard_", "")
            out_path = output_dir_path / f"pairwise_{role}.csv"

            logger.info("=== Processing role: %s ===", role)

            self.input_csv = csv_file
            self.save_path = out_path

            try:
                self.evaluate_pairwise(save_each=save_each, workers=workers)
            except Exception as e:
                logger.error("Failed to evaluate role '%s': %s", role, e)
                continue

        logger.info("All roles processed. Results written to %s", output_dir_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Pairwise judge: steered (our method) vs baseline"
    )
    ap.add_argument(
        "-i", "--input_csv",
        default=None,
        help="Path to a single gold_prompt_experiments output CSV (mutually exclusive with --input_dir)",
    )
    ap.add_argument(
        "--input_dir",
        default=None,
        help="Directory of per-role CSVs (Comparison_GoldStandard_*.csv). Processes each in turn.",
    )
    ap.add_argument(
        "--output_dir",
        default="./experiment_data/pairwise_judge_experiments/",
        help="Output directory when using --input_dir (default: ./experiment_data/pairwise_judge_experiments/)",
    )
    ap.add_argument(
        "--gold_prompts_dir",
        default="./persona_data/gold_labels_prompts_dataset",
        help="Directory containing {role}_gold_label.json files",
    )
    ap.add_argument(
        "--judge_model",
        default="openai/gpt-4.1-mini",
        help="Judge model ID (default: openai/gpt-4.1-mini)",
    )
    ap.add_argument(
        "--base_url",
        default="https://openrouter.ai/api/v1",
        help="Base URL for the judge inference server (default: OpenRouter)",
    )
    ap.add_argument(
        "--api_key_env",
        default="OPENROUTER_API_KEY",
        help="Environment variable holding the API key (default: OPENROUTER_API_KEY)",
    )
    ap.add_argument(
        "-o", "--save_path",
        default="./experiment_data/pairwise_judge_experiments/pairwise_results.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "-r", "--roles",
        nargs="+", type=str, default=None,
        help="Only evaluate rows for these roles (default: all roles in CSV)",
    )
    ap.add_argument(
        "-l", "--layers",
        nargs="+", type=int, default=None,
        help="Only evaluate rows for these layers (default: all layers in CSV)",
    )
    ap.add_argument(
        "-s", "--sample_counts",
        nargs="+", type=int, default=None,
        help="Only evaluate rows for these sample counts (default: all sample counts in CSV)",
    )
    ap.add_argument(
        "-w", "--workers",
        type=int, default=20,
        help="Number of concurrent judge threads (default: 20, set to 1 for sequential)",
    )
    ap.add_argument(
        "--col_a",
        default="steered",
        help="CSV column to use as response A / our method (default: steered)",
    )
    ap.add_argument(
        "--col_b",
        default="assistant_axis",
        help="CSV column to use as response B / comparator (default: assistant_axis)",
    )
    args = ap.parse_args()

    if not args.input_csv and not args.input_dir:
        ap.error("One of --input_csv or --input_dir is required.")

    experiment = PairwiseJudgeExperiments(
        input_csv=args.input_csv,
        gold_prompts_dir=args.gold_prompts_dir,
        judge_model_id=args.judge_model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        save_path=args.save_path,
        col_a=args.col_a,
        col_b=args.col_b,
    )

    if args.input_dir:
        experiment.evaluate_pairwise_dir(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_each=True,
            workers=args.workers,
        )
    else:
        experiment.evaluate_pairwise(
            roles=args.roles,
            layers=args.layers,
            sample_counts=args.sample_counts,
            save_each=True,
            workers=args.workers,
        )
