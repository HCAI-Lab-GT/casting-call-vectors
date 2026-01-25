#!/usr/bin/env python3
"""Migrate existing outputs to the new multi-model directory structure.

This script moves:
  outputs/riasec_validation -> outputs/runs/smollm2-135m-instruct/{run_id}/
  outputs/validation_test -> outputs/runs/smollm2-135m-instruct/{run_id}/

And creates run_config.json files for each.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path


def get_dir_timestamp(dir_path: Path) -> str:
    """Get a timestamp from directory modification time."""
    mtime = dir_path.stat().st_mtime
    dt = datetime.fromtimestamp(mtime)
    return dt.strftime("%Y-%m-%d_%H%M%S")


def migrate_output_dir(
    source_dir: Path,
    base_output_dir: Path,
    model_id: str,
    model_slug: str,
    layer: int,
    description: str,
) -> Path:
    """Migrate a single output directory to new structure."""
    # Generate run_id from directory timestamp
    run_id = get_dir_timestamp(source_dir) + "_legacy"

    # Create target directory
    target_dir = base_output_dir / model_slug / run_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # Move vectors/ if exists
    source_vectors = source_dir / "vectors"
    if source_vectors.exists():
        target_vectors = target_dir / "vectors"
        if not target_vectors.exists():
            shutil.copytree(source_vectors, target_vectors)
            print(f"  Copied vectors: {source_vectors} -> {target_vectors}")

    # Move analysis/ if exists
    source_analysis = source_dir / "analysis"
    if source_analysis.exists():
        target_analysis = target_dir / "analysis"
        if not target_analysis.exists():
            shutil.copytree(source_analysis, target_analysis)
            print(f"  Copied analysis: {source_analysis} -> {target_analysis}")

    # Try to get additional info from analysis files
    num_questions = 5
    num_personas = 0
    analysis_file = source_analysis / "riasec_validation.json"
    if analysis_file.exists():
        with open(analysis_file) as f:
            analysis = json.load(f)
            num_questions = analysis.get("num_questions", 5)
            num_personas = analysis.get("total_personas", 0)

    # Count vectors if not found in analysis
    if num_personas == 0:
        vectors_dir = target_dir / "vectors"
        if vectors_dir.exists():
            num_personas = len(list(vectors_dir.glob("*.pt")))

    # Create run_config.json
    run_config = {
        "run_id": run_id,
        "model_id": model_id,
        "model_slug": model_slug,
        "layer": layer,
        "num_questions": num_questions,
        "personas_dir": "persona_data/vocational_personas/instructions",
        "git_hash": "legacy",
        "git_branch": "unknown",
        "started_at": datetime.fromtimestamp(source_dir.stat().st_mtime).isoformat(),
        "completed_at": datetime.fromtimestamp(source_dir.stat().st_mtime).isoformat(),
        "wandb_run_id": None,
        "base_output_dir": str(base_output_dir),
        "extra": {
            "migrated_from": str(source_dir),
            "description": description,
            "num_personas": num_personas,
        },
    }

    config_path = target_dir / "run_config.json"
    with open(config_path, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"  Created: {config_path}")

    return target_dir


def main():
    """Run the migration."""
    print("=" * 60)
    print("Migrating outputs to multi-model structure")
    print("=" * 60)

    outputs_dir = Path("outputs")
    runs_dir = outputs_dir / "runs"

    # Migrate riasec_validation
    riasec_dir = outputs_dir / "riasec_validation"
    if riasec_dir.exists():
        print(f"\nMigrating: {riasec_dir}")
        target = migrate_output_dir(
            source_dir=riasec_dir,
            base_output_dir=runs_dir,
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            model_slug="smollm2-135m-instruct",
            layer=4,
            description="RIASEC balanced validation (12 personas, 2 per type)",
        )
        print(f"  -> {target}")

    # Migrate validation_test
    validation_dir = outputs_dir / "validation_test"
    if validation_dir.exists():
        print(f"\nMigrating: {validation_dir}")
        target = migrate_output_dir(
            source_dir=validation_dir,
            base_output_dir=runs_dir,
            model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
            model_slug="smollm2-135m-instruct",
            layer=4,
            description="Initial pipeline validation test",
        )
        print(f"  -> {target}")

    # Show summary
    print("\n" + "=" * 60)
    print("Migration complete!")
    print("=" * 60)

    # List new structure
    print("\nNew structure:")
    if runs_dir.exists():
        for model_dir in sorted(runs_dir.iterdir()):
            if model_dir.is_dir():
                print(f"\n  {model_dir.name}/")
                for run_dir in sorted(model_dir.iterdir()):
                    if run_dir.is_dir():
                        config = run_dir / "run_config.json"
                        status = "✓" if config.exists() else "?"
                        vectors = len(list((run_dir / "vectors").glob("*.pt"))) if (run_dir / "vectors").exists() else 0
                        print(f"    {status} {run_dir.name}/ ({vectors} vectors)")

    print("\nNote: Original directories NOT removed. Remove manually after verification:")
    print("  rm -rf outputs/riasec_validation")
    print("  rm -rf outputs/validation_test")


if __name__ == "__main__":
    main()
