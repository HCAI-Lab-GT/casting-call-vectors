"""
Example: Running the optimized roles pipeline.

This script demonstrates how to use the optimized CPU/GPU two-phase architecture
for persona vector extraction.

Run this to understand the workflow or as a simple validation test.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from pvx import setup_logging
from pvx.implementations.roles_optimized import RoleQAGenerator, RoleActivationExtractor
from pvx.implementations.roles.role_dataset import RoleDataset

logger = setup_logging(name="example-roles-optimized")


def example_cpu_task():
    """Example: Run the CPU task to generate and judge Q/A responses."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE: CPU Task - Generate and Judge Q/A Responses")
    logger.info("=" * 70)

    # Initialize Q/A generator
    generator = RoleQAGenerator(
        target_model_id="qwen2.5:7b-instruct",
        temperature=0.9,
        max_new_tokens=2048,
    )

    roles = ["Lawyers"]

    for role in roles:
        logger.info(f"\nProcessing role: {role}")

        # Check if dataset exists
        try:
            dataset = RoleDataset.from_json(role, dirpath="./persona_data/role_datasets/")
        except Exception as e:
            logger.error(f"Dataset not found for {role}, skipping: {e}")
            continue

        # Generate and judge Q/A
        logger.info(f"Generating and judging Q/A pairs...")
        qa_data = generator.generate_and_judge_qa(
            role=role,
            dataset=dataset,
            target_pairs=5,  # Use small number for example
            max_retries=1,
        )

        # Save results
        output_path = generator.save_qa_responses(
            qa_data=qa_data,
            role=role,
            output_dir="./persona_data/model_qa_responses",
        )

        logger.info(f"✅ Saved {len(qa_data['positive'])} valid Q/A pairs to {output_path}")


def example_gpu_task():
    """Example: Run the GPU task to extract activations from Q/A responses."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE: GPU Task - Extract Activations from Q/A")
    logger.info("=" * 70)

    roles = ["Lawyers"]

    for role in roles:
        logger.info(f"\nProcessing role: {role}")

        # Check if Q/A responses exist
        safe_model_id = "qwen2.5:7b-instruct".replace("/", "__")
        qa_path = Path("./persona_data/model_qa_responses") / safe_model_id / f"{role}.json"

        if not qa_path.exists():
            logger.error(f"Q/A responses not found at {qa_path}")
            logger.info("Please run CPU task first")
            continue

        try:
            # Initialize extractor
            extractor = RoleActivationExtractor(
                role=role,
                target_model_id="qwen2.5:7b-instruct",
                layer=14,
                dataset_dirpath="./persona_data/role_datasets/",
                qa_responses_path=str(qa_path),
                safetensors_dir="./persona_data/model_inits/",
            )

            # Extract activations
            logger.info("Extracting activations...")
            prompt_vec, response_vec, all_layers_vec = extractor.extract_persona_vector()

            # Save to safetensors
            logger.info("Saving persona vectors...")
            extractor.save_to_safetensors(filepath="./persona_data/model_inits/")

            logger.info(f"✅ Successfully extracted persona vectors for {role}")
            logger.info(f"   Prompt persona vector shape: {tuple(prompt_vec.shape)}")
            logger.info(f"   Response persona vector shape: {tuple(response_vec.shape)}")

        except Exception as e:
            logger.error(f"Failed to extract activations for {role}: {e}")


def example_full_pipeline():
    """Example: Run the full pipeline (CPU + GPU)."""
    logger.info("\n" + "=" * 70)
    logger.info("EXAMPLE: Full Pipeline - CPU Task + GPU Task")
    logger.info("=" * 70)

    logger.info("\n[Phase 1] CPU Task: Generate and Judge Q/A...")
    example_cpu_task()

    logger.info("\n[Phase 2] GPU Task: Extract Activations...")
    example_gpu_task()

    logger.info("\n" + "=" * 70)
    logger.info("✅ Full pipeline completed!")
    logger.info("=" * 70)


def main():
    ap = argparse.ArgumentParser(
        description="Example: Optimized roles persona vector extraction"
    )
    ap.add_argument(
        "--phase",
        choices=["cpu", "gpu", "full"],
        default="full",
        help="Which phase to run (default: full)",
    )
    args = ap.parse_args()

    if args.phase == "cpu":
        example_cpu_task()
    elif args.phase == "gpu":
        example_gpu_task()
    else:  # full
        example_full_pipeline()


if __name__ == "__main__":
    main()
