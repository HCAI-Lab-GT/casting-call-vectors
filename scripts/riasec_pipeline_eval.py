import argparse

from pvx import setup_logging
from pvx.pvx_models.persona_dataset import PersonaDataset
from pvx.pvx_models.riasec_persona_model import RIASECPersonaModel
from pvx.utils.riasec_utils import RIASECHelpers

logger = setup_logging(name="riasec-persona-model")


def main():
    parser = argparse.ArgumentParser(description="RIASEC Pipeline Evaluation")
    # Note: "trait" is used as a generic term here for consistency with the broader persona
    # pipeline, even though RIASEC literature technically refers to these as "types" or
    # "vocational interest categories" rather than personality traits.
    parser.add_argument(
        "--trait", type=str, required=True, help="RIASEC type to evaluate (e.g., conventional)"
    )
    parser.add_argument(
        "--pregenerate", action="store_true", help="Pregenerate responses for the trait"
    )
    parser.add_argument(
        "--generate_dataset", action="store_true", help="Generate dataset and inject questions"
    )
    parser.add_argument(
        "--yaml_path", type=str, default="./configs/riasec.yaml", help="Path to RIASEC YAML file"
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name"
    )
    parser.add_argument("--layer", type=int, default=14, help="Layer for persona steering")
    parser.add_argument(
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not trait dataset)",
    )
    parser.add_argument("--judge_threshold", type=float, default=50, help="Threshold for LLM judge")
    parser.add_argument(
        "--target_count", type=int, default=5, help="Target Count for pregenerated responses."
    )
    args = parser.parse_args()

    if args.generate_dataset:
        logger.info(f"Generating dataset for trait: {args.trait}")
        PersonaDataset.from_json(
            trait=args.trait, from_riasec=True, riasec_config_path=args.yaml_path
        )
        logger.info("Dataset generated and questions injected.")

    accepted_responses = None
    if args.pregenerate:
        logger.info(f"Pregenerating responses for trait: {args.trait}")
        accepted_responses = RIASECPersonaModel.pre_generate_riasec_pos_neg_responses(
            trait=args.trait, target_count=args.target_count
        )
        logger.info("=== Accepted Responses ===")
        logger.info(accepted_responses)

        if accepted_responses:
            logger.info(f"Updating YAML file {args.yaml_path} for trait {args.trait}")
            RIASECHelpers.update_riasec_yaml(args.yaml_path, args.trait, accepted_responses)
            logger.info("YAML file updated.")

    logger.info(f"Loading or creating RIASECPersonaModel for trait: {args.trait}")
    pvx = RIASECPersonaModel.load_or_create(
        target_model_id=args.model_name,
        trait=args.trait,
        layer=args.layer,
        json_filepath=args.json_filepath,
    )

    logger.info("Running RIASEC judge on the model...")
    from pvx.pvx_models.judges.riasec_judge import RIASECJudge

    riasec_judge = RIASECJudge(riasec_yaml_path=args.yaml_path)
    results, counts = riasec_judge.evaluate_persona(
        pvx, alpha=-5, max_new_tokens=200, temperature=0.9
    )
    logger.info("=== RIASEC Judge Results ===")
    for trait, trait_results in results.items():
        logger.info(f"Trait: {trait}")
        for characteristic, res in trait_results.items():
            logger.info(
                f"Characteristic: {characteristic}\nResponse: {res['response']}\nJudgment: {res['judgment']}\n---"
            )
    logger.info("=== RIASEC YES Counts ===")
    logger.info(counts)
    print("Final RIASEC Judge Results:")
    for trait, trait_results in results.items():
        print(f"Trait: {trait}")
        for characteristic, res in trait_results.items():
            print(
                f"Characteristic: {characteristic}\nResponse: {res['response']}\nJudgment: {res['judgment']}\n---"
            )
    print("RIASEC YES Counts:")
    print(counts)


if __name__ == "__main__":
    main()
