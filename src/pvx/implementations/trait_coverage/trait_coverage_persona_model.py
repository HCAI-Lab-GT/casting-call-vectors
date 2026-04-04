import argparse
import json
from pathlib import Path
from typing import Optional, override

from pvx import setup_logging

from pvx.implementations.trait_coverage.trait_coverage_dataset import TraitCoverageDataset

from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.base.persona_model import PersonaModel

from pvx.implementations.judges.llm_as_judge import LLMJudge, PROMPT_TEMPLATE

logger = setup_logging(name="trait-coverage-persona-model")

# class TraitCoveragePersonaModel(AbstractPersonaModel):
class TraitCoveragePersonaModel(PersonaModel):
    def __init__(self,
                 trait: str,
                 dataset: Optional[TraitCoverageDataset] = None,
                 dataset_dirpath: str = './persona_data/trait_datasets/',
                 *args,
                 **kwargs):
        self.trait = trait
        self.dataset = dataset if dataset else TraitCoverageDataset.from_json(trait, dirpath=dataset_dirpath)
        
        super().__init__(trait=trait, *args, **kwargs)

    @override
    @classmethod
    def get_path(cls,
                 target_model_id: str, concept: str,
                 safetensors_dir: str = "./persona_data/trait_inits/", use_json: bool = False,
                 kwargs: Optional[dict] = None, obj: Optional[AbstractPersonaModel] = None):
        safe_model_id = target_model_id.replace("/", "__")

        layer = \
            (kwargs.get("layer") if kwargs else None) or \
            (obj.layer_steering if obj else None) or \
            "unknown_layer"

        sample_counts = \
            (kwargs.get("target_pairs") if kwargs else None) or \
            (obj.target_pairs if obj else None) or \
            "unknown_count"

        filepath = f"{concept}_persona_initialization/{safe_model_id}_layer{layer}_count{sample_counts}.{'safetensors' if not use_json else 'json'}"
        safetensors_path = Path(safetensors_dir) / filepath

        return safetensors_path, filepath
        
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-t", "--traits", nargs="+", type=str, default=None, help="Traits of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument(
        "-l", "--layers", nargs="+", type=int,default=[16], help="List of layers to extract activations from (default: 16)"
    )
    ap.add_argument(
        "-N", "--sample_counts", nargs="+", type=int,default=[40], help="List of question counts to extract activations from (default: 40)"
    )
    ap.add_argument("--temperature", type=float, default=0.2, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to generate response for",
    )
    ap.add_argument(
        "-f",
        "--safetensor_dir",
        type=str,
        default="./persona_data/trait_inits/",
        help="Directory for the model init safetensor (not role dataset)",
    )
    args = ap.parse_args()
    
    trait_list = args.traits or [
        "sarcastic",
        "verbose",
        "evil",
        "kind",
        "analytical",
        "empathetic",
        "humorous",
        "optimistic",
        "pessimistic",
        "creative",
        "brave",
        "cautious",
    ]

    for trait in trait_list:
        logger.info("=== Processing trait: %s ===", trait)

        for layer in args.layers:
            logger.info("Extracting activations from layer %d", layer)

            for sample_count in args.sample_counts:
                try:
                    pvx = TraitCoveragePersonaModel.load_or_create(
                        target_model_id=args.model,
                        concept=trait,
                        layer=layer,
                        target_pairs=sample_count,
                        safetensors_dir=args.safetensor_dir,
                    )
                    logger.info(
                        "Successfully loaded/created TraitCoveragePersonaModel on layer %d "
                        "with %d pairs for trait '%s'",
                        layer, sample_count, trait,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to load or create TraitCoveragePersonaModel on layer %d "
                        "with %d pairs for trait '%s': %s",
                        layer, sample_count, trait, e,
                    )
                    continue