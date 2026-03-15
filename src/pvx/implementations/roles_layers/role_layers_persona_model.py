import argparse
import json
from pathlib import Path
from typing import Optional, override

from pvx import setup_logging

from pvx.implementations.roles.role_persona_model import RolePersonaModel
from pvx.implementations.judges.llm_as_judge import LLMJudge, PROMPT_TEMPLATE

logger = setup_logging(name="role-layers-persona-model")

class RoleLayersPersonaModel(RolePersonaModel):
    @override
    @classmethod
    def get_path(cls, 
                 target_model_id: str, concept: str, 
                 safetensors_dir: str= "./persona_data/model_layer_inits/", use_json: bool=False, 
                 kwargs: Optional[dict]=None, obj: Optional[RolePersonaModel]=None):
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
    
    # @override
    # def save_to_safetensors(self, filepath: str = None, **args) -> str:
    #     filepath = filepath or self.safetensors_dir
    #     super().save_to_safetensors(filepath=filepath, **args)
        
    # @override
    # def save_to_json(self, filepath: str = None, **args) -> str:
    #     filepath = filepath or self.safetensors_dir
    #     super().save_to_json(filepath=filepath, **args)
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-m", "--model", type=str, default="allenai/Olmo-3-7B-Instruct", help="HF model typically",
    )
    ap.add_argument(
        "-r", "--roles", nargs="+", type=str, default=["Lawyers"], help="Roles of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering"
    )
    ap.add_argument(
        "-l", "--layers", nargs="+", type=int,default=[14], help="List of layers to extract activations from (default: 14)"
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
        default="./persona_data/model_layer_inits/",
        help="Directory for the model init safetensor (not role dataset)",
    )
    ap.add_argument(
        "--cot",
        action="store_true",
        help="Whether to use chain-of-thought prompting (default: False)",
    )
    ap.add_argument(
        "-t",
        "--trait",
        type=str,
        default=None,
        help="Trait to use for persona generation",
    )
    args = ap.parse_args()
    
    role_judge = LLMJudge(
        backend="openai",
        model="gpt-4.1-mini",
        prompt_template=PROMPT_TEMPLATE,
    )

    for role in args.roles:
        logger.info("=== Processing role: %s ===", role)
        
        for layer in args.layers:
            logger.info("Extracting activations from layer %d", layer)
        
            for sample_count in args.sample_counts:
                
                try:
                    pvx = RoleLayersPersonaModel.load_or_create(
                        target_model_id=args.model,
                        concept=role,
                        layer=layer,
                        target_pairs=sample_count,
                        safetensors_dir=args.safetensor_dir,
                    )
                    logger.info("Successfully loaded/created RoleLayersPersonaModel on layer %d for role '%s'", layer, role)
                except Exception as e:
                    logger.error("Failed to load or create RoleLayersPersonaModel on layer %d for role '%s': %s", layer, role, e)
                    continue
                
                
                if args.question:
                    logger.info("=== Question ===")
                    logger.info(args.question)
                    logger.info("")
                    
                    response = pvx.generate(
                        prompt=args.question,
                        alpha=args.alpha,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    base_response = pvx.generate(
                        prompt=args.question,
                        alpha=0,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                    )
                    logger.info("=== Base Response ===")
                    logger.info(base_response)
                    logger.info("=== Steered Response ===")
                    logger.info(response)
                    
                    score = role_judge(question=args.question, answer=response, role=role)
                    logger.info("=== Judge Score for role %s: %d / 100 ===", role, score)
                
                else:
                    trait_path = Path("./configs/scoring/interest_profiler.json")
                    
                    with trait_path.open() as f:
                        trait_data = json.load(f)
                        
                    testing_traits = [trait for trait in trait_data if trait["dimension"] == args.trait] if args.trait else trait_data

                    base = (
                        "You are answering the O*NET Interest Profiler. "
                        "For this work activity, would you LIKE to do it as part of a job? "
                    )
                    
                    answer_format = ""
                    if not args.cot:
                        answer_format = "Answer ONLY with 'Yes' or 'No' — no explanation."
                    else:
                        answer_format = "Show your reasoning and steps. End with only 'Yes' or 'No'."
                    
                    base += answer_format

                    score = 0

                    for trait in testing_traits:
                        convo = [
                            {"role": "system", "content": base},
                            {"role": "user", "content": f"Activity: {trait['text']}"},
                        ]

                        logger.info("=== Testing trait: %s ===", trait["text"])
                        convo[1]["content"] = f"Activity: {trait['text']}"
                        response = pvx.generate(
                            messages=convo,
                            alpha=args.alpha,
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                        )
                        logger.info("Response: %s", response)
                        
                        if "Yes" in response[-200:]:
                            score += 1
                            
                    logger.info("=== Final %s Score for role %s: %d / %d ===", args.trait, role, score, len(testing_traits))