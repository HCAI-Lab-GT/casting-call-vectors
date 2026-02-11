import argparse
import contextvars
import json
from pathlib import Path
import random
from typing import Optional

from pvx.implementations.roles.role_dataset import RoleDataset
from datetime import datetime
import torch
from tqdm import tqdm
from transformers.utils import logging as transformers_logging
from safetensors.torch import save_file

from pvx import setup_logging
from pvx.abstraction.pvx_models.abstract_persona_model import AbstractPersonaModel
from pvx.implementations.judges.role_judge import RoleJudge

torch.set_float32_matmul_precision("high")

# Disable transformers progress bars to avoid cluttering output
transformers_logging.set_verbosity_error()

# request-local steering state (per concurrent generate call)
_STEER_DELTA = contextvars.ContextVar("steer_delta", default=None)  # Tensor (1,H) or None

logger = setup_logging(name="role-persona-model")


class RolePersonaModel(AbstractPersonaModel):
    
    def __init__(self,
                 role: str, 
                 dataset: Optional[RoleDataset] = None,
                 dataset_dirpath: str = './persona_data/role_datasets/', 
                 *args, 
                 **kwargs):
        kwargs.setdefault("judge_cls", RoleJudge)
        self.role = role
        self.dataset = dataset if dataset else RoleDataset.from_json(role, dirpath=dataset_dirpath)
        
        super().__init__(*args, **kwargs)
        
    @classmethod
    def from_json(
        cls,
        json_filepath: str,
        role: Optional[str] = None,
    ) -> "RolePersonaModel":
        """
        Load a PersonaModel instance from a previously saved JSON file.

        Args:
            json_filepath (str): Path to the JSON file containing the saved initialization data

        Returns:
            PersonaModel: A new instance with the loaded persona vectors
        """

        with open(json_filepath, "r") as f:
            data = json.load(f)

        logger.info("Loading PersonaModel from: %s", json_filepath)

        # Create instance without extracting vectors
        instance = cls(
            role=role,
            target_model_id=data["target_model_id"],
            dataset=None,  # Dataset not needed when loading from JSON
            layer=data["layer_steering"],
            from_json=True,
        )

        # Load the persona vectors directly
        instance.prompt_persona_vector = torch.tensor(data["prompt_persona_vector"])
        instance.response_persona_vector = torch.tensor(data["response_persona_vector"])

        # Store additional metadata
        if "dataset_info" in data and data["dataset_info"]:
            instance.role = data["dataset_info"]["role"]

        logger.info("✅ Loaded PersonaModel from: %s", json_filepath)
        logger.info("   Model: %s", instance.target_model_id)
        logger.info("   Layer: %d", instance.layer_steering)
        logger.info("   Role: %s", instance.role if hasattr(instance, "role") else None)
        logger.info(
            "   Prompt persona vector shape: %s", str(tuple(instance.prompt_persona_vector.shape))
        )
        logger.info(
            "   Response persona vector shape: %s",
            str(tuple(instance.response_persona_vector.shape)),
        )

        return instance

    def save_to_json(self, filepath: str = "./persona_data/model_inits/") -> str:
        """
        Save the persona vectors and initialization config to JSON

        Args:
            filepath (str): Directory path to save the JSON file

        Returns:
            str: Path to the saved JSON file
        """

        filepath += f"{self.dataset.role}_persona_initialization/{self.target_model_id}.json"

        # Convert tensors to lists for JSON serialization
        initialization_data = {
            "target_model_id": self.target_model_id,
            "role": self.dataset.role if self.dataset else None,
            "layer_steering": self.layer_steering,
            "device": self.device,
            "prompt_persona_vector": self.prompt_persona_vector.tolist(),
            "response_persona_vector": self.response_persona_vector.tolist(),
            "all_layers_response_persona_vector": self.all_layers_response_persona_vector.tolist(),
            "prompt_persona_vector_shape": list(self.prompt_persona_vector.shape),
            "response_persona_vector_shape": list(self.response_persona_vector.shape),
            "created_at": datetime.now().isoformat(),
            "dataset_info": {
                "role": self.dataset.role,
                "num_questions": self.dataset.num_questions,
                "num_pos_prompts": len(self.dataset.positive_prompts),
            }
            if self.dataset
            else None,
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        with open(filepath, "w") as f:
            json.dump(initialization_data, f, indent=2)

        logger.info("✅ Initialization saved to: %s", filepath)
        return filepath
    
    @classmethod
    def load_or_create(
        cls,
        target_model_id: str = "qwen2.5:7b-instruct",
        dataset: Optional[RoleDataset] = None,
        role: Optional[str] = None,  # alternate to dataset for loading
        layer: float = 14,
        json_filepath: Optional[str] = None,
        safetensors_dir: str = "./persona_data/model_inits/",
    ) -> "RolePersonaModel":
        """
        Load a PersonaModel instance from saved files if they exist, otherwise create a new one.

        Priority order:
        1. Safetensors file (preferred - smaller, faster)
        2. Legacy JSON file (backward compatibility)
        3. Create new instance

        Args:
            target_model_id: Model identifier
            dataset: PersonaDataset for extraction (only used if creating new)
            role: Role name for loading
            layer: Layer for steering
            json_filepath: Legacy JSON path (optional, for backward compatibility)
            safetensors_dir: Directory containing safetensors files

        Returns:
            PersonaModel: Loaded or newly created instance
        """
        # Build safetensors path
        safe_model_id = target_model_id.replace("/", "__")
        safetensors_path = Path(safetensors_dir) / f"{role}_persona_initialization/{safe_model_id}.safetensors"

        # Try safetensors first (preferred format)
        if safetensors_path.exists():
            try:
                return cls.from_safetensors(str(safetensors_path), trait=role)
            except Exception as e:
                logger.warning("⚠️ Failed to load from safetensors: %s", e)

        # Fall back to legacy JSON
        json_filepath = (
            json_filepath
            or f"./persona_data/model_inits/{role}_persona_initialization/{target_model_id}.json"
        )

        try:
            if Path(json_filepath).exists():
                logger.info("Loading from legacy JSON (consider migrating to safetensors)")
                return cls.from_json(json_filepath, role=role)

        except Exception as e:
            logger.warning(
                "⚠️ Failed to load from JSON: %s. Creating a new PersonaModel instance.", e
            )

        return cls(
            target_model_id=target_model_id,
            role=role,
            dataset=dataset,
            layer=layer,
        )
    
    def save_to_safetensors(self, filepath: str = "./persona_data/model_inits/") -> str:
        """
        Save the persona vectors to safetensors format (much smaller than JSON).

        Metadata (role, model, layer, etc.) is embedded in the safetensors file
        and also appended to a manifest.json for easy querying.

        Args:
            filepath (str): Directory path to save the safetensors file

        Returns:
            str: Path to the saved safetensors file
        """
        # Sanitize model ID for filename (replace / with __)
        safe_model_id = self.target_model_id.replace("/", "__")
        role = self.dataset.role if self.dataset else "unknown"
        filename = f"{self.dataset.role}_persona_initialization/{safe_model_id}.safetensors"
        full_path = Path(filepath) / filename

        # Prepare tensors dict
        tensors = {
            "prompt_persona_vector": self.prompt_persona_vector.contiguous().cpu(),
            "response_persona_vector": self.response_persona_vector.contiguous().cpu(),
            "all_layers_response_persona_vector": self.all_layers_response_persona_vector.contiguous().cpu(),
        }

        # Prepare metadata (safetensors stores metadata as strings)
        metadata = {
            "target_model_id": self.target_model_id,
            "role": role,
            "layer_steering": str(self.layer_steering),
            "created_at": datetime.now().isoformat(),
            "prompt_persona_vector_shape": str(list(self.prompt_persona_vector.shape)),
            "response_persona_vector_shape": str(list(self.response_persona_vector.shape)),
            "all_layers_shape": str(list(self.all_layers_response_persona_vector.shape)),
        }

        if self.dataset:
            metadata["num_questions"] = str(self.dataset.num_questions)
            metadata["num_pos_prompts"] = str(len(self.dataset.positive_prompts))

        # Ensure directory exists
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Save tensors with metadata
        save_file(tensors, str(full_path), metadata=metadata)

        # Update manifest
        self._update_manifest(filepath, filename, metadata)

        logger.info("✅ Saved to safetensors: %s", full_path)
        return str(full_path)

    def _update_manifest(self, dirpath: str, filename: str, metadata: dict) -> None:
        """Update the manifest.json with info about this vector file."""
        manifest_path = Path(dirpath) / "manifest.json"

        # Load existing manifest or create new
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
        else:
            manifest = {"vectors": [], "updated_at": None}

        # Create entry for this file
        entry = {
            "filename": filename,
            "role": metadata.get("role"),
            "model": metadata.get("target_model_id"),
            "layer": int(metadata.get("layer_steering", 14)),
            "created_at": metadata.get("created_at"),
        }

        # Update or append (replace if same filename exists)
        vectors = manifest.get("vectors", [])
        if vectors is None:
            vectors = []
        existing_idx = next((i for i, v in enumerate(vectors) if v["filename"] == filename), None)
        if existing_idx is not None:
            vectors[existing_idx] = entry
        else:
            vectors.append(entry)
        manifest["vectors"] = vectors

        manifest["updated_at"] = datetime.now().isoformat()

        # Save manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def from_safetensors(
        cls,
        safetensors_path: str,
        role: Optional[str] = None,
    ) -> "AbstractPersonaModel":
        """
        Load a PersonaModel instance from a safetensors file.

        Args:
            safetensors_path (str): Path to the .safetensors file
            trait (str, optional): Override trait name (otherwise read from metadata)

        Returns:
            AbstractPersonaModel: A new instance with the loaded persona vectors
        """
        from safetensors import safe_open

        logger.info("Loading PersonaModel from: %s", safetensors_path)

        # Load tensors and metadata
        with safe_open(safetensors_path, framework="pt") as f:
            metadata = f.metadata()
            prompt_vec = f.get_tensor("prompt_persona_vector")
            response_vec = f.get_tensor("response_persona_vector")
            all_layers_vec = f.get_tensor("all_layers_response_persona_vector")

        # Extract metadata
        target_model_id = metadata.get("target_model_id", "unknown")
        layer = int(metadata.get("layer_steering", "14"))
        role = trait or metadata.get("role")

        # Create instance without extracting vectors
        instance = cls(
            role=role,
            target_model_id=target_model_id,
            dataset=None,
            layer=layer,
            from_json=True,  # Reuse flag to skip extraction
        )

        # Load the persona vectors directly
        instance.prompt_persona_vector = prompt_vec
        instance.response_persona_vector = response_vec
        instance.all_layers_response_persona_vector = all_layers_vec

        logger.info("✅ Loaded PersonaModel from safetensors: %s", safetensors_path)
        logger.info("   Model: %s", instance.target_model_id)
        logger.info("   Layer: %d", instance.layer_steering)
        logger.info("   Role: %s", role)
        logger.info("   Prompt persona vector shape: %s", tuple(prompt_vec.shape))
        logger.info("   Response persona vector shape: %s", tuple(response_vec.shape))

        return instance
    
    @torch.inference_mode()
    def extract_persona_vector(
        self, temperature: float = 0.9, max_new_tokens: int = 200
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract persona vectors from the dataset. Main function.

        Args:
            temperature: Sampling temperature for generation.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            Tuple of (prompt_persona_vector, response_persona_vector, response_average_all_layers)
        """

        # Get cartesian product of all positive-negative question pairs from dataset
        prompt_question_pairs = self.dataset.extract_pos_question_pairs()

        # Optimization #4: Pre-tokenize and cache all prompts
        logger.info("Pre-tokenizing %d pairs...", len(prompt_question_pairs))

        token_cache = {}
        for pos, question in prompt_question_pairs:
            # for system_prompt, key_suffix in [(pair.pos, 'pos'), (pair.neg, 'neg')]:
            # Prepare messages in chat format
            messages = [
                {"role": "system", "content": pos},
                {"role": "user", "content": question},
            ]

            enc = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            token_cache[(question, pos)] = (enc, torch.ones_like(enc))

        # Process each pair and accumulate activations (Tensor instead of list for speed)
        # Tensor of last activation of prompt
        sum_prompt_last_pos = None

        # Tensor of average activation of response
        sum_resp_avg_pos = None

        # Tensor of average activation of response across all layers
        sum_resp_avg_all_layers_pos = None

        # track number of pairs
        n = 0
        passed_pairs = set()
        retries = 0
        max_retries = 2

        with tqdm(
            total=self.target_pairs,
            desc=f"Extracting activations for role {self.dataset.role}",
        ) as pbar:
            while n < self.target_pairs and retries < max_retries:
                remaining = [p for p in prompt_question_pairs if p not in passed_pairs]
                epoch_pairs = random.sample(remaining, len(remaining))
                made_progress = False

                for pos, question in epoch_pairs:
                    pos_ids, pos_mask = token_cache[(question, pos)]

                    # pl: prompt hidden layer last activation
                    # ra: response hidden layer avg activation
                    # rall: response avg all activation
                    pl_pos, ra_pos, rall_pos, pos_response = self._get_activations(
                        pos_ids, pos_mask, temperature, max_new_tokens
                    )

                    pos_score = self.judge(question=question, answer=pos_response)
                    

                    if pos_score != 3:
                        logger.info("===FAILED CASE===")
                        logger.info("Question: " + question + ", System Prompt: " + pos + ", Score: " + str(pos_score))
                        continue

                    logger.info("===PASSED CASE===")
                    logger.info("Question: " + question + ", System Prompt: " + pos + ", Score: " + str(pos_score))
                    passed_pairs.add((pos, question))
                    pbar.update(1)
                    made_progress = True

                    if sum_prompt_last_pos is None:
                        sum_prompt_last_pos = torch.zeros_like(pl_pos, dtype=torch.float32)
                        sum_resp_avg_pos = torch.zeros_like(ra_pos, dtype=torch.float32)
                        sum_resp_avg_all_layers_pos = torch.zeros_like(
                            rall_pos, dtype=torch.float32
                        )

                    sum_prompt_last_pos += pl_pos.float()
                    sum_resp_avg_pos += ra_pos.float()
                    sum_resp_avg_all_layers_pos += rall_pos.float()
                    n += 1

                    if n >= self.target_pairs:
                        break

                retries += 1
                if not made_progress:
                    break

        prompt_last_pos_mean = sum_prompt_last_pos / max(n, 1)
        response_avg_pos_mean = sum_resp_avg_pos / max(n, 1)

        all_layers_response_avg_pos_mean = sum_resp_avg_all_layers_pos / max(n, 1)

        prompt_persona_vector = prompt_last_pos_mean  # (1, hidden)
        response_persona_vector = response_avg_pos_mean  # (1, hidden)
        all_layers_response_persona_vector = (
            all_layers_response_avg_pos_mean
        )
        # (num_states, 1, hidden)

        self.prompt_persona_vector = prompt_persona_vector.cpu()
        self.response_persona_vector = response_persona_vector.cpu()
        self.all_layers_response_persona_vector = all_layers_response_persona_vector.cpu()

        logger.info("Extracted persona vectors from %d pairs", len(prompt_question_pairs))
        logger.info("Prompt persona vector shape: %s", str(tuple(prompt_persona_vector.shape)))
        logger.info("Response persona vector shape: %s", str(tuple(response_persona_vector.shape)))
        logger.info(
            "All-layers response persona vector shape: %s",
            str(tuple(all_layers_response_persona_vector.shape)),
        )
        if n < self.target_pairs:
            logger.warning("Only %d pairs passed after %d retries.", len(passed_pairs), max_retries)

        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model typically",
    )
    ap.add_argument(
        "-r", "--role", type=str, default="lawyer", help="Role of the persona dataset"
    )
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument(
        "-a", "--alpha", type=float, default=0.3, help="Alpha value for persona steering"
    )
    ap.add_argument("--temperature", type=float, default=0.1, help="Temperature for sampling")
    ap.add_argument(
        "-q",
        "--question",
        type=str,
        default="Is it ever acceptable to break the rules?",
        help="Question to generate response for",
    )
    ap.add_argument(
        "-f",
        "--json_filepath",
        type=str,
        default=None,
        help="Filepath to model init file (not role dataset)",
    )
    args = ap.parse_args()

    pvx = RolePersonaModel.load_or_create(
        target_model_id=args.model_name,
        role=args.role,
        layer=14,
        json_filepath=args.json_filepath
    )

    response = pvx.generate(
        prompt=args.question,
        alpha=0,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    steer_response = pvx.generate(
        prompt=args.question,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    logger.info("=== Question ===")
    logger.info(args.question)
    logger.info("")
    logger.info("=== Non-Steered Answer ===")
    logger.info(response)
    logger.info("")
    logger.info("=== %s Steered Answer ===", args.role.upper())
    logger.info(steer_response)
