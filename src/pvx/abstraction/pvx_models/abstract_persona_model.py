import contextvars
import json
import os
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from pvx import Heartbeat, setup_logging
from pvx.implementations.judges.llm_as_judge import LLMJudge
from pvx.abstraction.pvx_models.abstract_dataset import AbstractDataset
from pvx.utils.judge_utils import JudgeConfig

torch.set_float32_matmul_precision("high")

# Disable transformers progress bars to avoid cluttering output
transformers_logging.set_verbosity_error()

# request-local steering state (per concurrent generate call)
_STEER_DELTA = contextvars.ContextVar("steer_delta", default=None)  # Tensor (1,H) or None

logger = setup_logging(name="abstract-persona-model")


class AbstractPersonaModel(ABC):
    """
    Model wrapper that extracts and utilizes persona vectors for steering model behavior.
    """

    def __init__(
        self,
        concept: str = None,
        dataset: Optional[AbstractDataset] = None,
        dataset_dirpath: str = None,
        target_model_id: str = "qwen2.5:7b-instruct",
        layer: float = 14,
        default_alpha: float = 3.0,
        judge_config: Optional[JudgeConfig] = None,
        judge_cls=LLMJudge,
        judge_threshold: float = 50.0,
        target_pairs: int = 20,
        from_json: bool = False,
        safetensors_dir = None,
        json_filepath = None
    ):
        """
        Initialize the PersonaModel with a target model and dataset for persona extraction.

        Args:
            target_model_id (str): Model identifier (HuggingFace or OpenAI).
            dataset (PersonaDataset | None): Dataset for persona extraction.
            layer (int): Layer index for extracting hidden activations.
            from_json (bool): Whether to load persona vectors from JSON file.
        """

        self._init_base(target_model_id, layer, default_alpha)
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "<|im_start|>{{ message['role'] }}\n"
                "{{ message['content'] }}"
                "<|im_end|>\n"
                "{% endfor %}"
                f"<|im_start|>\n"
            )
        
        # avoids linter error by ensuring self.dataset and self.concept are always set, even if not used when loading from JSON
        self.concept = concept if (not hasattr(self, 'concept') or concept) else self.concept
        self.dataset = dataset if (not hasattr(self, 'dataset') or dataset) else self.dataset
        
        # same for savetensors and json_filepath
        self.safetensors_dir = self.safetensors_dir if hasattr(self, 'safetensors_dir') else safetensors_dir
        self.json_filepath = self.json_filepath if hasattr(self, 'json_filepath') else json_filepath
        
        # skip extraction if not loading from JSON
        if from_json:
            return

        self.judge_threshold = judge_threshold
        
        if judge_config is None:
            judge_config = JudgeConfig(prompt_template=self.dataset.evaluation_prompt)
        else:
            if judge_config.prompt_template is None:
                judge_config.prompt_template = self.dataset.evaluation_prompt
        self.judge = judge_cls(**judge_config.to_kwargs())

        self.target_pairs = target_pairs

        # Extract persona vectors
        _, _, _ = self.extract_persona_vector()

        # Save initialization (with extracted persona vector) to JSON
        self.save_to_json(filepath=json_filepath)
        self.save_to_safetensors(filepath=safetensors_dir)

    @classmethod
    def base_model(
        cls,
        target_model_id: str = "qwen2.5:7b-instruct",
        layer: float = 14,
        default_alpha: float = 3.0,
    ):
        """
        Create a trait-agnostic PersonaModel instance with only the base model and tokenizer loaded.

        This class method initializes a PersonaModel without loading or generating any persona vectors
        or trait-specific data. It is useful for scenarios where persona steering will be applied later
        or dynamically at inference time.

        Args:
            target_model_id (str): Model identifier (HuggingFace or OpenAI).
            layer (float): Layer index for extracting hidden activations.
            default_alpha (float): Default alpha value for persona steering.

        Returns:
            PersonaModel: An instance of PersonaModel with only the base model and tokenizer loaded.

        Example:
            >>> model = PersonaModel.base_model(target_model_id="qwen2.5:7b-instruct", layer=14)
        """

        instance = cls.__new__(cls)
        instance._init_base(
            target_model_id=target_model_id, layer=layer, default_alpha=default_alpha
        )
        return instance

    def _init_base(self, target_model_id, layer, default_alpha):
        """
        Initialize the base model, tokenizer, and device configuration.

        This internal method is used by both the main constructor and the base_model classmethod
        to set up the core model, tokenizer, device, and related attributes. It does not load or
        generate any persona vectors or trait-specific data.

        Args:
            target_model_id (str): Model identifier (HuggingFace or OpenAI).
            layer (float): Layer index for extracting hidden activations.
            default_alpha (float): Default alpha value for persona steering.

        Returns:
            None

        Example:
            >>> self._init_base(target_model_id="qwen2.5:7b-instruct", layer=14, default_alpha=3.0)
        """

        self.target_model_id = target_model_id
        self.layer_steering = layer
        self.default_alpha = default_alpha

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_id, trust_remote_code=True)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        # Set EOD token id
        self.eos_token_ids = (
            [self.tokenizer.eos_token_id]
            if isinstance(self.tokenizer.eos_token_id, int)
            else self.tokenizer.eos_token_id
        )

        # Optimization #7: Use torch.compile if available (PyTorch 2.0+)
        # Note: Disabled by default due to compatibility issues with transformers + CUDA graphs
        # To enable, set environment variable: ENABLE_TORCH_COMPILE=1
        if hasattr(torch, "compile") and os.environ.get("ENABLE_TORCH_COMPILE", "0") == "1":
            try:
                # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
                self.model = torch.compile(self.model, mode="default", dynamic=True)
                logger.info("✅ Model compiled with torch.compile for faster inference")
            except Exception as e:
                logger.error("⚠️ torch.compile failed: %s", str(e))
                logger.error("   Continuing without compilation...")

        # persistent steering hook + cached base vector (on steered block's device/dtype)
        self._steer_hook_handle = None
        self._steer_block = None
        self._persona_base = None  # Tensor (1, H) on block device/dtype
        self._persona_base_key = None  # (device, dtype)
        self._persona_base_lock = threading.Lock()

        # install hook once
        self._install_steer_hook(layer_idx=self.layer_steering)
        
    def initialize_activations(self):
        sum_prompt_last = None
        sum_resp_avg = None
        sum_all_layers = None
        
        return sum_prompt_last, sum_resp_avg, sum_all_layers
    
    def aggregate_activations(self, sum_prompt, sum_resp, sum_all_layers, pl, ra, rall):
        """
        Aggregate activations for a single category (pos or neg).
        Initializes sums if None, otherwise accumulates.
        Returns updated sums.
        """
        if sum_prompt is None:
            sum_prompt = torch.zeros_like(pl, dtype=torch.float32)
            sum_resp = torch.zeros_like(ra, dtype=torch.float32)
            sum_all_layers = torch.zeros_like(rall, dtype=torch.float32)
        sum_prompt += pl.float()
        sum_resp += ra.float()
        sum_all_layers += rall.float()
        return sum_prompt, sum_resp, sum_all_layers
    
    def finalize_activations(self, sum_prompt, sum_resp, sum_all_layers, n):
        """
        Compute means from sums and count.
        """
        mean_prompt = sum_prompt / max(n, 1)
        mean_resp = sum_resp / max(n, 1)
        mean_all_layers = sum_all_layers / max(n, 1)
        return mean_prompt, mean_resp, mean_all_layers
    
    def compute_contrastive_persona_vector_individual(self, pos_mean, neg_mean):
        return pos_mean - neg_mean
    
    def compute_all_persona_vectors(self, prompt_last_pos_mean, 
                                    prompt_last_neg_mean, 
                                    response_avg_pos_mean, 
                                    response_avg_neg_mean, 
                                    response_all_pos_mean, 
                                    response_all_neg_mean):
        prompt_last_persona_vector = self.compute_contrastive_persona_vector_individual(prompt_last_pos_mean, prompt_last_neg_mean)
        response_avg_persona_vector = self.compute_contrastive_persona_vector_individual(response_avg_pos_mean, response_avg_neg_mean)
        response_all_persona_vector = self.compute_contrastive_persona_vector_individual(response_all_pos_mean, response_all_neg_mean)
        
        return prompt_last_persona_vector, response_avg_persona_vector, response_all_persona_vector
        
    def compute_contrastive_persona_vectors(self, 
                                            sum_prompt_pos, 
                                            sum_resp_pos, 
                                            sum_all_layers_pos, 
                                            sum_prompt_neg, 
                                            sum_resp_neg, 
                                            sum_all_layers_neg,
                                            n):
        prompt_last_pos_mean, response_avg_pos_mean, all_layers_response_avg_pos_mean = self.finalize_activations(
            sum_prompt_pos, 
            sum_resp_pos, 
            sum_all_layers_pos, 
            n
        )
        
        prompt_last_neg_mean, response_avg_neg_mean, all_layers_response_avg_neg_mean = self.finalize_activations(
            sum_prompt_neg, 
            sum_resp_neg, 
            sum_all_layers_neg, 
            n
        )
        
        prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector = self.compute_all_persona_vectors(
            prompt_last_pos_mean=prompt_last_pos_mean,
            prompt_last_neg_mean=prompt_last_neg_mean,
            response_avg_pos_mean=response_avg_pos_mean,
            response_avg_neg_mean=response_avg_neg_mean,
            response_all_pos_mean=all_layers_response_avg_pos_mean,
            response_all_neg_mean=all_layers_response_avg_neg_mean
        )
        
        self.prompt_persona_vector = prompt_persona_vector.cpu()
        self.response_persona_vector = response_persona_vector.cpu()
        self.all_layers_response_persona_vector = all_layers_response_persona_vector.cpu()
        
        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector

    @classmethod
    def from_json(
        cls,
        json_filepath: str,
        concept: Optional[str] = None,
    ) -> "AbstractPersonaModel":
        with open(json_filepath, "r") as f:
            data = json.load(f)

        logger.info("Loading PersonaModel from: %s", json_filepath)

        # Create instance without extracting vectors
        instance = cls(
            concept,
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
            instance.concept = data["dataset_info"]["concept"]

        logger.info("✅ Loaded PersonaModel from: %s", json_filepath)
        logger.info("   Model: %s", instance.target_model_id)
        logger.info("   Layer: %d", instance.layer_steering)
        logger.info("   Concept: %s", instance.concept)
        logger.info(
            "   Prompt persona vector shape: %s", str(tuple(instance.prompt_persona_vector.shape))
        )
        logger.info(
            "   Response persona vector shape: %s",
            str(tuple(instance.response_persona_vector.shape)),
        )

        return instance

    @classmethod
    def get_path(cls, target_model_id: str, concept: str, safetensors_dir: str, use_json=False, kwargs=None, obj=None):
        safe_model_id = target_model_id.replace("/", "__")
        filepath = f"{concept}_persona_initialization/{safe_model_id}.{'safetensors' if not use_json else 'json'}"
        safetensors_path = Path(safetensors_dir) / filepath
        
        return safetensors_path, filepath

    @classmethod
    def load_or_create(
        cls,
        target_model_id: str = "allenai/Olmo-3-7B-Instruct",
        dataset: Optional[AbstractDataset] = None,
        concept: Optional[str] = None,  # alternate to dataset for loading (trait or role)
        layer: int = 14,
        target_pairs: int = 40,
        json_filepath: Optional[str] = None,
        only_load: bool = False,
        safetensors_dir: str = "./persona_data/model_inits/",
        **kwargs
    ) -> "AbstractPersonaModel":
        logger.info("Attempting to load PersonaModel for concept '%s' and model '%s'", concept, target_model_id)
        safetensors_path, _ = cls.get_path(target_model_id, concept, safetensors_dir, use_json=False, kwargs=locals())
        
        logger.info(f"Safetensors Path: {safetensors_path}")

        # Try safetensors first (preferred format)
        if safetensors_path.exists():
            try:
                instance = cls.from_safetensors(str(safetensors_path), concept=concept)
                instance.safetensors_dir = safetensors_dir
                instance.target_pairs = target_pairs
                return instance
            except Exception as e:
                logger.warning("⚠️ Failed to load from safetensors: %s", e)

        # Fall back to legacy JSON
        json_filepath = (
            json_filepath
            or cls.get_path(target_model_id, concept, safetensors_dir, use_json=True)[0]
        )
        
        logger.info(f"JSON Path: {safetensors_path}")

        try:
            if Path(json_filepath).exists():
                logger.info("Loading from legacy JSON (consider migrating to safetensors)")
                instance = cls.from_json(json_filepath, concept=concept)
                instance.json_filepath = json_filepath
                instance.target_pairs = target_pairs
                return instance

        except Exception as e:
            logger.warning(
                "⚠️ Failed to load from JSON: %s.", e
            )
        
        if only_load:
            raise FileNotFoundError("No instance found and load only selected.")
        
        logger.info("Creating a new PersonaModel instance.")

        return cls(
            concept,
            target_model_id=target_model_id,
            dataset=dataset,
            layer=layer,
            target_pairs=target_pairs,
            safetensors_dir=safetensors_dir,
            json_filepath=json_filepath
        )
    
    def is_concept_role(self):
        return hasattr(self, "role")

    def save_to_json(self, filepath: str = "./persona_data/model_inits/", **args) -> str:
        """
        Save the persona vectors and initialization config to JSON

        Args:
            filepath (str): Directory path to save the JSON file

        Returns:
            str: Path to the saved JSON file
        """
        concept = self.dataset.concept if self.dataset else "unknown"
        full_path, filename = self.get_path(self.target_model_id, concept, filepath, use_json=True, obj=self)
        
        dataset_metadata_key = "num_pos_prompts" if self.is_concept_role() else "num_pos_neg_pairs"

        # Convert tensors to lists for JSON serialization
        initialization_data = {
            "target_model_id": self.target_model_id,
            "concept": self.dataset.concept if self.dataset else None,
            "layer_steering": self.layer_steering,
            "device": self.device,
            "prompt_persona_vector": self.prompt_persona_vector.tolist(),
            "response_persona_vector": self.response_persona_vector.tolist(),
            "all_layers_response_persona_vector": self.all_layers_response_persona_vector.tolist(),
            "prompt_persona_vector_shape": list(self.prompt_persona_vector.shape),
            "response_persona_vector_shape": list(self.response_persona_vector.shape),
            "created_at": datetime.now().isoformat(),
            "dataset_info": {
                "concept": self.dataset.concept,
                "num_questions": self.dataset.num_questions,
                dataset_metadata_key: len(self.dataset.positive_prompts) if self.is_concept_role() else len(self.dataset.positive_negative_pairs),
            }
            if self.dataset
            else None,
        }

        # Ensure directory exists
        Path(full_path).parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        with open(full_path, "w") as f:
            json.dump(initialization_data, f, indent=2)

        logger.info("✅ Initialization saved to: %s", filepath)
        return filepath

    def save_to_safetensors(self, filepath: str = "./persona_data/model_inits/", **args) -> str:
        """
        Save the persona vectors to safetensors format (much smaller than JSON).

        Metadata (trait, model, layer, etc.) is embedded in the safetensors file
        and also appended to a manifest.json for easy querying.

        Args:
            filepath (str): Directory path to save the safetensors file

        Returns:
            str: Path to the saved safetensors file
        """
        # Sanitize model ID for filename (replace / with __)
        concept = self.dataset.concept if self.dataset else "unknown"
        full_path, filename = self.get_path(self.target_model_id, concept, filepath, use_json=False, obj=self)

        # Prepare tensors dict
        tensors = {
            "prompt_persona_vector": self.prompt_persona_vector.contiguous().cpu(),
            "response_persona_vector": self.response_persona_vector.contiguous().cpu(),
            "all_layers_response_persona_vector": self.all_layers_response_persona_vector.contiguous().cpu(),
        }

        # Prepare metadata (safetensors stores metadata as strings)
        metadata = {
            "target_model_id": self.target_model_id,
            "concept": concept,
            "layer_steering": str(self.layer_steering),
            "created_at": datetime.now().isoformat(),
            "prompt_persona_vector_shape": str(list(self.prompt_persona_vector.shape)),
            "response_persona_vector_shape": str(list(self.response_persona_vector.shape)),
            "all_layers_shape": str(list(self.all_layers_response_persona_vector.shape)),
        }

        if self.dataset:
            metadata["num_questions"] = str(self.dataset.num_questions)
            if self.is_concept_role():
                metadata["num_pos_prompts"] = str(len(self.dataset.positive_prompts))
            else:  
                metadata["num_pos_neg_pairs"] = str(len(self.dataset.positive_negative_pairs))

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
            "concept": metadata.get("concept"),
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
        concept: Optional[str] = None,
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
        concept = concept or metadata.get("concept")

        # Create instance without extracting vectors
        instance = cls(
            concept,
            target_model_id=target_model_id,
            dataset=None,
            layer=layer,
            from_json=True,  # Reuse flag to skip extraction
        )

        # Load the persona vectors directly
        instance.prompt_persona_vector = prompt_vec
        instance.response_persona_vector = response_vec
        instance.all_layers_response_persona_vector = all_layers_vec
        
        logger.info(instance.concept)

        logger.info("✅ Loaded PersonaModel from safetensors: %s", safetensors_path)
        logger.info("   Model: %s", instance.target_model_id)
        logger.info("   Layer: %d", instance.layer_steering)
        logger.info("   Concept: %s", concept)
        logger.info("   Prompt persona vector shape: %s", tuple(prompt_vec.shape))
        logger.info("   Response persona vector shape: %s", tuple(response_vec.shape))

        return instance


    @abstractmethod
    def extract_persona_vector(
        self, temperature: float = 0.2, max_new_tokens: int = 200
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    @torch.inference_mode()
    def generate(
        self,
        prompt: str | None = None,
        messages: list[dict[str, str]] | None = None,
        alpha: float | None = 3,
        max_new_tokens: int = 2000,
        temperature: float = 0.2,
        top_p=0.99,
    ) -> str:
        """
        Generate a response to the prompt with persona steering.

        Args:
            prompt: The user prompt string.
            messages: Optional list of messages in chat format.
            alpha: Steering strength for persona vector.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.

        Returns:
            str: persona influenced response
        """

        if prompt is None and messages is None:
            raise ValueError("Prompt and Messages cannot both be None")

        self.model.eval()

        if messages is None:
            # Prepare messages in chat format
            assert prompt is not None  # Guaranteed by check above
            messages = [{"role": "user", "content": prompt}]

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize and move to device
        enc = self.tokenizer(formatted, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask", None)

        eos_ids = (
            set(self.eos_token_ids)
            if isinstance(self.eos_token_ids, (list, tuple, set))
            else {int(self.eos_token_ids)}
        )

        past = None
        gen_ids: list[int] = []

        # steering is now per-request via ContextVar; no per-call hook install/remove
        with self._steering_delta(alpha):
            # Prefill on full prompt
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
            )
            past = out.past_key_values

            # Sample first generated token from prefill logits
            logits = out.logits[0, -1, :]
            tok = self._top_p_sample(logits, temperature=temperature, top_p=top_p)
            if tok in eos_ids:
                return ""

            # accumulate first token
            gen_ids.append(tok)
            last_token = torch.tensor([[tok]], device=self.device, dtype=torch.long)

            with Heartbeat(logger, "generating output...", interval=30):
                # Decode loop (KV-cache)
                for _ in range(max_new_tokens - 1):
                    out = self.model(
                        input_ids=last_token,
                        past_key_values=past,
                        use_cache=True,
                    )
                    past = out.past_key_values  # update kv cache

                    # Sample index of next token
                    logits = out.logits[0, -1, :]
                    tok = self._top_p_sample(logits, temperature=temperature, top_p=top_p)
                    if tok in eos_ids:
                        break

                    gen_ids.append(tok)
                    last_token.fill_(tok)

        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    @torch.inference_mode()
    def _get_activations(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        temperature: float = 0.2,
        max_new_tokens: int = 20000,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        """
        Generate response and extract prompt last activation, response average activation, and response average activations all layers.

        Args:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor | None): Optional attention mask tensor.
            temperature (float): Sampling temperature.

        Returns:
            Tuple :
                - Tensor: prompt_last_activation
                - Tensor: response_average
                - Tensor: response_average_all_layers
                - str: response_text
        """
        # prompt forward (KV cache)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )

        # Store cache
        past = out.past_key_values
        prompt_last = out.hidden_states[self.layer_steering + 1][
            :, -1, :
        ]  # (1, hidden), 0th layer init embedding

        # generate first token
        logits0 = out.logits[0, -1, :]
        tok = self._top_p_sample(logits0, top_p=0.99, temperature=temperature)

        # early exit if no response tokens
        if tok in self.eos_token_ids:
            # no response tokens
            num_states = len(out.hidden_states)
            resp_avg = torch.zeros_like(prompt_last)
            resp_avg_all_layers = torch.zeros(
                (num_states,) + prompt_last.shape,
                device=prompt_last.device,
                dtype=prompt_last.dtype,
            )
            return prompt_last, resp_avg, resp_avg_all_layers, ""

        # accumulate response activations
        acc_dtype = (
            torch.float32
            if prompt_last.dtype in (torch.float16, torch.bfloat16)
            else prompt_last.dtype
        )
        resp_sum = torch.zeros_like(prompt_last, dtype=acc_dtype)

        num_states = len(out.hidden_states)
        resp_sum_all_layers = torch.zeros(
            (num_states,) + prompt_last.shape,
            device=prompt_last.device,
            dtype=acc_dtype,
        )

        # count generated token to calculate average
        count = 0

        # (1,1) tensor reused
        last_token = torch.tensor([[tok]], device=input_ids.device, dtype=torch.long)

        gen_ids = []

        # response generation loop
        for _ in range(max_new_tokens):
            # only pass last token (KV cache)
            out = self.model(
                input_ids=last_token,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = out.past_key_values  # update kv cache

            # add final state of target layer
            final_act = out.hidden_states[self.layer_steering + 1][:, -1, :]  # (1, hidden)
            resp_sum += final_act.to(acc_dtype)

            # add final state of all layers
            all_layers_last_token = torch.stack([hs[:, -1, :] for hs in out.hidden_states], dim=0)
            resp_sum_all_layers += all_layers_last_token.to(acc_dtype)

            # increment count for average calc later
            count += 1

            # Sample index of next token
            logits = out.logits[0, -1, :]
            next_token_id = self._top_p_sample(logits, top_p=0.99, temperature=temperature)  # int

            if int(next_token_id) in self.eos_token_ids:
                break
            gen_ids.append(int(next_token_id))
            last_token.fill_(int(next_token_id))  # reuse (1,1) buffer

        if count > 0:
            resp_avg = (resp_sum / count).to(prompt_last.dtype)
            resp_avg_all_layers = (resp_sum_all_layers / count).to(prompt_last.dtype)
        else:
            resp_avg = torch.zeros_like(prompt_last)
            resp_avg_all_layers = torch.zeros_like(resp_sum_all_layers, dtype=prompt_last.dtype)

        response_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return prompt_last, resp_avg, resp_avg_all_layers, response_text

    def _install_steer_hook(self, layer_idx: int) -> None:
        """
        Installs one persistent hook that reads per-request delta.

        Args:
            layer_idx (int): layer to add steering hook to
        """

        # Get the decoder blocks and specific target steer block
        blocks = self._get_decoder_blocks(self.model)
        block = blocks[layer_idx]
        self._steer_block = block

        def hook(_module, _inp, out):
            """
            Hook function to add delta to the last-token hidden state at the output of the specified decoder block.
            The delta is added to the last token's hidden state.
            register_forward_hook is called on each forward pass of the block

            Args:
                _module (torch.nn.Module): The module instance.
                _inp (tuple): Input tensors to the module.
                out (torch.Tensor | tuple): Output tensor(s) from the module.

            Returns:
                torch.Tensor | tuple: Modified output tensor(s).
            """

            d = _STEER_DELTA.get()  # (1, H) or None
            if d is None:
                return out

            # out is usually Tensor [B,T,H], sometimes tuple(Tensor, ...)
            if isinstance(out, tuple):
                hs = out[0]
                hs[:, -1, :] += d  # in-place add on last position
                return (hs,) + out[1:]

            out[:, -1, :] += d
            return out

        # register once
        self._steer_hook_handle = block.register_forward_hook(hook)

    def _get_persona_base(self) -> torch.Tensor:
        """
        Cached base persona vector on the steered block device/dtype

        Returns:
            torch.Tensor: persona vector
        """

        if not hasattr(self, "prompt_persona_vector"):
            raise RuntimeError("prompt_persona_vector not initialized")

        # pick device/dtype from the steered block (works with device_map sharding)
        try:
            p = next(self._steer_block.parameters())
        except StopIteration:
            p = next(self.model.parameters())

        key = (p.device, p.dtype)
        if self._persona_base is not None and self._persona_base_key == key:
            return self._persona_base

        with self._persona_base_lock:
            if self._persona_base is None or self._persona_base_key != key:
                self._persona_base = self.response_persona_vector.to(
                    device=p.device, dtype=p.dtype
                ).view(1, -1)  # (1, H)
                self._persona_base_key = key
        return self._persona_base

    @contextmanager
    def _steering_delta(self, alpha: float):
        """
        Per-request steering context (stores delta in ContextVar)

        Args:
            alpha (float): alpha applied on runtime
        """

        if alpha == 0:
            yield
            return

        base = self._get_persona_base()  # (1, H)
        delta = alpha * base  # (1, H)
        tok = _STEER_DELTA.set(delta)

        try:
            yield
        finally:
            _STEER_DELTA.reset(tok)

    def close(self):
        """
        Removes hooks. Typically unused
        """

        if self._steer_hook_handle is not None:
            self._steer_hook_handle.remove()
            self._steer_hook_handle = None

    def _get_decoder_blocks(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """
        Retrieve the list of decoder blocks from the model.

        Args:
            model (torch.nn.Module): The model instance.

        Returns:
            list[torch.nn.Module]: List of decoder blocks.
        """

        # Common HF layouts
        if hasattr(model, "model") and hasattr(model.model, "layers"):  # LLaMA/Qwen/Mistral
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2
            return model.transformer.h
        raise RuntimeError("Unsupported model layout: cannot locate decoder blocks.")

    def _top_p_sample(
        self, logits: torch.Tensor, top_p: float = 0.9, temperature: float = 0.2
    ) -> int:
        """
        Perform nucleus (top-p) sampling from logits.

        Args:
            logits (torch.Tensor): Tensor of shape (vocab_size,) representing model logits.
            top_p (float): Cumulative probability threshold for nucleus sampling.
            temperature (float): Sampling temperature.

        Returns:
            int: Sampled token index.
        """

        # Calculate probabilities from logits
        logits = logits / temperature
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Find the cutoff index - keep tokens until cumulative probability exceeds top_p
        # Shift cumulative_probs by 1 to include at least the first token
        cutoff_mask = cumulative_probs - sorted_probs <= top_p

        # Ensure at least one token is included
        cutoff_mask[0] = True

        # Get nucleus probabilities and renormalize
        nucleus_probs = sorted_probs.clone()
        nucleus_probs[~cutoff_mask] = 0.0
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        # Sample from the nucleus
        sampled_index = torch.multinomial(nucleus_probs, num_samples=1)

        # Map back to the original indices
        original_index = sorted_indices.gather(-1, sampled_index)

        return original_index.item()  # Convert tensor to integer
