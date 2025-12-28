import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading
import contextvars
from contextlib import contextmanager, nullcontext
import argparse
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as transformers_logging

from pvx import setup_logging, Heartbeat
from pvx.pvx_models.persona_dataset import PersonaDataset

torch.set_float32_matmul_precision('high')

# Disable transformers progress bars to avoid cluttering output
transformers_logging.set_verbosity_error()

# request-local steering state (per concurrent generate call)
_STEER_DELTA = contextvars.ContextVar("steer_delta", default=None)  # Tensor (1,H) or None

logger = setup_logging(name="persona-model")

class PersonaModel:
    '''
    Model wrapper that extracts and utilizes persona vectors for steering model behavior.
    '''
    def __init__(self,
                 target_model_id: str = "qwen2.5:7b-instruct",
                 dataset: Optional[PersonaDataset] = None,
                 trait: str = None,
                 layer: float = 14,
                 default_alpha: float = 3.0,
                 from_json: bool = False):
        '''
        Initialize the PersonaModel with a target model and dataset for persona extraction.
        
        Args:
            target_model_id (str): Model identifier (HuggingFace or OpenAI).
            dataset (PersonaDataset | None): Dataset for persona extraction.
            layer (int): Layer index for extracting hidden activations.
            from_json (bool): Whether to load persona vectors from JSON file.
        '''
        
        self._init_base(target_model_id, layer, default_alpha)
        
        # skip extraction if not loading from JSON
        if from_json:
            return
        
        # load dataset from file if provided
        self.dataset = dataset if dataset else PersonaDataset.from_json(trait)
        self.trait = self.dataset.trait
        
        # Extract persona vectors
        _, _, _ = self.extract_persona_vector()
        
        # Save initialization (with extracted persona vector) to JSON
        self.save_to_json()

        
    @classmethod
    def base_model(cls,
                       target_model_id: str = "qwen2.5:7b-instruct",
                       layer: float = 14,
                       default_alpha: float = 3.0):

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
        instance._init_base(target_model_id=target_model_id, layer=layer, default_alpha=default_alpha)
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
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)

        # Set EOD token id
        self.eos_token_ids = [self.tokenizer.eos_token_id] if isinstance(self.tokenizer.eos_token_id, int) else self.tokenizer.eos_token_id

        # Optimization #7: Use torch.compile if available (PyTorch 2.0+)
        # Note: Disabled by default due to compatibility issues with transformers + CUDA graphs
        # To enable, set environment variable: ENABLE_TORCH_COMPILE=1
        if hasattr(torch, 'compile') and os.environ.get('ENABLE_TORCH_COMPILE', '0') == '1':
            try:
                # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
                self.model = torch.compile(self.model, mode='default', dynamic=True)
                logger.info("✅ Model compiled with torch.compile for faster inference")
            except Exception as e:
                logger.error(f"⚠️ torch.compile failed: %s", str(e))
                logger.error("   Continuing without compilation...")
        
        # # Optimization #2: Cache for persona vector reshape
        # self._persona_reshaped_cache = None
        
        # persistent steering hook + cached base vector (on steered block's device/dtype)
        self._steer_hook_handle = None
        self._steer_block = None
        self._persona_base = None          # Tensor (1, H) on block device/dtype
        self._persona_base_key = None      # (device, dtype)
        self._persona_base_lock = threading.Lock()

        # install hook once
        self._install_steer_hook(layer_idx=self.layer_steering)
        
    @classmethod
    def from_json(cls, json_filepath: str) -> 'PersonaModel':
        """
        Load a PersonaModel instance from a previously saved JSON file.
        
        Args:
            json_filepath (str): Path to the JSON file containing the saved initialization data
            
        Returns:
            PersonaModel: A new instance with the loaded persona vectors
        """
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        
        logger.info("Loading PersonaModel from: %s", json_filepath)
        
        # Create instance without extracting vectors
        instance = cls(
            target_model_id=data["target_model_id"],
            dataset=None,  # Dataset not needed when loading from JSON
            layer=data["layer_steering"],
            from_json=True
        )

        # Load the persona vectors directly
        instance.prompt_persona_vector = torch.tensor(data["prompt_persona_vector"])
        instance.response_persona_vector = torch.tensor(data["response_persona_vector"])

        # # Optimization #2: Initialize cache for persona vector reshape
        # instance._persona_reshaped_cache = None

        # Store additional metadata
        if "dataset_info" in data and data["dataset_info"]:
            instance.trait = data["dataset_info"]["trait"]

        logger.info("✅ Loaded PersonaModel from: %s", json_filepath)
        logger.info("   Model: %s", instance.target_model_id)
        logger.info("   Layer: %d", instance.layer_steering)
        logger.info("   Trait: %s", instance.trait if hasattr(instance, 'trait') else None)
        logger.info("   Prompt persona vector shape: %s", str(tuple(instance.prompt_persona_vector.shape)))
        logger.info("   Response persona vector shape: %s", str(tuple(instance.response_persona_vector.shape)))

        return instance

    @classmethod
    def load_or_create(cls,
                       target_model_id: str = "qwen2.5:7b-instruct",
                       dataset: Optional[PersonaDataset] = None,
                       trait: str = None, # alternate to dataset for loading
                       layer: float = 14,
                       json_filepath: str = None) -> 'PersonaModel':
        """
        Load a PersonaModel instance from a JSON file if it exists, otherwise create a new one.
        
        Args:
            json_filepath (str): Path to the JSON file containing the saved initialization data
            
        Returns:
            PersonaModel: A new instance with the loaded persona vectors or a newly created one
        """
        print(target_model_id)
        json_filepath = json_filepath if json_filepath else f"./persona_data/model_inits/{trait}_persona_initialization/{target_model_id}.json"

        try:
            if Path(json_filepath).exists():
                return cls.from_json(json_filepath)

        except Exception as e:
            logger.warning("⚠️ Failed to load from JSON: %s. Creating a new PersonaModel instance.", e)
            pass

        return cls(
            target_model_id=target_model_id,
            dataset=dataset,
            trait=trait,
            layer=layer,
        )

    def save_to_json(self, filepath: str = "./persona_data/model_inits/") -> str:
        """
        Save the persona vectors and initialization config to JSON
        
        Args:
            filepath (str): Directory path to save the JSON file
            
        Returns:
            str: Path to the saved JSON file
        """

        filepath += f"{self.dataset.trait}_persona_initialization/{self.target_model_id}.json"

        # Convert tensors to lists for JSON serialization
        initialization_data = {
            "target_model_id": self.target_model_id,
            "trait": self.dataset.trait if self.dataset else None,
            "layer_steering": self.layer_steering,
            "device": self.device,
            "prompt_persona_vector": self.prompt_persona_vector.tolist(),
            "response_persona_vector": self.response_persona_vector.tolist(),
            "all_layers_response_persona_vector": self.all_layers_response_persona_vector.tolist(),
            "prompt_persona_vector_shape": list(self.prompt_persona_vector.shape),
            "response_persona_vector_shape": list(self.response_persona_vector.shape),
            "created_at": datetime.now().isoformat(),
            "dataset_info": {
                "trait": self.dataset.trait,
                "num_questions": self.dataset.num_questions,
                "num_pos_neg_pairs": len(self.dataset.positive_negative_pairs)
            } if self.dataset else None
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(initialization_data, f, indent=2)

        logger.info("✅ Initialization saved to: %s", filepath)
        return filepath

    @torch.inference_mode()
    def extract_persona_vector(self,
                               temperature: float = 0.9,
                               max_new_tokens: int = 200) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        Extract persona vectors from the dataset. Main function.
        
        Args:
            temperature: Sampling temperature for generation.
            max_new_tokens: Maximum number of tokens to generate.
            
        Returns:
            Tuple of (prompt_persona_vector, response_persona_vector, response_average_all_layers)
        '''
        # Get cartesian product of all positive-negative question pairs from dataset
        trait_pairs = self.dataset.extract_pos_neg_question_pairs()

        # Randomly subsample 20 questions
        orig_n = len(trait_pairs)
        if orig_n > 20:
            trait_pairs = random.sample(trait_pairs, 20)
            logger.info("Randomly subsampled 20 questions from %d pairs", orig_n)

        # Optimization #4: Pre-tokenize and cache all prompts
        logger.info("Pre-tokenizing %d pairs...", len(trait_pairs))

        token_cache = {}
        for pos, neg, question in trait_pairs:
            # for system_prompt, key_suffix in [(pair.pos, 'pos'), (pair.neg, 'neg')]:
            for system_prompt in (pos, neg):
                # Prepare messages in chat format
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                ]

                # Pretokenizing

                ## for debugging so can see formatted
                # formatted_prompt = self.tokenizer.apply_chat_template(
                #     messages, tokenize=False, add_generation_prompt=True
                # )
                # enc = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
                ##

                enc = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.device)

                token_cache[(question, system_prompt)] = (
                    enc,
                    torch.ones_like(enc)
                )

        # Process each pair and accumulate activations (Tensor instead of list for speed)
        # Tensor of last activation of prompt
        sum_prompt_last_pos = None
        sum_prompt_last_neg = None

        # Tensor of average activation of response
        sum_resp_avg_pos = None
        sum_resp_avg_neg = None

        # Tensor of average activation of response across all layers
        sum_resp_avg_all_layers_pos = None
        sum_resp_avg_all_layers_neg = None

        n = 0

        # Extract activations for each pair
        for pos, neg, question in tqdm(trait_pairs,
                         total = len(trait_pairs),
                         desc = f"Extracting activations for trait {self.dataset.trait}"):
            pos_ids, pos_mask = token_cache[(question, pos)]
            neg_ids, neg_mask = token_cache[(question, neg)]

            # pl: prompt hidden layer last activation
            # ra: response hidden layer avg activation
            # rall: response avg all activation
            pl_pos, ra_pos, rall_pos = self._get_activations(pos_ids, pos_mask, temperature, max_new_tokens) # positive
            pl_neg, ra_neg, rall_neg = self._get_activations(neg_ids, neg_mask, temperature, max_new_tokens) # negative

            if sum_prompt_last_pos is None:
                sum_prompt_last_pos = torch.zeros_like(pl_pos, dtype=torch.float32)
                sum_prompt_last_neg = torch.zeros_like(pl_neg, dtype=torch.float32)
                sum_resp_avg_pos = torch.zeros_like(ra_pos, dtype=torch.float32)
                sum_resp_avg_neg = torch.zeros_like(ra_neg, dtype=torch.float32)
                sum_resp_avg_all_layers_pos = torch.zeros_like(rall_pos, dtype=torch.float32)
                sum_resp_avg_all_layers_neg = torch.zeros_like(rall_neg, dtype=torch.float32)

            sum_prompt_last_pos += pl_pos.float()
            sum_prompt_last_neg += pl_neg.float()
            sum_resp_avg_pos += ra_pos.float()
            sum_resp_avg_neg += ra_neg.float()
            sum_resp_avg_all_layers_pos += rall_pos.float()
            sum_resp_avg_all_layers_neg += rall_neg.float()
            n += 1

        prompt_last_pos_mean = sum_prompt_last_pos / max(n, 1)
        prompt_last_neg_mean = sum_prompt_last_neg / max(n, 1)
        response_avg_pos_mean = sum_resp_avg_pos / max(n, 1)
        response_avg_neg_mean = sum_resp_avg_neg / max(n, 1)

        all_layers_response_avg_pos_mean = sum_resp_avg_all_layers_pos / max(n, 1)
        all_layers_response_avg_neg_mean = sum_resp_avg_all_layers_neg / max(n, 1)

        prompt_persona_vector = prompt_last_pos_mean - prompt_last_neg_mean # (1, hidden)
        response_persona_vector = response_avg_pos_mean - response_avg_neg_mean # (1, hidden)
        all_layers_response_persona_vector = all_layers_response_avg_pos_mean - all_layers_response_avg_neg_mean
        # (num_states, 1, hidden)

        self.prompt_persona_vector = prompt_persona_vector.cpu()
        self.response_persona_vector = response_persona_vector.cpu()
        self.all_layers_response_persona_vector = all_layers_response_persona_vector.cpu()

        logger.info("Extracted persona vectors from %d pairs", len(trait_pairs))
        logger.info("Prompt persona vector shape: %s", str(tuple(prompt_persona_vector.shape)))
        logger.info("Response persona vector shape: %s", str(tuple(response_persona_vector.shape)))
        logger.info("All-layers response persona vector shape: %s", str(tuple(all_layers_response_persona_vector.shape)))

        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector

    @torch.inference_mode()
    def generate(self,
                 prompt: str | None = None,
                 messages: list[str] | None = None,
                 alpha: float | None = 3,
                 max_new_tokens: int = 2000,
                 temperature: float = 0.9,
                 top_p=.99) -> str:
        '''
        Generate a response to the prompt with persona steering.
        
        Args:
            prompt: The user prompt string.
            messages: Optional list of messages in chat format.
            alpha: Steering strength for persona vector.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            
        Returns:
            str: persona influenced response
        '''
        if prompt is None and messages is None:
            raise ValueError("Prompt and Messages cannot both be None")
        
        self.model.eval()

        if messages is None:
            # Prepare messages in chat format
            messages = [
                {"role": "user", "content": prompt}
            ]

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
                    past = out.past_key_values # update kv cache

                    # Sample index of next token
                    logits = out.logits[0, -1, :]
                    tok = self._top_p_sample(logits, temperature=temperature, top_p=top_p)
                    if tok in eos_ids:
                        break

                    gen_ids.append(tok)
                    last_token.fill_(tok)

        output_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return output_text


    @torch.inference_mode()
    def _get_activations(self, 
                         input_ids: torch.Tensor, 
                         attention_mask: torch.Tensor | None,
                         temperature: float = .9,
                         max_new_tokens: int = 20000,
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
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
        '''
        # prompt forward (KV cache)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )

        # Store cache
        past = out.past_key_values
        prompt_last = out.hidden_states[self.layer_steering + 1][:, -1, :]  # (1, hidden), 0th layer init embedding

        # generate first token
        logits0 = out.logits[0, -1, :]
        tok = self._top_p_sample(logits0, top_p=0.99, temperature=temperature)

        # early exit if no response tokens
        if tok in self.eos_token_ids:
            # no response tokens
            num_states = len(out.hidden_states)
            resp_avg = torch.zeros_like(prompt_last)
            resp_avg_all_layers = torch.zeros((num_states,) + prompt_last.shape, device=prompt_last.device, dtype=prompt_last.dtype)
            return prompt_last, resp_avg, resp_avg_all_layers

        # accumulate response activations
        acc_dtype = torch.float32 if prompt_last.dtype in (torch.float16, torch.bfloat16) else prompt_last.dtype
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
        # last_token = input_ids[:, -1:].clone()

        last_token = torch.tensor([[tok]], device=input_ids.device, dtype=torch.long)

        # response generation loop
        for _ in range(max_new_tokens):
            # only pass last token (KV cache)
            out = self.model(
                input_ids=last_token,
                past_key_values=past,
                output_hidden_states=True,
                use_cache=True,
            )
            past = out.past_key_values # update kv cache

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
            
            last_token.fill_(int(next_token_id))  # reuse (1,1) buffer

        if count > 0:
            resp_avg = (resp_sum / count).to(prompt_last.dtype)
            resp_avg_all_layers = (resp_sum_all_layers / count).to(prompt_last.dtype)
        else:
            resp_avg = torch.zeros_like(prompt_last)
            resp_avg_all_layers = torch.zeros_like(resp_sum_all_layers, dtype=prompt_last.dtype)

        return prompt_last, resp_avg, resp_avg_all_layers

    def _install_steer_hook(self, layer_idx: int) -> None:
        '''
        Installs one persistent hook that reads per-request delta.
        
        Args:
            layer_idx (int): layer to add steering hook to
        '''
        # Get the decoder blocks and specific target steer block
        blocks = self._get_decoder_blocks(self.model)
        block = blocks[layer_idx]
        self._steer_block = block

        def hook(_module, _inp, out):
            '''
            Hook function to add delta to the last-token hidden state at the output of the specified decoder block.
            The delta is added to the last token's hidden state.
            register_forward_hook is called on each forward pass of the block
            
            Args:
                _module (torch.nn.Module): The module instance.
                _inp (tuple): Input tensors to the module.
                out (torch.Tensor | tuple): Output tensor(s) from the module.
                
            Returns:
                torch.Tensor | tuple: Modified output tensor(s).
            '''
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
        '''
        Cached base persona vector on the steered block device/dtype
        
        Returns:
            torch.Tensor: persona vector
        '''
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
                self._persona_base = (
                    self.prompt_persona_vector.to(device=p.device, dtype=p.dtype).view(1, -1)
                )  # (1, H)
                self._persona_base_key = key
        return self._persona_base

     
    @contextmanager
    def _steering_delta(self, alpha: float):
        '''
        Per-request steering context (stores delta in ContextVar)
        
        Args:
            alpha (float): alpha applied on runtime
        '''
        if alpha == 0:
            yield
            return
        
        base = self._get_persona_base()   # (1, H)
        delta = alpha * base              # (1, H)
        tok = _STEER_DELTA.set(delta)
        
        try:
            yield
        finally:
            _STEER_DELTA.reset(tok)

    def close(self):
        '''
        Removes hooks. Typically unused
        '''
        if self._steer_hook_handle is not None:
            self._steer_hook_handle.remove()
            self._steer_hook_handle = None

    def _get_decoder_blocks(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        '''
        Retrieve the list of decoder blocks from the model.
        
        Args:
            model (torch.nn.Module): The model instance.
            
        Returns:
            list[torch.nn.Module]: List of decoder blocks.
        '''
        # Common HF layouts
        if hasattr(model, "model") and hasattr(model.model, "layers"):         # LLaMA/Qwen/Mistral
            return model.model.layers
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):  # GPT-2
            return model.transformer.h
        raise RuntimeError("Unsupported model layout: cannot locate decoder blocks.")

    def _top_p_sample(self, logits: torch.Tensor, top_p: float=0.9, temperature: float=1.0) -> int:
        '''
        Perform nucleus (top-p) sampling from logits.
        
        Args:
            logits (torch.Tensor): Tensor of shape (vocab_size,) representing model logits.
            top_p (float): Cumulative probability threshold for nucleus sampling.
            temperature (float): Sampling temperature.
            
        Returns:
            int: Sampled token index.
        '''
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate Persona Dataset")
    ap.add_argument("-t", "--trait", type=str, default="humorous", help="Trait of the persona dataset")
    ap.add_argument("-n", "--max_new_tokens", type=int, default=2000, help="Max tokens to generate")
    ap.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha value for persona steering")
    ap.add_argument("--temperature", type=float, default=0.9, help="Temperature for sampling")
    ap.add_argument("-q", "--question", type=str, default="What is the theory of relativity?", help="Question to generate response for")

    args = ap.parse_args()
    
    pvx = PersonaModel.load_or_create(
        target_model_id=model_name,
        trait=args.trait,
        layer=14,
        json_filepath=model_args.get("json_filepath"),
    )
            
    # Example 1: Create new PersonaModel from dataset (extracts vectors)
    # dataset = PersonaDataset.from_json(trait="humorous")
    # persona_model = PersonaModel(target_model_id="Qwen/Qwen2.5-1.5B-Instruct", dataset=dataset, layer=14)

    # Example 2: Load existing PersonaModel from JSON (skips vector extraction)
    # persona_model = PersonaModel.from_json(
    #     "persona_data/model_inits/verbose_persona_initialization_Qwen/Qwen2.5-1.5B-Instruct.json"
    # )
    
    response = persona_model.generate(
        prompt=args.question,
        alpha=0,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    steer_response = persona_model.generate(
        prompt=args.question,
        alpha=args.alpha,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    logger.info("=== Question ===")
    logger.info(args.question)
    logger.info('')
    logger.info("=== Non-Steered Answer ===")
    logger.info(response)
    logger.info('')
    logger.info("=== %s Steered Answer ===", args.trait)
    logger.info(steer_response)