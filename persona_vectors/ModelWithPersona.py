from tqdm import tqdm
from .PersonaDataset import PersonaDataset
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
torch.set_float32_matmul_precision('high')
from typing import List, Tuple, Dict, Optional
from openai import OpenAI
from datetime import datetime
import random 

class ModelWithPersona:

    @classmethod
    def from_json(cls, json_filepath: str):
        """
        Load a ModelWithPersona instance from a previously saved JSON file.
        
        Args:
            json_filepath: Path to the JSON file containing the saved initialization data
            
        Returns:
            ModelWithPersona: A new instance with the loaded persona vectors
        """
        with open(json_filepath, 'r') as f:
            data = json.load(f)
        
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
        
        # Optimization #2: Initialize cache for persona vector reshape
        instance._persona_reshaped_cache = None
        
        # Store additional metadata
        if "dataset_info" in data and data["dataset_info"]:
            instance.trait = data["dataset_info"]["trait"]
        
        print(f"✅ Loaded ModelWithPersona from: {json_filepath}")
        print(f"   Model: {instance.target_model_id}")
        print(f"   Layer: {instance.layer_steering}")
        if hasattr(instance, 'trait'):
            print(f"   Trait: {instance.trait}")
        print(f"   Prompt persona vector shape: {instance.prompt_persona_vector.shape}")
        print(f"   Response persona vector shape: {instance.response_persona_vector.shape}")
        
        return instance

    def __init__(self, target_model_id: str = "qwen2.5:7b-instruct", dataset: Optional[PersonaDataset] = None, layer: float = -1, from_json: bool = False):
        self.target_model_id = target_model_id
        self.dataset = dataset
        self.layer_steering = layer

        self.tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.to(self.device)
        self.eos_token_ids = [self.tokenizer.eos_token_id] if isinstance(self.tokenizer.eos_token_id, int) else self.tokenizer.eos_token_id
        
        # Optimization #7: Use torch.compile if available (PyTorch 2.0+)
        # Note: Disabled by default due to compatibility issues with transformers + CUDA graphs
        # To enable, set environment variable: ENABLE_TORCH_COMPILE=1
        import os
        if hasattr(torch, 'compile') and os.environ.get('ENABLE_TORCH_COMPILE', '0') == '1':
            try:
                # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
                self.model = torch.compile(self.model, mode='default', dynamic=True)
                print("✅ Model compiled with torch.compile for faster inference")
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")
                print("   Continuing without compilation...")
        
        # Optimization #2: Cache for persona vector reshape
        self._persona_reshaped_cache = None
        
        # Only extract and save if not loading from JSON
        if not from_json:
            _, _ = self.extract_persona_vector()
            self.save_initialization_to_json()
    
    def save_initialization_to_json(self, filepath: str = "./model_with_persona/"):
        """Save the persona vectors and initialization config to JSON"""
        
        filepath += f"{self.dataset.trait}_persona_initialization_{self.target_model_id}.json"
        
        # Convert tensors to lists for JSON serialization
        initialization_data = {
            "target_model_id": self.target_model_id,
            "trait": self.dataset.trait if self.dataset else None,
            "layer_steering": self.layer_steering,
            "device": self.device,
            "prompt_persona_vector": self.prompt_persona_vector.tolist(),
            "response_persona_vector": self.response_persona_vector.tolist(),
            "response_average_all_layers": self.response_persona_vector_all_layers.tolist(),
            "prompt_persona_vector_shape": list(self.prompt_persona_vector.shape),
            "response_persona_vector_shape": list(self.response_persona_vector.shape),
            "created_at": datetime.now().isoformat(),
            "dataset_info": {
                "trait": self.dataset.trait,
                "num_questions": self.dataset.num_questions,
                "num_pos_neg_pairs": len(self.dataset.positive_negative_pairs)
            } if self.dataset else None
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(initialization_data, f, indent=2)
        
        print(f"✅ Initialization saved to: {filepath}")
        return filepath
    @classmethod
    def load_from_json(cls, filepath: str):
        """Load ModelWithPersona from a saved JSON file"""
        
        # Load the JSON data
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        print(f"Loading ModelWithPersona from: {filepath}")
        print(f"Original trait: {data.get('trait', 'Unknown')}")
        print(f"Created: {data.get('created_at', 'Unknown')}")
        
        # Create instance without running extract_persona_vector
        instance = cls.__new__(cls)  # Create instance without calling __init__
        
        # Set basic attributes
        instance.target_model_id = data["target_model_id"]
        instance.layer_steering = data["layer_steering"]
        instance.device = "cuda" if torch.cuda.is_available() else "cpu"  # Use current device
        
        # Load model and tokenizer
        print(f"Loading model: {instance.target_model_id}")
        instance.tokenizer = AutoTokenizer.from_pretrained(instance.target_model_id)
        instance.model = AutoModelForCausalLM.from_pretrained(
            instance.target_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if not torch.cuda.is_available():
            instance.model = instance.model.to(instance.device)
        
        instance.eos_token_ids = [instance.tokenizer.eos_token_id] if isinstance(instance.tokenizer.eos_token_id, int) else instance.tokenizer.eos_token_id
        
        # Load persona vectors from JSON
        instance.prompt_persona_vector = torch.tensor(data["prompt_persona_vector"]).to(instance.device)
        instance.response_persona_vector = torch.tensor(data["response_persona_vector"]).to(instance.device)
        
        # Set dataset to None (since we're loading from JSON)
        dataset_trait = data["dataset_info"]["trait"]
        dataset = PersonaDataset.load_dataset_from_json(trait=dataset_trait)
        instance.dataset = dataset
        
        print(f"✅ ModelWithPersona loaded successfully!")
        print(f"Prompt persona vector shape: {instance.prompt_persona_vector.shape}")
        print(f"Response persona vector shape: {instance.response_persona_vector.shape}")
        
        return instance
    
    def extract_hidden_layer_activations(self, system_prompt: str, user_prompt: str, layer: int = -1):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )

        last_hidden_state = outputs.hidden_states[layer] 
        final_activation = last_hidden_state[:, -1, :] 
        
        return final_activation.cpu()

    def top_p_sample(self, logits, top_p=0.9, temperature=1.0): 
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
    
    def get_prompt_last_and_response_average(self, system_prompt: str, user_prompt: str, temperature: float = 0.9, max_new_tokens: int = 200):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        generated = input_ids[0].tolist()

        # Optimization #3: Initialize KV-cache for faster generation
        past_key_values = None

        # First forward pass to get prompt activation
        input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                output_hidden_states=True,
                use_cache=True  # Optimization #3: Enable KV-cache
            )
        
        past_key_values = outputs.past_key_values  # Optimization #3: Store cache
        # Optimization #6: Keep in GPU memory (FP16) instead of moving to CPU
        prompt_last_activation = outputs.hidden_states[self.layer_steering][:, -1, :]

        hidden_activation_sum = None
        hidden_activation_sum_per_layer = None
        count = 0

        # Optimization #1: Remove tqdm (no progress bar in this loop)
        for _ in range(max_new_tokens):
            # Optimization #3: Only pass the last token (KV-cache optimization)
            input_tensor = torch.tensor([[generated[-1]]], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    past_key_values=past_key_values,  # Optimization #3: Reuse cache
                    output_hidden_states=True,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values  # Optimization #3: Update cache
            final_activation = outputs.hidden_states[self.layer_steering][:, -1, :]


            # print(f"Computed Final Activation - Iteration {iteration}")

            if hidden_activation_sum is None:
                hidden_activation_sum = torch.zeros_like(final_activation)
                hidden_activation_sum_per_layer = torch.zeros_like(all_layers_last_token)
            
            hidden_activation_sum += final_activation
            hidden_activation_sum_per_layer += all_layers_last_token
            count += 1

            logits = outputs.logits[0, -1, :]

            next_token = self.top_p_sample(logits, temperature=temperature, top_p=0.99)
            generated.append(next_token)

            if next_token in self.eos_token_ids:
                break
        
        # Optimization #6: Keep in GPU memory (FP16) instead of moving to CPU
        if count > 0:
            response_average = hidden_activation_sum / count
        else:
            response_average = torch.zeros(1, self.model.config.hidden_size, 
                                          device=self.device, 
                                          dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
        
        return prompt_last_activation, response_average, response_average_all_layers

    def extract_persona_vector(self, temperature: float = 0.9, max_new_tokens: int = 200):

        trait_pairs = self.dataset.extract_pos_neg_question_pairs()
        
        # Randomly subsample 20 questions
        if len(trait_pairs) > 20:
            trait_pairs = random.sample(trait_pairs, 20)
            print(f"Randomly subsampled 20 questions from {len(self.dataset.extract_pos_neg_question_pairs())} pairs")

        # Optimization #4: Pre-tokenize and cache all prompts
        print(f"Pre-tokenizing {len(trait_pairs)} pairs...")
        tokenized_prompts = {}
        for pair in trait_pairs:
            for system_prompt, key_suffix in [(pair.pos, 'pos'), (pair.neg, 'neg')]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": pair.question}
                ]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                # Cache both formatted prompt and question for lookup
                cache_key = (pair.question, system_prompt)
                tokenized_prompts[cache_key] = formatted_prompt

        all_prompt_last_pos = []
        all_prompt_last_neg = []

        all_response_avg_pos = []
        all_response_avg_neg = []

        all_layers_response_avg_pos = []
        all_layers_response_avg_neg = []

        for i, pair in tqdm(enumerate(trait_pairs), total=len(trait_pairs), desc=f"Extracting activations for trait {self.dataset.trait}"):
            prompt_last_pos, response_avg_pos, response_avg_all_layers_pos = self.get_prompt_last_and_response_average(system_prompt=pair.pos, user_prompt=pair.question, temperature=temperature, max_new_tokens=max_new_tokens)
            prompt_last_neg, response_avg_neg, response_avg_all_layers_neg = self.get_prompt_last_and_response_average(system_prompt=pair.neg, user_prompt=pair.question, temperature=temperature, max_new_tokens=max_new_tokens)

            all_prompt_last_pos.append(prompt_last_pos)
            all_prompt_last_neg.append(prompt_last_neg)
            all_response_avg_pos.append(response_avg_pos)
            all_response_avg_neg.append(response_avg_neg)
        
        # Optimization #6: Keep all tensors on GPU during computation
        stacked_prompt_last_pos = torch.stack(all_prompt_last_pos, dim=0)
        stacked_prompt_last_neg = torch.stack(all_prompt_last_neg, dim=0)
        stacked_response_avg_pos = torch.stack(all_response_avg_pos, dim=0)
        stacked_response_avg_neg = torch.stack(all_response_avg_neg, dim=0)

        stacked_all_layers_response_avg_pos = torch.stack(all_layers_response_avg_pos, dim=0)
        stacked_all_layers_response_avg_neg = torch.stack(all_layers_response_avg_neg, dim=0)

        prompt_last_pos_mean = torch.mean(stacked_prompt_last_pos, dim=0)
        prompt_last_neg_mean = torch.mean(stacked_prompt_last_neg, dim=0)
        response_avg_pos_mean = torch.mean(stacked_response_avg_pos, dim=0)
        response_avg_neg_mean = torch.mean(stacked_response_avg_neg, dim=0)

        all_layers_response_avg_pos_mean = torch.mean(stacked_all_layers_response_avg_pos)
        all_layers_response_avg_neg_mean = torch.mean(stacked_all_layers_response_avg_neg)

        
        prompt_persona_vector = prompt_last_pos_mean - prompt_last_neg_mean
        response_persona_vector = response_avg_pos_mean - response_avg_neg_mean
        all_layers_response_persona_vector = all_layers_response_avg_pos_mean - all_layers_response_avg_neg_mean

        # Optimization #6: Move to CPU only at the end for storage
        self.prompt_persona_vector = prompt_persona_vector.cpu()
        self.response_persona_vector = response_persona_vector.cpu()
        
        print(f"Extracted persona vectors from {len(trait_pairs)} pairs")
        print(f"Prompt persona vector shape: {prompt_persona_vector.shape}")
        print(f"Response persona vector shape: {response_persona_vector.shape}")
        
        return prompt_persona_vector, response_persona_vector, all_layers_response_persona_vector


    def inference_with_persona(self, prompt: str, alpha: float, max_new_tokens: int = 200, temperature: float = 0.9):

        messages = [
            {"role": "user", "content": prompt}
        ]

        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        generated = input_ids[0].tolist()

        # Optimization #2: Cache persona vector reshape
        if alpha != 0:
            if self._persona_reshaped_cache is None or self._persona_reshaped_cache.device != self.device:
                persona_tensor = self.prompt_persona_vector.to(self.device).to(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                )
                self._persona_reshaped_cache = persona_tensor.reshape(1, 1, -1)
            persona_reshaped = self._persona_reshaped_cache

        # Optimization #3: Initialize KV-cache for faster generation
        past_key_values = None

        # Optimization #1: Remove tqdm progress bar (removed from loop)
        for step in range(max_new_tokens):
            
            # Optimization #3: Only pass last token after first iteration (KV-cache optimization)
            if step == 0:
                input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
            else:
                input_tensor = torch.tensor([[generated[-1]]], dtype=torch.long).to(self.device)

            if alpha == 0:
                # Standard generation without persona steering
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_tensor,
                        past_key_values=past_key_values,  # Optimization #3: Use KV-cache
                        use_cache=True
                    )
                
                past_key_values = outputs.past_key_values  # Optimization #3: Update cache
                logits = outputs.logits[0, -1, :]

                # Sample next token
                next_token = self.top_p_sample(logits, temperature=temperature, top_p=0.99)
                generated.append(next_token)

                if next_token in self.eos_token_ids:
                    break
                continue
            
            # Optimization #6: Only request hidden_states when alpha != 0
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    past_key_values=past_key_values,  # Optimization #3: Use KV-cache
                    output_hidden_states=True,  # Optimization #6: Only when needed
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values  # Optimization #3: Update cache
            
            # Get final hidden state
            final_activation = outputs.hidden_states[self.layer_steering][:, -1, :]
            
            # Apply persona steering
            steered_hidden = final_activation + alpha * persona_reshaped

            # Get logits from language model head
            logits = self.model.lm_head(steered_hidden)[0, -1, :]

            # Sample next token
            next_token = self.top_p_sample(logits, temperature=temperature, top_p=0.99)
            generated.append(next_token)

            if next_token in self.eos_token_ids:
                break

        # Decode only the generated part (skip the input)
        return self.tokenizer.decode(generated[len(input_ids[0]):], skip_special_tokens=True)
    


if __name__ == "__main__":
    
    # Example 1: Create new ModelWithPersona from dataset (extracts vectors)
    dataset = PersonaDataset.load_dataset_from_json(trait="sarcastic")
    model_with_persona = ModelWithPersona(target_model_id="Qwen/Qwen2.5-7B-Instruct", dataset=dataset, layer=14)
    
    # Example 2: Load existing ModelWithPersona from JSON (skips vector extraction)
    # model_with_persona = ModelWithPersona.from_json("./model_with_persona/sarcastic_persona_initialization.json")

