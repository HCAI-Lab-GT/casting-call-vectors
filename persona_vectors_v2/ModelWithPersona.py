import tqdm
from Dataset import PersonaDataset
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from prompts import PROMPTS
import torch
import ollama
from typing import List, Tuple, Dict, Optional
import os
from openai import OpenAI
import re
import datetime

class ModelWithPersona:

    def __init__(self, target_model_id: str = "qwen2.5:7b-instruct", dataset: Optional[PersonaDataset] = None, layer: float = -1):
        self.target_model_id = target_model_id
        self.dataset = dataset
        self.layer_steering = layer

        self.tokenizer = AutoTokenizer.from_pretrained(target_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available() :
            self.model = self.model.to(self.device)
        self.target_model_id = target_model_id
        self.eos_token_ids = [self.tokenizer.eos_token_id] if isinstance(self.tokenizer.eos_token_id, int) else self.tokenizer.eos_token_id
        _, _ = self.extract_persona_vector()
        self.save_initialization_to_json()
    
    def save_initialization_to_json(self, filepath: str = "./model_with_persona/"):
        """Save the persona vectors and initialization config to JSON"""
        
        filepath += f"{self.dataset.trait}_persona_initialization.json"
        
        # Convert tensors to lists for JSON serialization
        initialization_data = {
            "target_model_id": self.target_model_id,
            "trait": self.dataset.trait if self.dataset else None,
            "layer_steering": self.layer_steering,
            "device": self.device,
            "prompt_persona_vector": self.prompt_persona_vector.tolist(),
            "response_persona_vector": self.response_persona_vector.tolist(),
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

        input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                output_hidden_states=True
            )
        
        prompt_last_activation = outputs.hidden_states[self.layer_steering][:, -1, :].cpu()

        hidden_activation_sum = None
        count = 0

        for _ in tqdm(range(max_new_tokens), desc="Generating with persona steering"):
            input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    output_hidden_states=True
                )
            
            final_activation = outputs.hidden_states[self.layer_steering][:, -1, :]

            if hidden_activation_sum is None:
                hidden_activation_sum = torch.zeros_like(final_activation)
            
            hidden_activation_sum += final_activation
            count += 1

            logits = outputs.logits[0, -1, :]

            next_token = self.top_p_sample(logits, temperature=temperature, top_p=0.99)
            generated.append(next_token)

            if next_token in self.eos_token_ids:
                break
        if count > 0:
            response_average = (hidden_activation_sum / count).cpu()
        else:
            response_average = torch.zeros(1, self.model.config.hidden_size)
        
        return prompt_last_activation, response_average

    def extract_persona_vector(self, temperature: float = 0.9, max_new_tokens: int = 200):

        trait_pairs = self.dataset.extract_pos_neg_question_pairs()

        all_prompt_last_pos = []
        all_prompt_last_neg = []

        all_response_avg_pos = []
        all_response_avg_neg = []

        for i, pair in tqdm(enumerate(trait_pairs), total=len(trait_pairs), desc=f"Extracting activations for trait {self.dataset.trait}"):
            prompt_last_pos, response_avg_pos = self.get_prompt_last_and_response_average(system_prompt=pair.pos, user_prompt=pair.question, temperature=temperature, max_new_tokens=max_new_tokens)
            prompt_last_neg, response_avg_neg = self.get_prompt_last_and_response_average(system_prompt=pair.neg, user_prompt=pair.question, temperature=temperature, max_new_tokens=max_new_tokens)

            all_prompt_last_pos.append(prompt_last_pos)
            all_prompt_last_neg.append(prompt_last_neg)
            all_response_avg_pos.append(response_avg_pos)
            all_response_avg_neg.append(response_avg_neg)
        
        stacked_prompt_last_pos = torch.stack(all_prompt_last_pos, dim=0)
        stacked_prompt_last_neg = torch.stack(all_prompt_last_neg, dim=0)
        stacked_response_avg_pos = torch.stack(all_response_avg_pos, dim=0)
        stacked_response_avg_neg = torch.stack(all_response_avg_neg, dim=0)

        prompt_last_pos_mean = torch.mean(stacked_prompt_last_pos, dim=0)
        prompt_last_neg_mean = torch.mean(stacked_prompt_last_neg, dim=0)
        response_avg_pos_mean = torch.mean(stacked_response_avg_pos, dim=0)
        response_avg_neg_mean = torch.mean(stacked_response_avg_neg, dim=0)
        
        prompt_persona_vector = prompt_last_pos_mean - prompt_last_neg_mean
        response_persona_vector = response_avg_pos_mean - response_avg_neg_mean

        self.prompt_persona_vector = prompt_persona_vector
        self.response_persona_vector = response_persona_vector
        
        print(f"Extracted persona vectors from {len(trait_pairs)} pairs")
        print(f"Prompt persona vector shape: {prompt_persona_vector.shape}")
        print(f"Response persona vector shape: {response_persona_vector.shape}")
        
        return prompt_persona_vector, response_persona_vector


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

        # Convert persona vector to torch tensor
        persona_tensor = self.prompt_persona_vector.to(self.device).to(
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        persona_reshaped = persona_tensor.reshape(1, 1, -1)

        for _ in tqdm(range(max_new_tokens), desc="Generating with persona steering"):

            if alpha == 0:
                # Standard generation without persona steering
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=torch.tensor([generated], dtype=torch.long).to(self.device)
                    )
                logits = outputs.logits[0, -1, :]

                # Sample next token
                next_token = self.top_p_sample(logits, temperature=temperature, top_p=0.99)
                generated.append(next_token)

                if next_token in self.eos_token_ids:
                    break
                continue
            
            # Convert to tensor and forward pass
            input_tensor = torch.tensor([generated], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_tensor,
                    output_hidden_states=True
                )
            
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

    dataset = PersonaDataset.load_dataset_from_json(trait="sarcastic")
    model_with_persona = ModelWithPersona(target_model_id="Qwen/Qwen2.5-32B-Instruct", dataset=dataset, layer=14)

