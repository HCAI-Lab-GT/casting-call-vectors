import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import asyncio

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
from data_generation.TraitGenerator import TraitGenerator
from persona_vector_generation.ActivationExtractor import ActivationExtractor
import numpy as np
from tqdm import tqdm


class PersonaInference:
    def __init__(self, trait: str, model_id: str = "Qwen/Qwen2.5-32B-Instruct", trait_file: str = None):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            output_hidden_states=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            print('No cuda avail!')
            self.model = self.model.to(self.device)
        
        if trait_file is not None:
            with open(trait_file, 'r') as f:
                trait_data = json.load(f)
            self.persona_vector = np.array(trait_data['persona_vector'])
        else:
            persona_extractor = ActivationExtractor(self.model, self.tokenizer, model_id=self.model_id)
            self.persona_vector, _, _ = asyncio.run(persona_extractor.extract_persona_vector(trait))
        
        print("EOS token IDs:", self.tokenizer.eos_token_id)
        self.eos_token_ids = [self.tokenizer.eos_token_id] if isinstance(self.tokenizer.eos_token_id, int) else self.tokenizer.eos_token_id

    
    def stable_softmax(self, x, temperature=1.0):
        """Compute stable softmax using PyTorch"""
        x = torch.from_numpy(x) if isinstance(x, np.ndarray) else x
        x = x / temperature
        x = x - torch.max(x)  # prevent overflow
        exp_x = torch.exp(x)
        return (exp_x / torch.sum(exp_x)).numpy()

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

    def inference_with_persona(self, prompt: str, alpha: float = 2, temperature: float = 1, max_new_tokens=200):
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
        persona_tensor = torch.from_numpy(self.persona_vector).to(self.device).to(
            torch.float16 if torch.cuda.is_available() else torch.float32
        )
        persona_reshaped = persona_tensor.reshape(1, 1, -1)

        # Get EOS token(s)

        for _ in tqdm(range(max_new_tokens), desc="Generating with persona steering"):

            if alpha == 0:
                # Standard generation without persona steering
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=torch.tensor([generated], dtype=torch.long).to(self.device)
                    )
                logits = outputs.logits[0, -1, :]

                # Convert to numpy for sampling
                # logits_np = logits.cpu().detach().numpy().astype(np.float64)

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
            final_activation = outputs.hidden_states[-1][:, -1, :]
            
            # Apply persona steering
            steered_hidden = final_activation + alpha * persona_reshaped

            # Get logits from language model head
            logits = self.model.lm_head(steered_hidden)[0, -1, :]

            # Convert to numpy for sampling
            logits_np = logits.cpu().detach().numpy().astype(np.float64)

            # Sample next token
            next_token = self.top_p_sample(logits, temperature=temperature, top_p=0.99)
            generated.append(next_token)

            if next_token in self.eos_token_ids:
                break

        # Decode only the generated part (skip the input)
        return self.tokenizer.decode(generated[len(input_ids[0]):], skip_special_tokens=True)

        
if __name__ == "__main__":
    persona_inference = PersonaInference(trait='sarcastic', trait_file='persona_vectors/sarcastic_persona_vector.json')
    token = asyncio.run(persona_inference.inference_with_persona(prompt="Hey do you like my hat", alpha=0.5, temperature=0.8, max_new_tokens=100))
    print(token)
