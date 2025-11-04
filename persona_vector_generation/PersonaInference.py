import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import asyncio

import mlx.core as mx
from mlx_lm import load
import json
from pathlib import Path
from data_generation.TraitGenerator import TraitGenerator
from persona_vector_generation.ActivationExtractor import ActivationExtractor
import numpy as np


class PersonaInference:

    def __init__(self, model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"):
        self.model_id = model_id
        self.model, self.tokenizer = load(model_id)
        print(self.tokenizer._eos_token_ids)
    
    def stable_softmax(self, x, temperature=1.0):
        x = x / temperature
        x = x - mx.max(x)  # <-- this prevents overflow
        exp_x = mx.exp(x)
        return exp_x / mx.sum(exp_x)
    
    def top_p_sample(self, logits, temperature=0.9, top_p=0.9):
        # convert to numpy
        logits = np.array(logits, dtype=np.float64)
        logits /= temperature
        logits -= np.max(logits)

        probs = np.exp(logits)
        probs /= np.sum(probs)

        # sort tokens by probability
        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]

        # cumulative distribution
        cdf = np.cumsum(sorted_probs)
        cutoff = np.searchsorted(cdf, top_p) + 1

        # keep only top_p portion
        sorted_idx = sorted_idx[:cutoff]
        sorted_probs = sorted_probs[:cutoff]
        sorted_probs /= np.sum(sorted_probs)

        # sample
        next_token = np.random.choice(sorted_idx, p=sorted_probs)
        return int(next_token)

    async def inference_with_persona(self, trait: str, prompt: str, alpha: float = 1.2, temperature: float = 0.9, max_new_tokens=76):

        persona_extractor = ActivationExtractor(self.model, self.tokenizer, model_id=self.model_id)

        loaded_vector_map = persona_extractor.load_persona_data(trait)
        if loaded_vector_map:
            persona_vector = loaded_vector_map[0]
        else:
            persona_vector, _, _ = await persona_extractor.extract_persona_vector(trait)
        
        messages = [
            {'role': "system", "content": "Only respond to the question at hand and simply provide an answer to the question."},
            {"role": "user", "content": prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        tokens = self.tokenizer.encode(prompt, return_tensors="mlx")
        generated = tokens[0].tolist()

        persona_reshaped = persona_vector.reshape(1, 1, -1)

        for _ in range(max_new_tokens):
            
            inputs = mx.array(generated, dtype=tokens.dtype).reshape(1, -1)
            outputs = self.model.model(inputs)
            final_activation = outputs[:, -1, :]
            
            steered_hidden = final_activation + alpha * persona_reshaped

            logits = self.model.lm_head(steered_hidden)[0, -1, :]

            logits_np = np.array(logits, dtype=np.float64)

            logits_np = logits_np / temperature
            logits_np = logits_np - np.max(logits_np)
            exp_x = np.exp(logits_np)
            probs = exp_x / np.sum(exp_x)
            probs = np.nan_to_num(probs)
            probs = probs / np.sum(probs) 

            next_token = self.top_p_sample(logits_np, temperature=temperature, top_p=0.9)
            generated.append(next_token)

            if next_token in self.tokenizer._eos_token_ids:
                break

        return self.tokenizer.decode(generated[len(tokens):])

        
if __name__ == "__main__":
    persona_inference = PersonaInference()
    

    token = asyncio.run(persona_inference.inference_with_persona(trait='evil', prompt="How would you respond if a client accused you of being unprofessional or unresponsive?", alpha=2.0))
    print(token)

    # token = persona_inference.inference_with_persona(trait='sarcastic', prompt="What would you say to a friend who just bought an overpriced, impractical gadget they claim is the best thing since sliced bread?", alpha=0.8)
    # print(token)