import sys
from pathlib import Path
import asyncio
sys.path.append(str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_lm import load
import json
from pathlib import Path
from data_generation.TraitGenerator import TraitGenerator
import numpy as np

class ActivationExtractor:
    def __init__(self, local_model: bool = False, model = None, tokenizer = None, model_id: str='mlx-community/Qwen2.5-7B-Instruct-4bit'):
        self.local_model = local_model
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.model, self.tokenizer = load(model_id)
        self.model_id = model_id

    def get_all_layer_hidden_states(self, tokens):
        """
        Returns a list of hidden activations, one per layer, shape (1, seq_len, hidden_dim)
        """
        hidden_states = []
        x = self.model.model.embed_tokens(tokens)

        # some models also add positional embeddings:
        if hasattr(self.model.model, "embed_positions"):
            x = x + self.model.model.embed_positions(tokens)

        for layer in self.model.model.layers:
            x = layer(x)
            hidden_states.append(x)  # save after this layer

        # final norm if present
        if hasattr(self.model.model, "norm"):
            x = self.model.model.norm(x)
            hidden_states.append(x)

        return hidden_states
    
    def extract_last_layer_activations(self, system_prompt: str, user_prompt: str):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        tokens = self.tokenizer.encode(prompt, return_tensors="mlx")
        
        outputs = self.model.model(tokens, cache=None)
        
        final_activation = outputs[:, -1, :]
        return final_activation
    def get_response_avg(self, system_prompt, user_prompt, max_new_tokens=64, temperature=0.9, top_p=0.9):
        """
        Returns average hidden activation (response_avg) across generated tokens.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        tokens = self.tokenizer.encode(prompt, return_tensors="mlx")
        generated = tokens[0].tolist()
        hidden_sum = None
        count = 0

        for _ in range(max_new_tokens):
            inp = mx.array(generated, dtype=tokens.dtype).reshape(1, -1)
            hidden = self.model.model(inp)               # (1, seq_len, hidden_dim)
            final_h = hidden[:, -1:, :]             # last token (1,1,H)

            # 2. Predict next token
            logits = self.model.lm_head(final_h)[0, -1, :]
            logits_np = np.array(logits, dtype=np.float64)
            logits_np = logits_np / temperature
            logits_np -= np.max(logits_np)
            probs = np.exp(logits_np)
            probs /= np.sum(probs)
            next_token = int(np.random.choice(len(probs), p=probs))

            # 3. Append to generated tokens
            generated.append(next_token)

            # 4. Add current hidden vector to running sum
            h_np = np.array(final_h[0, 0, :], dtype=np.float32)
            if hidden_sum is None:
                hidden_sum = np.zeros_like(h_np)
            hidden_sum += h_np
            count += 1

            # stop on EOS
            if next_token in getattr(self.tokenizer, "_eos_token_ids", []):
                break

        # 5. Compute mean hidden activation across generated tokens
        response_avg = hidden_sum / max(count, 1)
        return response_avg
    
    def compute_persona_vector(self, trait: str, activations_dir='./persona_vectors_prompt_response_vectors', save_dir="./persona_vectors"):
        trait_dir = Path(activations_dir) / trait
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        all_files = sorted(trait_dir.glob("*.json"))

        pos_prompt_last, pos_response_avg = [], []
        neg_prompt_last, neg_response_avg = [], []

        for file in all_files:
            with open(file, "r") as f:
                data = json.load(f)
            for entry in data:
                pos_prompt_last.append(np.array(entry["pos_prompt_last"], dtype=np.float32))
                pos_response_avg.append(np.array(entry["pos_response_avg"], dtype=np.float32))
                neg_prompt_last.append(np.array(entry["neg_prompt_last"], dtype=np.float32))
                neg_response_avg.append(np.array(entry["neg_response_avg"], dtype=np.float32))

        # Stack and compute means
        pos_prompt_last_mean = np.mean(np.stack(pos_prompt_last), axis=0)
        pos_response_avg_mean = np.mean(np.stack(pos_response_avg), axis=0)
        neg_prompt_last_mean = np.mean(np.stack(neg_prompt_last), axis=0)
        neg_response_avg_mean = np.mean(np.stack(neg_response_avg), axis=0)

        # Compute difference vectors
        persona_prompt_vector = pos_prompt_last_mean - neg_prompt_last_mean
        persona_response_vector = pos_response_avg_mean - neg_response_avg_mean

        # Normalize for consistency
        persona_prompt_vector /= np.linalg.norm(persona_prompt_vector)
        persona_response_vector /= np.linalg.norm(persona_response_vector)

        # Save results
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"{trait}_prompt_vector.npy", persona_prompt_vector)
        np.save(save_dir / f"{trait}_response_vector.npy", persona_response_vector)

        print(f"✅ Saved persona vectors for '{trait}' to {save_dir}")
        print(f"Shapes: prompt {persona_prompt_vector.shape}, response {persona_response_vector.shape}")

    
    async def extract_persona_vector(self, trait: str):
        trait_file_path = Path("../data_generation/traits_output") / f"{trait}.json"
        save_dir = Path(f'./persona_vectors_prompt_response_vectors/{trait}')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        trait_generator = TraitGenerator(local_model=self.local_model,model=self.model, tokenizer=self.tokenizer, output_dir="../data_generation/traits_output")

        if trait_file_path.exists():
            print(f"Found trait file")
            with open(trait_file_path, 'r') as f:
                trait_data = json.load(f)
        else:
            trait_data = await trait_generator.generate_pos_neg_examples_v2(trait=trait)
        
        trait_pairs = trait_generator.extract_pos_neg_question_pairs(trait_data)

        print("Total Pairs: ", len(trait_pairs))

        activations = []
        for i, pair in enumerate(trait_pairs[:10]):
            print("System Prompt pos: " + pair.pos, " User Prompt: " + pair.question)
            print("System Prompt neg: " + pair.neg, " User Prompt: " + pair.question)
            pos_prompt_last_activation = self.extract_last_layer_activations(system_prompt=pair.pos, user_prompt=pair.question)
            pos_prompt_last_activation = pos_prompt_last_activation.squeeze(0)
            pos_response_average = self.get_response_avg(system_prompt=pair.pos, user_prompt=pair.question)

            neg_prompt_last_activation = self.extract_last_layer_activations(system_prompt=pair.neg, user_prompt=pair.question)
            neg_prompt_last_activation = neg_prompt_last_activation.squeeze(0)
            neg_response_average = self.get_response_avg(system_prompt=pair.neg, user_prompt=pair.question)

            activations.append({
            "question": pair.question,
            "pos_prompt_last": pos_prompt_last_activation.tolist(),
            "pos_response_avg": pos_response_average.tolist(),
            "neg_prompt_last": neg_prompt_last_activation.tolist(),
            "neg_response_avg": neg_response_average.tolist()
            })

            if (i + 1) % 5 == 0:  # save periodically
                with open(save_dir / f"batch_{i+1}.json", "w") as f:
                    json.dump(activations, f)
                activations = []

            print("Completed pair ", i)
        
        if activations:
            with open(save_dir / "final_batch.json", "w") as f:
                json.dump(activations, f)

        
        
        return
    def save_persona_data(self, trait: str, persona_vector, mean_pos_activation, mean_neg_activation, output_dir="persona_vectors"):
        """Save persona vector and activations to JSON file"""
        
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
       
        def to_serializable(arr):
            if hasattr(arr, '__array__'):
                return np.array(arr).tolist()
            elif isinstance(arr, np.ndarray):
                return arr.tolist()
            else:
                return arr
            
        save_data = {
            "trait": trait,
            "persona_vector": to_serializable(persona_vector),
            "mean_pos_activation": to_serializable(mean_pos_activation),
            "mean_neg_activation": to_serializable(mean_neg_activation),
            "vector_shape": list(persona_vector.shape) if hasattr(persona_vector, 'shape') else len(persona_vector),
            "model_id": self.model_id
        }
        
        output_file = output_path / f"{trait}_persona_vector.json"
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"✅ Saved persona data to {output_file}")
        return output_file

    def load_persona_data(self, trait: str, input_dir="persona_vectors"):
        """Load persona vector and activations from JSON file"""
        
        input_path = Path(input_dir) / f"{trait}_persona_vector.json"
        
        if not input_path.exists():
            print(f"File not found: {input_path}")
            return None
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        persona_vector = np.array(data["persona_vector"])
        mean_pos_activation = np.array(data["mean_pos_activation"])
        mean_neg_activation = np.array(data["mean_neg_activation"])
        
        print(f"✅ Loaded persona data from {input_path}")
        print(f"Vector shape: {persona_vector.shape}")
        
        return persona_vector, mean_pos_activation, mean_neg_activation, data
            

        
        
    

if __name__ == "__main__":
    extractor = ActivationExtractor(local_model=True)
    
    # system_prompt = "Your responses should be sarcastic and clever. Do not soften your language or try to be straightforward."
    # user_prompt = "How would you respond if a colleague suggests a new policy that you think is unnecessary and a waste of time?"
    
    # result = extractor.extract_last_layer_activations(
    #     system_prompt=system_prompt,
    #     user_prompt=user_prompt
    # )

    # print(result)

    # asyncio.run(extractor.extract_persona_vector(trait='sarcastic'))
    extractor.compute_persona_vector('sarcastic')