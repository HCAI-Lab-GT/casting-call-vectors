import sys
from pathlib import Path
import asyncio
sys.path.append(str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path
from data_generation.TraitGenerator import TraitGenerator
import numpy as np
import dill as pickle
from tqdm import tqdm



class ActivationExtractor:
    def __init__(self, model = None, tokenizer = None, model_id: str='Qwen/Qwen2.5-7B-Instruct'):
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available() and model is None:
            self.model = self.model.to(self.device)
        self.model_id = model_id
    
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
        
        # Tokenize and move to device
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get model outputs (hidden states)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Extract last layer activation for the final token
        last_hidden_state = outputs.hidden_states[-1]  # Last layer
        final_activation = last_hidden_state[:, -1, :]  # Last token
        
        # print("Activation shape:", final_activation.shape)
        return final_activation.cpu()
    
    async def extract_persona_vector(self, trait: str):
        trait_file_path = Path("../data_generation/traits_output") / f"{trait}.json"
        
        trait_generator = TraitGenerator(model_id=self.model_id)

        if trait_file_path.exists():
            print(f"Found trait file")
            with open(trait_file_path, 'r') as f:
                trait_data = json.load(f)
        else:
            trait_data = await trait_generator.generate_pos_neg_examples_v2(trait=trait)
        
        trait_pairs = trait_generator.extract_pos_neg_question_pairs(trait_data)

        print("Total Pairs: ", len(trait_pairs))

        pos_activations = []
        neg_activations = []
        for i, pair in tqdm(enumerate(trait_pairs), total=len(trait_pairs), desc=f"Extracting activations for trait {trait}"):
            final_activations_pos = self.extract_last_layer_activations(system_prompt=pair.pos, user_prompt=pair.question)
            final_activations_neg = self.extract_last_layer_activations(system_prompt=pair.neg, user_prompt=pair.question)

            # Convert to numpy and squeeze
            pos_activations.append(final_activations_pos.squeeze(0).numpy())
            neg_activations.append(final_activations_neg.squeeze(0).numpy())
        
        np_pos_activations = np.array(pos_activations)
        np_neg_activations = np.array(neg_activations)

        mean_pos_activation = np.mean(np_pos_activations, axis=0)
        mean_neg_activation = np.mean(np_neg_activations, axis=0)

        persona_vector = mean_pos_activation - mean_neg_activation

        self.save_persona_data(trait, persona_vector, mean_pos_activation, mean_neg_activation)

        return persona_vector, mean_pos_activation, mean_neg_activation
    
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
    extractor = ActivationExtractor()
    pickle.dump(extractor, open("activation_extractor.pkl", "wb"))
    # system_prompt = "Your responses should be sarcastic and clever. Do not soften your language or try to be straightforward."
    # user_prompt = "How would you respond if a colleague suggests a new policy that you think is unnecessary and a waste of time?"
    
    # result = extractor.extract_last_layer_activations(
    #     system_prompt=system_prompt,
    #     user_prompt=user_prompt
    # )

    # print(result)
    persona_vector, mean_pos_activation, mean_neg_activation = asyncio.run(extractor.extract_persona_vector(trait='sarcastic'))
    print(persona_vector)