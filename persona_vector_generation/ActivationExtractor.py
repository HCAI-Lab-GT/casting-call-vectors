import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import mlx.core as mx
from mlx_lm import load
import json
from pathlib import Path
from data_generation.TraitGenerator import TraitGenerator
import numpy as np

class ActivationExtractor:
    def __init__(self, model_id: str='mlx-community/Qwen2.5-7B-Instruct-4bit'):
        self.model, self.tokenizer = load(model_id)
        self.model_id = model_id
    
    def extract_last_layer_activations(self, system_prompt: str, user_prompt: str):
        # Format messages for chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        tokens = self.tokenizer.encode(prompt, return_tensors="mlx")
        
        # Forward pass through model to get hidden states
        # We need to manually call the model's forward method
        outputs = self.model(tokens, cache=None)
        
        hidden = outputs

        # Get the last token’s activation (right before generation starts)
        final_activation = hidden[0, -1, :]   # shape: (hidden_dim,)
        print("Activation shape:", final_activation.shape)
        return final_activation
    
    def extract_persona_vector(self, trait: str):
        trait_file_path = Path("../data_generation/traits_output") / f"{trait}.json"
        
        trait_generator = TraitGenerator(model=self.model, tokenizer=self.tokenizer)

        if trait_file_path.exists():
            print(f"Found trait file")
            with open(trait_file_path, 'r') as f:
                trait_data = json.load(f)
        else:
            trait_data = trait_generator.generate_pos_neg_examples_v2(trait=trait)
        
        trait_pairs = trait_generator.extract_pos_neg_question_pairs(trait_data)

        print("Total Pairs: ", len(trait_pairs))

        pos_activations = []
        neg_activations = []
        for i, pair in enumerate(trait_pairs):
            final_activations_pos = self.extract_last_layer_activations(system_prompt=pair.pos, user_prompt=pair.question)
            final_activations_neg = self.extract_last_layer_activations(system_prompt=pair.neg, user_prompt=pair.question)

            pos_activations.append(final_activations_pos)
            neg_activations.append(final_activations_neg)

            print("Completed pair ", i)
        
        np_pos_activations = np.array(pos_activations)
        np_neg_activations = np.array(neg_activations)

        mean_pos_activation = np.mean(np_pos_activations, axis=0)
        mean_neg_activation = np.mean(np_neg_activations, axis=0)

        persona_vector = mean_pos_activation - mean_neg_activation

        return persona_vector, mean_pos_activation, mean_neg_activation
            

        
        
    

if __name__ == "__main__":
    extractor = ActivationExtractor()
    
    # system_prompt = "Your responses should be sarcastic and clever. Do not soften your language or try to be straightforward."
    # user_prompt = "How would you respond if a colleague suggests a new policy that you think is unnecessary and a waste of time?"
    
    # result = extractor.extract_last_layer_activations(
    #     system_prompt=system_prompt,
    #     user_prompt=user_prompt
    # )

    # print(result)

    persona_vector, mean_pos_activation, mean_neg_activation = extractor.extract_persona_vector(trait='sarcastic')
    print(persona_vector)