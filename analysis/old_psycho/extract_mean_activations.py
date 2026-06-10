import torch
import glob
import os
import torch.nn.functional as F
from safetensors.torch import load_file, save_file, safe_open
from pathlib import Path

base = "persona_data/model_inits"
cache_dir = "analysis/cache"
os.makedirs(cache_dir, exist_ok=True)

stack_cache = os.path.join(cache_dir, "stacked_personas.safetensors")
pca_cache = os.path.join(cache_dir, "pca_components.safetensors")
pc_sim_cache = os.path.join(cache_dir, "pc_cosine_similarity.safetensors")


def build_stacked_vectors(key: str, files):
    vectors = []
    for path in files:
        with safe_open(path, framework="pt") as f:
            vectors.append(
                f.get_tensor(key).flatten()
            )

    stack = torch.stack(vectors)
    mean_vector = stack.mean(dim=0)
    
    return stack, mean_vector

def load_build_stacked_prompt_tensors():
    
    base = "persona_data/model_inits"
    cache_dir = "analysis/cache"
    os.makedirs(cache_dir, exist_ok=True)

    stack_cache = os.path.join(cache_dir, "stacked_personas.safetensors")
    
    # ------------------------------------------------
    # 1. Load or build stacked prompt persona tensors
    # ------------------------------------------------
    if os.path.exists(stack_cache):
        print("Loading cached stacked personas")
        data = load_file(stack_cache)
        prompt_stack = data["prompt_persona_vector"]
        mean_vector = data["mean_vector"]
    else:
        print("Building stacked persona tensors")

        folders = glob.glob(os.path.join(base, "*_persona_initialization"))
        folders = [f for f in folders if os.path.basename(f)[0].isupper()]

        files = []
        for folder in folders:
            files.extend(glob.glob(os.path.join(folder, "*.safetensors")))

        # prompt_vectors = []
        # for path in files:
        #     with safe_open(path, framework="pt") as f:
        #         prompt_vectors.append(
        #             f.get_tensor("prompt_persona_vector").flatten()
        #         )

        # prompt_stack = torch.stack(prompt_vectors)
        # mean_vector = prompt_stack.mean(dim=0)
        
        prompt_stack, mean_vector_prompt = build_stacked_vectors(key="prompt_persona_vector", files=files)
        response_stack, mean_vector_response = build_stacked_vectors(key="response_persona_vector", files=files)

        save_file(
            {
                "prompt_persona_vector": mean_vector_prompt.contiguous(),
                "response_persona_vector": mean_vector_response.contiguous(),
                "all_layers_response_persona_vector": torch.empty(0),
                "prompt_persona_vector_stacked": prompt_stack.contiguous(),
                "response_persona_vector_stacked": response_stack.contiguous()
            },
            stack_cache,
            metadata={
                "target_model_id": "allenai/Olmo-3-7B-Instruct",
                "concept": "Mean",
                "layer_steering": str(16)
            }
        )


def load_safetensors_file(filepath: str):
    
    # prompt_vec = f.get_tensor("prompt_persona_vector")
    # response_vec = f.get_tensor("response_persona_vector")
    # all_layers_vec = f.get_tensor("all_layers_response_persona_vector")
    
    path = Path(filepath)
    if not path.exists():
        print(f"File not found: {filepath}")
        return

    print(f"Loading: {filepath}")
    with safe_open(filepath, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

if __name__ == "__main__":
    # Example usage: change this path to your safetensors file
    # safetensors_path = "persona_data/model_inits/Mean_persona_initialization/allenai__Olmo-3-7B-Instruct.safetensors"
    # load_safetensors_file(safetensors_path)
    load_build_stacked_prompt_tensors()