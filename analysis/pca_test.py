import torch
import glob
import os
import torch.nn.functional as F
from safetensors.torch import load_file, save_file, safe_open

base = "persona_data/model_inits"
cache_dir = "analysis/cache"
os.makedirs(cache_dir, exist_ok=True)

stack_cache = os.path.join(cache_dir, "stacked_personas.safetensors")
pca_cache = os.path.join(cache_dir, "pca_components.safetensors")
pc_sim_cache = os.path.join(cache_dir, "pc_cosine_similarity.safetensors")


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

    prompt_vectors = []
    for path in files:
        with safe_open(path, framework="pt") as f:
            prompt_vectors.append(
                f.get_tensor("prompt_persona_vector").flatten()
            )

    prompt_stack = torch.stack(prompt_vectors)
    mean_vector = prompt_stack.mean(dim=0)

    # save_file(
    #     {
    #         "prompt_persona_vector": prompt_stack.contiguous(),
    #         "response_persona_vector": torch.empty(0),
    #         "all_layer_persona_vector": torch.empty(0),
    #         "mean_vector": mean_vector.contiguous(),
    #     },
    #     stack_cache,
    # )
    
    save_file(
        {
            "prompt_persona_vector": prompt_stack.contiguous(),
            "response_persona_vector": torch.empty(0),
            "all_layer_persona_vector": torch.empty(0),
            "mean_vector": mean_vector.contiguous(),
        },
        stack_cache,
    )



# ------------------------------------------------
# 2. Load or compute PCA components
# ------------------------------------------------
if os.path.exists(pca_cache):
    print("Loading cached PCA components")
    data = load_file(pca_cache)
    components = data["prompt_persona_vector"]
else:
    print("Computing PCA components")

    centered = prompt_stack - mean_vector
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

    components = Vt[:6].clone().contiguous()

    save_file(
        {
            "prompt_persona_vector": components,
            "response_persona_vector": torch.empty(0),
            "all_layer_persona_vector": torch.empty(0),
            "mean_vector": mean_vector.contiguous(),
        },
        pca_cache,
    )


# ------------------------------------------------
# 3. Load or compute cosine similarity of PCs
# ------------------------------------------------
if os.path.exists(pc_sim_cache):
    print("Loading cached PC cosine similarity")
    data = load_file(pc_sim_cache)
    pc_similarity = data["prompt_persona_vector"]
else:
    print("Computing PC cosine similarity")

    pc_normalized = F.normalize(components, dim=1)
    pc_similarity = pc_normalized @ pc_normalized.T  # (K, K)

    save_file(
        {
            "prompt_persona_vector": pc_similarity.contiguous(),
            "response_persona_vector": torch.empty(0),
            "all_layer_persona_vector": torch.empty(0),
            "mean_vector": torch.empty(0),
        },
        pc_sim_cache,
    )

print("PC similarity shape:", pc_similarity.shape)
print(pc_similarity)
