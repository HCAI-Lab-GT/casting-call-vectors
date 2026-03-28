"""
Modal wrapper for gold_prompt_experiments.py

Runs the gold prompt evaluation experiment on Modal's cloud infrastructure with GPU support.

Usage:
    # Default run (Lawyers role)
    modal run scripts/modal_batching/modal_gold_prompt_experiments.py

    # Fire-and-forget (like sbatch)
    modal run --detach scripts/modal_batching/modal_gold_prompt_experiments.py \
        --roles "Lawyers,Soldier"

    # Specific roles, layers, alphas
    modal run --detach scripts/modal_batching/modal_gold_prompt_experiments.py \
        --roles "Lawyers,Scholar" --layers "16" --alphas "2.5" --sample-counts "20,40,50"

    # With judge scoring enabled
    modal run --detach scripts/modal_batching/modal_gold_prompt_experiments.py \
        --roles "Lawyers" --judge-responses

    # Custom model / single question
    modal run --detach scripts/modal_batching/modal_gold_prompt_experiments.py \
        --model "allenai/Olmo-3-7B-Instruct" --question "What do you think about justice?"

    # Check status / logs
    modal app list
    modal app logs <app-id>
"""

import modal

# ---------------------------------------------------------------------------
# 1) IMAGE — GPU-capable container with torch, transformers, and pvx.
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.3.0",
        "transformers>=4.44.0",
        "accelerate>=1.0.0",
        "safetensors>=0.4.0",
        "pandas>=2.0",
        "pyyaml>=6.0.2",
        "hf-transfer>=0.1.9",
        "openai>=1.53.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "datasets>=3.1.0",
        "anthropic>=0.75.0",
        "sentence-transformers>=5.2.0",
        "tenacity>=8.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/data/hf_cache"})
    # Copy the local pvx package into the image so `from pvx import ...` works.
    .add_local_python_source("pvx")
)

# ---------------------------------------------------------------------------
# 2) VOLUME — persistent shared storage for experiment CSVs and model files.
#
#    First-time setup (run once from your local machine):
#        modal volume create personality-vectors-data
#        modal volume put personality-vectors-data \
#            ./experiment_data/gold_prompt_experiments /gold_prompt_experiments
#        modal volume put personality-vectors-data \
#            ./persona_data /persona_data
#        modal volume put personality-vectors-data \
#            ./configs /configs
# ---------------------------------------------------------------------------
volume = modal.Volume.from_name("personality-vectors-data", create_if_missing=True)

# ---------------------------------------------------------------------------
# 3) APP
# ---------------------------------------------------------------------------
app = modal.App("gold-prompt-experiments")


# ---------------------------------------------------------------------------
# 4) FUNCTION — runs in the cloud with a GPU for model inference.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu="H100:3",
    cpu=4,
    memory=16384,
    timeout=7200,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_dotenv()],
)
def run_gold_prompt_experiments(
    model: str = "allenai/Olmo-3-7B-Instruct",
    roles: list[str] | None = None,
    layers: list[int] | None = None,
    sample_counts: list[int] | None = None,
    alphas: list[float] | None = None,
    temperatures: list[float] | None = None,
    max_new_tokens: int = 2000,
    question: str | None = None,
    questions_file: str = "/data/configs/validation_questions.jsonl",
    safetensors_dir: str = "/data/persona_data/model_layer_inits/",
    pt_dir: str = "/data/persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
    assistant_axis_layer: int | None = None,
    assistant_axis_alpha: float = 2.5,
    save_dir: str = "/data/gold_prompt_experiments",
    gold_prompts_dir: str = "/data/persona_data/gold_labels_prompts_dataset",
    judge_responses: bool = False,
):
    """
    Runs inside a Modal container in the cloud with GPU access.
    """
    from pvx.experiments.gold_prompt_experiments import GoldPromptExperiments
    from pvx.implementations.roles_layers.role_layers_persona_model import RoleLayersPersonaModel

    roles = roles or ["Lawyers"]
    layers = layers or [16]
    sample_counts = sample_counts or [20, 40, 50]
    alphas = alphas or [2.5]
    temperatures = temperatures or [0.2]

    experiment = GoldPromptExperiments(
        persona_model=RoleLayersPersonaModel,
        target_model_id=model,
        safetensors_dir=safetensors_dir,
        assistant_axis_pt_dir=pt_dir,
        assistant_axis_layer=assistant_axis_layer if assistant_axis_layer is not None else layers[0],
        assistant_axis_alpha=assistant_axis_alpha,
        gold_prompts_dir=gold_prompts_dir,
        validation_questions_file=questions_file,
        save_dir=save_dir,
    )

    experiment.evaluate_model(
        questions=[question] if question else None,
        roles=roles,
        layers=layers,
        sample_counts=sample_counts,
        alphas=alphas,
        temperatures=temperatures,
        max_new_tokens=max_new_tokens,
        save_each=True,
        judge_responses=judge_responses,
    )

    volume.commit()


# ---------------------------------------------------------------------------
# 5) LOCAL ENTRYPOINT — CLI interface that runs on your machine.
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    model: str = "allenai/Olmo-3-7B-Instruct",
    roles: str = "",
    layers: str = "",
    sample_counts: str = "",
    alphas: str = "",
    temperatures: str = "",
    max_new_tokens: int = 2000,
    question: str = "",
    questions_file: str = "/data/configs/validation_questions.jsonl",
    safetensors_dir: str = "/data/persona_data/model_layer_inits/",
    pt_dir: str = "/data/persona_data/assistant-axis/olmo-3-7b-instruct/vectors/",
    assistant_axis_layer: int = 16,
    assistant_axis_alpha: float = 2.5,
    save_dir: str = "/data/gold_prompt_experiments",
    gold_prompts_dir: str = "/data/persona_data/gold_labels_prompts_dataset",
    judge_responses: bool = False,
):
    role_list = [r.strip() for r in roles.split(",") if r.strip()] or None
    layer_list = [int(l.strip()) for l in layers.split(",") if l.strip()] or None
    sample_count_list = [int(s.strip()) for s in sample_counts.split(",") if s.strip()] or None
    alpha_list = [float(a.strip()) for a in alphas.split(",") if a.strip()] or None
    temp_list = [float(t.strip()) for t in temperatures.split(",") if t.strip()] or None
    layer = assistant_axis_layer if assistant_axis_layer >= 0 else None

    run_gold_prompt_experiments.remote(
        model=model,
        roles=role_list,
        layers=layer_list,
        sample_counts=sample_count_list,
        alphas=alpha_list,
        temperatures=temp_list,
        max_new_tokens=max_new_tokens,
        question=question if question else None,
        questions_file=questions_file,
        safetensors_dir=safetensors_dir,
        pt_dir=pt_dir,
        assistant_axis_layer=layer,
        assistant_axis_alpha=assistant_axis_alpha,
        save_dir=save_dir,
        gold_prompts_dir=gold_prompts_dir,
        judge_responses=judge_responses,
    )
