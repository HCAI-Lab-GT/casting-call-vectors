# Casting Call Vectors

Code for **Role Steering of Language Models for Social Simulations** (Social Sim'26 Workshop at
COLM 2026, [arXiv:2608.00023](https://arxiv.org/abs/2608.00023)).

The paper extracts one judge-filtered contrastive mean-difference direction per role (a
*CastVector*) from OLMo-3-7B-Instruct at layer 16, steers with `h -> h + alpha * v` at
`alpha` in {1.0, 1.5, 2.0, 2.5}, and judges role-profile alignment on 228 role-agnostic
questions across 275 roles.

- **Code** (this repository, MIT): the extraction, steering and judging pipeline, the scoring
  prompts, and the scripts that recompute every figure and the numbers they check.
- **Data** ([HCAI-Lab-GT/casting-call-vectors](https://huggingface.co/datasets/HCAI-Lab-GT/casting-call-vectors)
  on Hugging Face, CC BY 4.0): the elicitation battery, the 275 CastVectors, the
  assistant-axis control vectors and rollouts, the prompted role references, and every judged
  response behind the paper's tables. The dataset card documents provenance and known
  irregularities in the source data.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Steering and extraction need a GPU that
fits a 7B model; the figure scripts run on CPU.

```bash
git clone https://github.com/HCAI-Lab-GT/casting-call-vectors.git
cd casting-call-vectors
uv sync
uv run python scripts/fetch_data.py      # downloads the dataset (about 860 MB) into ./data
```

Set `CASTVECTORS_DATA` to use a copy of the dataset somewhere else. API keys are only needed
to generate or judge new responses; copy `.env.example` to `.env` and fill in what you use.

## Reproduce the figures and checked numbers

The analysis steps write intermediate tables under `analysis/empirical/*/data/`, and the figure
scripts read them. Run the steps first, then the figures, from each script's own folder:

```bash
for step in rsa_geometry_behavior vector_magnitude_effect_on_stability \
            intrinsic_dimensionality behavioral_noise_floor; do
  (cd analysis/empirical/$step && uv run python run_analysis.py)
done
for fig in headline controllability appendix_controllability pairwise geometry; do
  (cd scripts/figures && uv run python ${fig}_figs.py)
done
```

Each figure script recomputes the statistics the paper reports and prints an `OK` or `FAIL`
line per check before writing its PDFs to `scripts/figures/out/`. The scripts follow the
paper's conventions, documented in their docstrings; the main one is that duplicate
(role, alpha, question) rows are dropped, keeping the first.

## Steer with a released CastVector

```python
import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "allenai/Olmo-3-7B-Instruct"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
v = load_file("data/vectors/castvectors_olmo3-7b-instruct_L16_n50.safetensors")["chef"]
alpha = 2.0

def add_vector(module, inputs, output):
    # Add to the last position of each forward pass: the final prompt token during
    # prefill, then each generated token, as the paper's runs did.
    hidden = output[0] if isinstance(output, tuple) else output
    hidden[:, -1, :] += alpha * v.to(hidden.device, hidden.dtype)
    return output

handle = model.model.layers[16].register_forward_hook(add_vector)  # output of block 16
msgs = [{"role": "user", "content": "How do you decide what to do first on a busy day?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=300, do_sample=True, temperature=0.2)
print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
handle.remove()
```

The paper's runs used the pipeline's own steering code (`AbstractPersonaModel.generate`),
which applies the same intervention. Extraction reads the same place, `hidden_states[17]`
(the output of decoder block 16): the last prompt token for prompt vectors, and the mean over
generated tokens for the response vectors that ship as CastVectors.

## Rerun the pipeline

Every stage is a plain entry point; `--help` lists all flags. Outputs go to `./outputs/`.

1. **CPU phase: generate and judge Q/A pairs.** OLMo answers each role's elicitation
   questions under the role's system prompts and under the default assistant, and a judge
   keeps fully role-playing positives and assistant-style negatives.

   ```bash
   uv run python src/pvx/implementations/roles_optimized/roles_optimized_cpu.py \
       --roles accountant activist --answer_model allenai/Olmo-3-7B-Instruct \
       --backend openai --target_pairs 50
   ```

2. **GPU phase: extract the role vectors** as layer-16 mean differences over the retained
   pairs.

   ```bash
   uv run python src/pvx/implementations/roles_optimized/roles_optimized_gpu.py \
       --activation_extraction_model allenai/Olmo-3-7B-Instruct \
       --roles accountant activist --layer 16 --counts 50
   ```

3. **Steering evaluation: steer, compare with the prompted reference, and judge.** Pass
   `--safetensors_dir data/vectors/castvectors_olmo3-7b-instruct_L16_n50.safetensors` to use
   the released vectors instead of ones you extracted.

   ```bash
   uv run python src/pvx/experiments/gold_prompt_experiments.py \
       --model allenai/Olmo-3-7B-Instruct --roles accountant \
       --layers 16 --sample_counts 50 --alphas 1 1.5 2 2.5 --temperatures 0.2 \
       --judge_responses
   ```

4. **Pairwise judging** of steered against prompted responses, with position swaps:
   `src/pvx/experiments/pairwise_judge_experiments.py`.

The judge prompts are in `src/pvx/implementations/judges/llm_as_judge.py` and
`src/pvx/utils/prompts.py`. The generation of role profiles and elicitation questions happened
before this pipeline; its outputs are the `battery/role_datasets/` files in the dataset.

## Intended use

The role profiles and vectors are model- and prompt-derived constructs, not empirical
representations of people who hold an occupation or social role. They are released for
research on transparent construction, auditing, and failure analysis of synthetic
role-conditioned agents. They should not be used for hiring, personnel assessment, worker
profiling, inference about real individuals, or other high-stakes decisions, and they should
not be used for deceptive impersonation.

## License and citation

Code: MIT (see `LICENSE`). Data: CC BY 4.0, with the attributions listed on the dataset card.
If you use either, please cite the paper (see `CITATION.cff`).
