#!/bin/bash
#SBATCH --job-name=gold_prompt_experiments_aa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/gold_prompt_experiments_aa_%j.out
#SBATCH --error=logs/slurm/gold_prompt_experiments_aa_%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/cluster/slurm/gold_prompt_experiments_assistant_axis.sh \
    -r celebrity \
    --sample_counts 50 \
    --assistant_axis_alpha 2.5 \
    --assistant_axis_layer 16

Defaults in this script already match:
  -r celebrity --sample_counts 50 --assistant_axis_alpha 2.5 --assistant_axis_layer 16

Primary arguments:
  -r, --roles                     Space-separated role names (default: celebrity)
  -s, --sample_counts             Space-separated sample counts (default: 20 50)
      --assistant_axis_alpha      Assistant Axis alpha for assistant-axis response (default: 2.5)
      --assistant_axis_layer      Assistant Axis layer argument (default: 16)

Optional arguments:
  -m, --model                     Target model ID
  -a, --alphas                    Space-separated steering alphas
  -q, --questions_file            JSONL file with validation questions
      --question                  Optional single question override
  -d, --save_dir                  Results directory
  -l, --layers                    Space-separated layers for steered model
  -t, --temperatures              Space-separated temperatures
  -n, --max_new_tokens            Max generation tokens
  -f, --safetensors_dir           Safetensors directory for steered model
      --pt_dir                    Directory containing Assistant Axis .pt files
  -g, --gold_prompts_dir          Gold prompts directory
EOF
}

# Defaults
MODEL="allenai/Olmo-3-7B-Instruct"
ROLES=("celebrity")
ALPHAS=(1 1.5 2 2.5)
QUESTIONS_FILE="./configs/validation_questions.jsonl"
QUESTION=""
SAVE_DIR="./experiment_data/gold_prompt_experiments/"
LAYERS=(16)
SAMPLE_COUNTS=(20 50)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
SAFETENSORS_DIR="./persona_data/model_inits/"
# PT_DIR="../assistant-axis-pvx/outputs/olmo-3-7b-instruct/vectors/"
PT_DIR="persona_data/assistant-axis/olmo-3-7b-instruct/vectors/"
ASSISTANT_AXIS_ALPHA="2.5"
ASSISTANT_AXIS_LAYER="16"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="$2"; shift 2
      ;;
    -r|--roles)
      shift
      ROLES=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ROLES+=("$1")
        shift
      done
      ;;
    -a|--alphas)
      shift
      ALPHAS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        ALPHAS+=("$1")
        shift
      done
      ;;
    -q|--questions_file|--questions-file)
      QUESTIONS_FILE="$2"; shift 2
      ;;
    --question)
      QUESTION="$2"; shift 2
      ;;
    -d|--save_dir|--save-dir)
      SAVE_DIR="$2"; shift 2
      ;;
    -l|--layers)
      shift
      LAYERS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        LAYERS+=("$1")
        shift
      done
      ;;
    -s|--sample_counts|--sample-counts)
      shift
      SAMPLE_COUNTS=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        SAMPLE_COUNTS+=("$1")
        shift
      done
      ;;
    -t|--temperatures)
      shift
      TEMPERATURES=()
      while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
        TEMPERATURES+=("$1")
        shift
      done
      ;;
    -n|--max_new_tokens)
      MAX_NEW_TOKENS="$2"; shift 2
      ;;
    -f|--safetensors_dir|--safetensors-dir)
      SAFETENSORS_DIR="$2"; shift 2
      ;;
    --pt_dir|--pt-dir)
      PT_DIR="$2"; shift 2
      ;;
    --assistant_axis_alpha|--assistant-axis-alpha)
      ASSISTANT_AXIS_ALPHA="$2"; shift 2
      ;;
    --assistant_axis_layer|--assistant-axis-layer)
      ASSISTANT_AXIS_LAYER="$2"; shift 2
      ;;
    -g|--gold_prompts_dir|--gold-prompts-dir)
      GOLD_PROMPTS_DIR="$2"; shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

echo "MODEL: $MODEL"
echo "ROLES: ${ROLES[*]}"
echo "ALPHAS: ${ALPHAS[*]}"
echo "QUESTIONS_FILE: $QUESTIONS_FILE"
echo "QUESTION: ${QUESTION:-<none>}"
echo "SAVE_DIR: $SAVE_DIR"
echo "LAYERS: ${LAYERS[*]}"
echo "SAMPLE_COUNTS: ${SAMPLE_COUNTS[*]}"
echo "TEMPERATURES: ${TEMPERATURES[*]}"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "SAFETENSORS_DIR: $SAFETENSORS_DIR"
echo "PT_DIR: $PT_DIR"
echo "ASSISTANT_AXIS_ALPHA: $ASSISTANT_AXIS_ALPHA"
echo "ASSISTANT_AXIS_LAYER: $ASSISTANT_AXIS_LAYER"
echo "GOLD_PROMPTS_DIR: $GOLD_PROMPTS_DIR"

cd /workspace/personality-vectors

if command -v module >/dev/null 2>&1; then
  module load uv || true
fi
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

uv sync

cmd=(
  srun uv run python src/pvx/experiments/gold_prompt_experiments.py
  --model "$MODEL"
  --roles "${ROLES[@]}"
  --alphas "${ALPHAS[@]}"
  --questions_file "$QUESTIONS_FILE"
  --save_dir "$SAVE_DIR"
  --layers "${LAYERS[@]}"
  --sample_counts "${SAMPLE_COUNTS[@]}"
  --temperatures "${TEMPERATURES[@]}"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --safetensors_dir "$SAFETENSORS_DIR"
  --pt_dir "$PT_DIR"
  --assistant_axis_alpha "$ASSISTANT_AXIS_ALPHA"
  --assistant_axis_layer "$ASSISTANT_AXIS_LAYER"
  --gold_prompts_dir "$GOLD_PROMPTS_DIR"
)

if [[ -n "$QUESTION" ]]; then
  cmd+=(--question "$QUESTION")
fi

"${cmd[@]}"