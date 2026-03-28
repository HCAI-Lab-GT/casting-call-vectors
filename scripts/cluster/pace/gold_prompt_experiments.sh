#!/bin/bash
#SBATCH --job-name=gold_prompt_experiments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/gold_prompt_experiments_%j.out
#SBATCH --error=logs/slurm/gold_prompt_experiments_%j.err

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch slurm/gold_prompt_experiments.sh \
    --role chemist \
    --alpha 2.5 \
    --questions_file ./configs/validation_questions.jsonl \
    --save_dir ./experiment_data/gold_prompt_experiments/Steered_Nonsteered_GoldStandard_Comparison.csv

Primary arguments:
  -r, --role            Role name (single role), e.g. chemist
  -a, --alpha           Single alpha value passed to --alphas in Python script
  -q, --questions_file  JSONL file with validation questions
  -d, --save_dir        Output directory or explicit CSV filepath

Optional arguments:
  -m, --model           Target model ID (default: allenai/Olmo-3-7B-Instruct)
  -l, --layers          Space-separated layers (default: 16)
  -s, --sample_counts   Space-separated sample counts (default: 40)
  -t, --temperatures    Space-separated temperatures (default: 0.2)
  -n, --max_new_tokens  Max generation tokens (default: 2000)
  -f, --safetensors_dir Safetensors directory
  -g, --gold_prompts_dir Gold prompts directory
EOF
}

# Defaults
MODEL="allenai/Olmo-3-7B-Instruct"
ROLE=""
ALPHA=""
QUESTIONS_FILE="./configs/validation_questions.jsonl"
SAVE_DIR="./experiment_data/gold_prompt_experiments/Steered_Nonsteered_GoldStandard_Comparison.csv"
LAYERS=(16)
SAMPLE_COUNTS=(20 40 50)
TEMPERATURES=(0.2)
MAX_NEW_TOKENS=2000
SAFETENSORS_DIR="./persona_data/model_layer_inits/"
GOLD_PROMPTS_DIR="./persona_data/gold_labels_prompts_dataset"

# Parse CLI args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--model)
      MODEL="$2"; shift 2
      ;;
    -r|--role)
      ROLE="$2"; shift 2
      ;;
    -a|--alpha)
      ALPHA="$2"; shift 2
      ;;
    -q|--questions_file|--questions-file)
      QUESTIONS_FILE="$2"; shift 2
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

if [[ -z "$ROLE" ]]; then
  echo "Error: --role is required" >&2
  usage
  exit 1
fi

if [[ -z "$ALPHA" ]]; then
  echo "Error: --alpha is required" >&2
  usage
  exit 1
fi

echo "MODEL: $MODEL"
echo "ROLE: $ROLE"
echo "ALPHA: $ALPHA"
echo "QUESTIONS_FILE: $QUESTIONS_FILE"
echo "SAVE_DIR: $SAVE_DIR"
echo "LAYERS: ${LAYERS[*]}"
echo "SAMPLE_COUNTS: ${SAMPLE_COUNTS[*]}"
echo "TEMPERATURES: ${TEMPERATURES[*]}"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "SAFETENSORS_DIR: $SAFETENSORS_DIR"
echo "GOLD_PROMPTS_DIR: $GOLD_PROMPTS_DIR"

cd /workspace/personality-vectors

# Optional env setup
if command -v module >/dev/null 2>&1; then
  module load uv || true
fi
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

uv sync

srun uv run python src/pvx/experiments/gold_prompt_experiments.py \
  --model "$MODEL" \
  --roles "$ROLE" \
  --alphas "$ALPHA" \
  --questions_file "$QUESTIONS_FILE" \
  --save_dir "$SAVE_DIR" \
  --layers "${LAYERS[@]}" \
  --sample_counts "${SAMPLE_COUNTS[@]}" \
  --temperatures "${TEMPERATURES[@]}" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --safetensors_dir "$SAFETENSORS_DIR" \
  --gold_prompts_dir "$GOLD_PROMPTS_DIR"
