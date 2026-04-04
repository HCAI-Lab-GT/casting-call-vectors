#!/bin/bash
#SBATCH --job-name=create_gold_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --gres=none
#SBATCH --output=logs/slurm/create_gold_standard_prompts/create_gold_cpu-%j.out
#SBATCH --error=logs/slurm/create_gold_standard_prompts/create_gold_cpu-%j.err

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "This script must be submitted to SLURM via sbatch"
    exit 1
fi

ROLE=""
BACKEND="openai"
MODEL="openai/gpt-4.1-mini"
NUM_DIALOGUE_PROMPTS=5
TEMPERATURE=0.7
MAX_NEW_TOKENS=2048
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role|-r)
            ROLE="$2"
            shift 2
            ;;
        --backend|-b)
            BACKEND="$2"
            shift 2
            ;;
        --model|-m)
            MODEL="$2"
            shift 2
            ;;
        --num-dialogue-prompts)
            NUM_DIALOGUE_PROMPTS="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            if [ -z "$ROLE" ]; then
                ROLE="$1"
                shift
            else
                echo "Unknown argument: $1" >&2
                exit 1
            fi
            ;;
    esac
done

RUN_DIR="./slurm/create_gold_standard_prompts/create_gold_cpu-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/stdout.log")
exec 2> >(tee -a "$RUN_DIR/stderr.log" >&2)

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$RUN_DIR"
fi

if [ -z "$ROLE" ]; then
    echo "Usage: sbatch scripts/cluster/slurm/create_gold_standard_prompts_cpu.sbatch --role <role_name>"
    echo "   or: sbatch scripts/cluster/slurm/create_gold_standard_prompts_cpu.sbatch <role_name>"
    echo "Exactly one role is expected per job."
    exit 1
fi

echo "=== CLI ARGS ==="
echo "ROLE: $ROLE"
echo "BACKEND: $BACKEND"
echo "MODEL: $MODEL"
echo "NUM_DIALOGUE_PROMPTS: $NUM_DIALOGUE_PROMPTS"
echo "TEMPERATURE: $TEMPERATURE"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "==============="

# Some clusters do not expose the `module` function in non-login batch shells.
if ! command -v module >/dev/null 2>&1; then
    for init_file in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
        if [ -f "$init_file" ]; then
            # shellcheck disable=SC1090
            source "$init_file"
            break
        fi
    done
fi

if command -v module >/dev/null 2>&1; then
    module load uv || true
fi

UV_BIN=""
if command -v uv >/dev/null 2>&1; then
    UV_BIN="uv"
elif [ -x ".venv/bin/uv" ]; then
    UV_BIN=".venv/bin/uv"
else
    echo "Error: uv command not found in PATH and .venv/bin/uv does not exist." >&2
    exit 127
fi

source .venv/bin/activate
"$UV_BIN" sync

srun "$UV_BIN" run python scripts/patch_scripts/create_gold_standard_prompts.py \
    --roles "$ROLE" \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --num_dialogue_prompts "$NUM_DIALOGUE_PROMPTS" \
    --temperature "$TEMPERATURE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite
