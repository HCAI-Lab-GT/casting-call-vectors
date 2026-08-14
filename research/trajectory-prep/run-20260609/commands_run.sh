#!/usr/bin/env bash
# commands_run.sh — every command actually executed during run
# persona_trajectory_20260609T052641Z (Apple M1 Max / macOS, no CUDA).
# This file is a faithful transcript for resume/reproduction, NOT meant to be
# run top-to-bottom blindly. Paths are absolute to the session working dir.
set -euo pipefail
PACK=/Users/glenn/dev/persona_trajectory
RUN=$PACK/runs/persona_trajectory_20260609T052641Z

############################################################
# 0. System probe (hardware / toolchain reality check)
############################################################
uname -a; sw_vers; sysctl -n machdep.cpu.brand_string; sysctl -n hw.memsize
which nvidia-smi || echo "nvidia-smi NOT FOUND"      # -> not found (no CUDA)
system_profiler SPDisplaysDataType | grep -A3 "Chipset Model"   # -> Apple M1 Max, 24 GPU cores
python3 -c "import shutil;print(shutil.disk_usage('/Users/glenn'))"   # -> ~26.5 GB free

############################################################
# 1. Clone official + upstream repos (read-only, shallow)
############################################################
mkdir -p "$PACK/work" && cd "$PACK/work"
git clone --depth 1 https://github.com/epfl-dlab/pretraining_persona.git pretraining_persona
git clone --depth 1 https://github.com/safety-research/persona_vectors.git upstream_persona_vectors
git -C pretraining_persona rev-parse HEAD          # c79339342e91e9ba07308d730bed4286ade633bc
git -C upstream_persona_vectors rev-parse HEAD      # b8e0f044fe2410a6fad579f38324f03f13b4e917

############################################################
# 2. Inspect pipeline (grounds the cost model) — read-only
############################################################
cd "$PACK/work/pretraining_persona"
cat requirements.txt .env.example
cat pipeline/checkpoint_grids.sh pipeline/checkpoint_sweep.sh \
    pipeline/generate_vec.sh pipeline/instructed_continuations.sh
sed -n '1,320p' README.md
# Confirm vLLM/unsloth are guarded / out of the generation import chain:
grep -rnE "from vllm|import vllm|from unsloth|import unsloth" source/
grep -rnE "_is_mps_available|prefer_transformers|use_transformers" source/eval_persona.py source/model_utils.py
# Count workload (20 questions x 5 instruction pairs per trait):
python3 -c "import json;d=json.load(open('data/trait_data_extract/humorous_character_neutral_q.json'));print(len(d['questions']),len(d['instruction']))"

############################################################
# 3. HF metadata (no download): model is public, 1487 branches
############################################################
curl -s https://huggingface.co/api/models/allenai/Olmo-3-1025-7B | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['gated'],len(d['siblings']))"
curl -s https://huggingface.co/api/models/allenai/Olmo-3-1025-7B/refs | python3 -c "import sys,json;print(len(json.load(sys.stdin)['branches']))"
curl -s https://huggingface.co/allenai/Olmo-3-1025-7B/raw/main/config.json   # olmo3 32L/4096h

############################################################
# 4. Build CUDA-free MPS venv (subset of requirements.txt)
#    -> full script: logs/bootstrap_mps_env.sh ; log: logs/bootstrap_mps_env.log
############################################################
cd "$PACK/work/pretraining_persona"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python \
  torch==2.9.0 transformers==4.57.3 accelerate==1.7.0 datasets==3.6.0 \
  pandas==2.3.1 scikit-learn==1.5.2 scipy==1.17.1 fire==0.7.0 PyYAML==6.0.2 \
  tqdm==4.67.1 openai==1.99.1 backoff==2.2.1 pydantic==2.12 \
  safetensors sentencepiece protobuf
# (deliberately NOT installed: vllm, unsloth, bitsandbytes — CUDA-only)
.venv/bin/python -c "import torch;print('mps',torch.backends.mps.is_available())"  # True
uv pip freeze --python .venv/bin/python > "$RUN/logs/pip_freeze.txt"

############################################################
# 5. Generation-only smoke on MPS (REAL repo CLI, skip_judge)
#    -> full script: logs/download_and_smoke.sh ; log: logs/download_and_smoke.log
#    Downloads ONE checkpoint (stage1-step5000, ~15 GB) then:
############################################################
cd "$PACK/work/pretraining_persona"
PY=.venv/bin/python
# download one checkpoint
$PY -c "from huggingface_hub import snapshot_download; \
  snapshot_download('allenai/Olmo-3-1025-7B', revision='stage1-step5000', \
  allow_patterns=['*.json','*.safetensors','tokenizer*','*.model','*.txt'])"
# generation-only extraction, pos + neg, 2 questions, n=1, no judge:
for POL in pos neg; do
  CUDA_VISIBLE_DEVICES="" $PY -m source.eval_persona \
    --model allenai/Olmo-3-1025-7B --revision stage1-step5000 \
    --trait humorous_character_neutral_q \
    --output_path data/model_responses/extract/Olmo-3-1025-7B/stage1-step5000/humorous_character_neutral_q_${POL}_instruct.csv \
    --persona_instruction_type $POL --version extract \
    --n_per_question 1 --max_tokens 64 --repetition_penalty 1.1 \
    --generation_batch_size 8 --batch_process True \
    --skip_judge True --max_questions 2 --overwrite True
done
# activation-capture probe (proves generate_vec forward pass works on MPS):
$PY -c "import torch;from source.model_utils import load_model; \
  m,t=load_model('allenai/Olmo-3-1025-7B',revision='stage1-step5000'); \
  i=t(['If Alex was asked to explain how a computer works'],return_tensors='pt',padding=True); \
  i={k:v.to(m.device) for k,v in i.items()}; \
  o=m(**i,output_hidden_states=True); print(len(o.hidden_states),tuple(o.hidden_states[-1].shape))"

############################################################
# NOT RUN (blocked) — would require, and are documented in result_card.md:
#   - judge/filtering + vector extraction  : needs OPENAI_API_KEY or DEEPSEEK_API_KEY
#   - second checkpoint / full grid         : needs >30 GB / >=240 GB disk (have ~26.5)
#   - any CUDA path (vLLM)                  : needs an NVIDIA GPU
############################################################

############################################################
# 6. REAL GPU run on GT PACE ICE (resolves the no-GPU blocker)
#    Full scripts: logs/ice_setup.sh, logs/ice_smoke.sbatch
#    Job log: logs/ice_smoke_5378587.out ; samples: gpu_samples.csv (real A100)
############################################################
# SSH is preconfigured as `pace-ice` (login-ice.pace.gatech.edu, user gmatlin3,
# account ic). A live ControlMaster socket made it non-interactive this session.
ssh pace-ice 'mkdir -p ~/scratch/persona_trajectory && cd $_ && \
  git clone --depth 1 https://github.com/epfl-dlab/pretraining_persona.git'
# Build CUDA venv + download checkpoint on the LOGIN node (has internet).
# IMPORTANT: load the python module in the PARENT login shell so the script
# inherits it (non-login `bash script.sh` does NOT initialize lmod):
scp logs/ice_setup.sh   pace-ice:scratch/persona_trajectory/
scp logs/ice_smoke.sbatch pace-ice:scratch/persona_trajectory/pretraining_persona/
ssh pace-ice "nohup bash -lc 'module purge && module load python/3.11.9 && \
  cd ~/scratch/persona_trajectory && bash ice_setup.sh' \
  > ~/scratch/persona_trajectory/logs/ice_setup.log 2>&1 &"
# ice_setup.sh: venv (py3.11) -> pip install vllm==0.11.2 FIRST (numpy pin
# dropped; conflicts with vllm), then transformers==4.57.3 + gen-chain deps;
# snapshot_download(allenai/Olmo-3-1025-7B, revision=stage1-step5000).
# Submit the A100 smoke (no -A, no -p; auto-routed):
ssh pace-ice 'cd ~/scratch/persona_trajectory/pretraining_persona && sbatch ice_smoke.sbatch'
# ice_smoke.sbatch key env (the fixes that made it work on a quota'd, air-gapped node):
#   export HOME=$TMPDIR/.../fakehome      # all ~/.cache writers -> node-local
#   export XDG_CACHE_HOME / VLLM_CACHE_ROOT / TRITON_CACHE_DIR / TORCHINDUCTOR_CACHE_DIR=$TMPDIR/...
#   export HF_HOME=~/scratch/.../hf_cache ; HF_HUB_OFFLINE=1 ; VLLM_ENFORCE_EAGER=1 ; VLLM_ATTENTION_BACKEND=FLASH_ATTN
#   MODEL=<local snapshot dir>            # load by path, not repo-id+branch (offline-safe)
#   --gres=gpu:a100:1 ; nvidia-smi sampling -> gpu_samples.csv
# Monitor + pull results:
ssh pace-ice 'sacct -j 5378587 -X -o State,Elapsed,AllocTRES'
scp pace-ice:scratch/persona_trajectory/run_out/gpu_samples.csv ./gpu_samples.csv
############################################################

############################################################
# 7. CLOSE THE JUDGE BLOCKER with an OPEN-WEIGHT judge (no API key, no egress)
#    Full script: logs/ice_judged_smoke.sbatch ; result: ice_judged_meta.json
############################################################
# Small env-gated patch to source/judge.py: open models tokenize numbers per-digit,
# so the single-token logprob trick fails -> JUDGE_NUMERIC_TEXT=1 generates+parses.
scp work/pretraining_persona/source/judge.py pace-ice:scratch/persona_trajectory/pretraining_persona/source/judge.py
# Download an open instruct model as judge:
ssh pace-ice 'HF_HOME=~/scratch/persona_trajectory/hf_cache ~/scratch/persona_trajectory/venv/bin/python -c \
  "from huggingface_hub import snapshot_download; snapshot_download(\"Qwen/Qwen2.5-7B-Instruct\")"'
# 2-A100 job: GPU1 serves the judge (vLLM OpenAI server), GPU0 runs the pipeline:
#   CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
#       --model <qwen_local> --served-model-name judge --port 8000 --enforce-eager &
#   export OPENAI_BASE_URL=http://127.0.0.1:8000/v1 OPENAI_API_KEY=sk-local-dummy JUDGE_NUMERIC_TEXT=1
#   CUDA_VISIBLE_DEVICES=0 python -m source.eval_persona ... --judge_model judge --version extract  # judged
#   CUDA_VISIBLE_DEVICES=0 python -m source.generate_vec ... --threshold 50 --overwrite             # vector!
#   CUDA_VISIBLE_DEVICES=0 python -m source.eval_persona ... --version eval                          # baseline judged
#   CUDA_VISIBLE_DEVICES=0 python -m source.eval_persona ... --vector_path <vec> --coef 0.3 --layer 20  # steered judged
ssh pace-ice 'cd ~/scratch/persona_trajectory/pretraining_persona && sbatch ice_judged_smoke.sbatch'
# RESULT (job 5378804): vector built from 10 Qwen-judged pairs; Δτ = +1.2
#   (baseline humor 14.85 -> steered 16.05). Judge cost $0 (free 2nd ICE GPU).
############################################################
