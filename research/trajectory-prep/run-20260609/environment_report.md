# Environment Report — Persona Trajectory smoke

- Run ID: `persona_trajectory_20260609T052641Z`
- Generated (UTC): 2026-06-09
- Operator: Claude Code (autonomous overnight run)

## TL;DR — the headline blocker

**This machine is an Apple M1 Max laptop (macOS, arm64). There is no NVIDIA
GPU and no CUDA.** The paper's pipeline targets "a single NVIDIA A100." The
deliverables that presuppose CUDA (GPU type, CUDA/driver versions, GPU-hours,
peak VRAM, `nvidia-smi` sampling) **cannot be measured here**. What *can* be
done on this hardware — and was — is a real generation-only smoke through the
repo's built-in Apple-Silicon (MPS / transformers) fallback, plus a fully
grounded GPU-hour/cost model for the eventual CUDA run. See `result_card.md`
and `gpu_cost_summary.json`.

## Hardware

| Field | Value |
|---|---|
| Machine | Apple M1 Max (`arm64`), host `ipsec-10-2-65-252.vpn.gatech.edu` (on GT VPN) |
| OS | macOS 26.3 (build 25D125); kernel Darwin 25.3.0 |
| CPU | Apple M1 Max |
| GPU | Apple M1 Max integrated, 24 GPU cores — **Metal / MPS only** |
| Accelerator API | Metal Performance Shaders (`torch.backends.mps.is_available() == True`) |
| RAM | 64 GiB unified memory (68,719,476,736 bytes), shared CPU/GPU |
| **NVIDIA GPU** | **none** (`nvidia-smi` not found) |
| **CUDA** | **none** |
| Disk (root volume) | 995 GB total, **~26.5 GB free** (see Disk constraint below) |

### "Peak VRAM" is not measurable here
On Apple Silicon there is no discrete VRAM; the GPU shares the 64 GiB unified
pool with the CPU. The `gpu_samples.csv` deliverable is therefore header-only:
the monitor (`scripts/03_monitor_command.sh`) records rows only when
`nvidia-smi` exists. A note row documents the absence.

### Disk constraint (a real cost-envelope fact)
~26.5 GB free. One OLMo-3-7B checkpoint ≈ 15 GB (3 safetensors shards).
- **One** checkpoint fits (with ~11 GB headroom). Used for the smoke.
- **Two** checkpoints (~30 GB) do **not** fit — the goal's "best overnight
  success" (2 checkpoints) is impossible on this disk.
- The **full 16-checkpoint OLMo-3 grid ≈ 240 GB** of model weights alone, before
  any generations — physically impossible here. The real run needs a host with
  ≫240 GB scratch in addition to a GPU.

## Software toolchain

| Tool | Version |
|---|---|
| System Python | 3.14.3 (Homebrew) — *too new for the repo's pinned wheels* |
| Run venv Python | **3.11.15** (fetched & created by `uv`) |
| uv | 0.10.8 |
| git | 2.53.0 |
| gh | 2.93.0 |

### Python env created for the smoke (CUDA-free subset of `requirements.txt`)
Created with `uv venv --python 3.11` at `work/pretraining_persona/.venv`.
Installed (pins match `requirements.txt` where they install on macOS):

| Package | Installed | Repo pin |
|---|---|---|
| torch | 2.9.0 (**MPS available: True**) | 2.9.0 |
| transformers | 4.57.3 | 4.57.3 |
| accelerate | 1.7.0 | 1.7.0 |
| datasets | 3.6.0 | 3.6.0 |
| pandas | 2.3.1 | 2.3.1 |
| numpy | 2.4.6 | 2.4.4 |
| scipy | 1.17.1 | 1.17.1 |
| scikit-learn | 1.5.2 | 1.5.2 |
| openai | 1.99.1 | 1.99.1 |
| fire | 0.7.0 | 0.7.0 |
| tokenizers | 0.22.2 | (via transformers) |
| safetensors | 0.7.0 | — |
| huggingface-hub | 0.36.2 | — |

Full freeze: `logs/pip_freeze.txt`.

### Deliberately **NOT** installed (the dependency-level blocker)
These are pinned in `requirements.txt` but have **no macOS-arm64 wheels** and are
CUDA-only; `pip install -r requirements.txt` therefore fails on this machine:

| Package | Pin | Why it can't install here |
|---|---|---|
| `vllm` | 0.11.2 | CUDA inference engine; no darwin-arm64 wheel |
| `unsloth` | 2025.5.9 | CUDA/Triton fine-tuning lib; no darwin-arm64 wheel |
| `bitsandbytes` | 0.45.5 | CUDA quantization kernels; no darwin-arm64 wheel |

**Why the smoke still runs without them:** the repo guards the vLLM import
(`source/eval_persona.py:13-22`, `_VLLM_AVAILABLE=False` on ImportError) and has
an explicit MPS path (`source/model_utils.py:60-80`, `_is_mps_available()`,
float16, `.to("mps")`). `source/utils.py` (the only unguarded `unsloth` import)
is **not** in the generation import chain. `checkpoint_sweep.sh` even sets
`PREFER_TRANSFORMERS=True` by default. Verified: all of
`source.{config,prompts,activation_steer,judge,model_utils,eval_persona,generate_vec}`
import successfully in the venv (see `logs/bootstrap_mps_env.log`).

## Repositories (cloned, read-only)

| Repo | URL | Commit | Date |
|---|---|---|---|
| Official | `github.com/epfl-dlab/pretraining_persona` | `c79339342e91e9ba07308d730bed4286ade633bc` | 2026-05-15 |
| Upstream | `github.com/safety-research/persona_vectors` | `b8e0f044fe2410a6fad579f38324f03f13b4e917` | 2026-04-22 |

Both cloned cleanly (official: 2,777 files). Layout matches the prep pack's
expectations (`analysis/ data/ pipeline/ source/ requirements.txt`).

## Model

| Field | Value |
|---|---|
| Model | `allenai/Olmo-3-1025-7B` (note: HF id uses `Olmo`, not `OLMo`) |
| License / gating | apache-2.0, **public (not gated)** — no HF token required |
| Architecture | `olmo3`: 32 layers, hidden 4096, 32 attn heads, 32 KV heads (no GQA), ctx 65,536, vocab 100,278 |
| Weights | 3 safetensors shards (~15 GB total) |
| Revisions on HF | **1,487 branches**; the 16-checkpoint grid (`stage1-step3000 … main`) is fully present |
| Smoke checkpoint | `stage1-step5000` (the goal's primary early checkpoint) |
| Persona-vector shape | `[33 × 4096]` (n_layers+1 hidden states × hidden_dim) |

## API keys present in environment
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `HF_TOKEN` — **all missing.**
Consequence: the LLM-as-judge stage (trait/coherence scoring, `threshold=50`
filtering) cannot run, so a *scientific* vector extraction is out of scope
tonight. The smoke uses `--skip_judge True` (generation-only) and the model is
public, so no key is needed for what was run.

## Hugging Face cache
Empty at session start. After the smoke it holds the single `stage1-step5000`
snapshot (~15 GB). Cache root: `~/.cache/huggingface`.

---

## ADDENDUM — second environment: GT PACE ICE (real NVIDIA A100)

The "no GPU" blocker above was resolved by moving to **Georgia Tech PACE ICE**
(the user has SSH configured as `pace-ice`). This is where the *measured* GPU
numbers come from (`ice_run_meta.json`, `gpu_samples.csv`).

| Field | Value |
|---|---|
| Cluster | GT PACE **ICE** (`login-ice.pace.gatech.edu`), account `ic` (instructional, **free**, no `-A`) |
| Submit | `sbatch` (no `-A`, no `-p` — auto-routed), `--gres=gpu:a100:1`, `--time=01:00:00` |
| Node / partition / QOS | `atl1-1-01-005-13-0` / `ice-gpu` / `coe-ice` |
| **GPU** | **NVIDIA A100 80GB PCIe**, driver **595.58.03**, CUDA compute cap **8.0** |
| Login Python | 3.9.21; **module `python/3.11.9`** used for the venv (spack) |
| CUDA modules | `cuda/12.1.1`, `cuda/12.6.1` (not needed — torch/vLLM wheels bundle CUDA 12.8) |
| Env (scratch venv) | `torch 2.9.0+cu128`, `vllm 0.11.2`, `transformers 4.57.3`, `accelerate 1.7.0` |
| Storage | home **30 GB (was 100% full)**; scratch **Lustre `/storage/ice1`, 4.6 PB free** (300 GB quota, ~89% used, near 1M-inode cap) |
| Cache handling | HOME full → redirected `$HOME` + all `~/.cache` writers to node-local `$TMPDIR` |

**Install on the target platform:** the repo's pinned `requirements.txt` is **not
pip-resolvable as-is** even on Linux+CUDA — `numpy==2.4.4` conflicts with
`vllm==0.11.2`. Working recipe: drop the `numpy` pin (and `unsloth`,
`bitsandbytes`, `google-genai`, none of which are in the gen chain), install
`vllm==0.11.2` first, then `transformers==4.57.3` + the rest. vLLM resolves the
`olmo3` config to its `Olmo2ForCausalLM` implementation.

**Air-gapped compute nodes:** ICE GPU nodes have **no internet**. Pre-download
the model on the login node into scratch, then load by **local snapshot path**
(not repo-id + branch) to avoid HF online branch→commit resolution under
`HF_HUB_OFFLINE`.

**Measured peak VRAM nuance:** OLMo-3-7B weights occupy **~14 GB**; vLLM with
`gpu_memory_utilization=0.85` *reserves* **~67 GB** of the 80 GB card for its KV
cache pool. The README's claim that "24 GB VRAM should be sufficient" is correct
for the **transformers** path (~15 GB observed); vLLM simply grabs more when it
is available.
