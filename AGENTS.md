# casting-call-vectors: agent rules

Project-specific work lives in subfolders with their own notes: `conceptors/` (read `conceptors/CLAUDE.md` and
`conceptors/HANDOFF.md` first) and `docs/`.

## Storage on PACE ICE: where everything lives

**Everything this repo puts on PACE goes on the CS 7634 course volume.** That covers data, models, logs, code
releases, the Python environment and caches. Scratch and home are off limits for this repo, because scratch has a
300 GB and 1,000,000-file quota and home has 30 GB. On 5 Nov 2026 one 12 GB code checkout per deploy filled scratch
to 300.2 GB and blocked a deploy while jobs waited. This rule overrides the global default that puts code and
environments on scratch.

The root is `/storage/ice-shared/cs7634/staff/conceptors` (`$CONCEPTORS_DATA`). The volume holds 10 TB, with about
7 TB free on 5 Nov 2026.

| What | Path under `/storage/ice-shared/cs7634/staff/conceptors/` | Set by |
|---|---|---|
| Run outputs (activations, fits, texts, judgments, `derived.json`) | `runs/<run>/` | job scripts, via `$CONCEPTORS_DATA` |
| Slurm logs | `logs/` | `sbatch -o "$CONCEPTORS_DATA/logs/%x-%j.out"` |
| Model weights and the HF cache | `hf_cache/` | `HF_HOME` |
| Assistant Axis clone | `assistant-axis/` | `AA_DIR` |
| Code releases (one checkout per commit) | `releases/casting-call-vectors/<full sha>/` | `pace-code-release --release-root` |
| Python venv | `venvs/casting-call-vectors/` | `UV_PROJECT_ENVIRONMENT` |
| uv cache | `uv-cache/` | `UV_CACHE_DIR` |

The other places a job touches:

- **Per-job temporary files** (vLLM and torch compile caches, staging): node-local `$TMPDIR`, which is created and
  wiped per job. `judge_job.sh` already makes a private cache there, because a shared NFS compile cache hit stale
  handles.
- **Secrets** (the HF token, SSH keys): home only, at `~/.cache/huggingface/token` with mode 600. They never go on
  the course volume, because its top level is world-writable and `staff/` is readable by other users.
- **Your laptop:** code and small copies of results for the deck. `*.npz` activation dumps stay out of git.

How the scripts keep to this:

- `conceptors/pace/env.sh` is the single source of these paths. Every job script sources it, and none hard-codes a
  path. Change a location there, never in a launcher.
- `conceptors/pace/setup_env.sh` builds the venv, but only when `uv.lock` changes, because the venv is shared by
  every release.
- The volume's top level is world-writable with no sticky bit, so env.sh refuses to run if any folder from `staff/`
  down to the venv, the HF cache or the working release isn't yours or can be written by others. Keep `umask 022`,
  and never `chmod g+w` or `o+w` anything under `staff/`.
- Deploy a commit with this command:

  ```bash
  ~/dotfiles/scripts/pace-code-release --host pace-ice-agent \
    --repo-url https://github.com/HCAI-Lab-GT/casting-call-vectors.git --revision <full sha> \
    --release-root /storage/ice-shared/cs7634/staff/conceptors/releases/casting-call-vectors \
    --tracked-identity uv.lock --tracked-identity conceptors
  ```

- Each release is a full clone of about 12 GB, because data is tracked in git. Once no queued or running job uses a
  release, prune it: `squeue -u gmatlin3 -h -o %Z` lists each job's work directory, and every release a job still
  uses must stay.

Before any long or unattended run, check the quotas with `pace-quota` on the login node. Then confirm that the run's
outputs, logs and caches all resolve under `$CONCEPTORS_DATA`.
