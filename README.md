under construction

## Repository Structure

```
persona-vectors/
├── pyproject.toml
├── uv.lock
├── CLAUDE.md
│
├── src/pvx/                              # Core library (installed as `pvx`)
│   ├── abstraction/
│   │   ├── judges/                       # Judge abstract base classes
│   │   └── pvx_models/                   # Model abstract base classes
│   ├── implementations/
│   │   ├── base/                         # Base persona model impl
│   │   ├── judges/                       # Judge implementations
│   │   ├── riasec/                       # RIASEC personality impl
│   │   ├── roles/                        # Role-based persona model
│   │   ├── roles_layers/                 # Role + layer persona model
│   │   ├── roles_optimized/              # Optimized role model
│   │   ├── steer_only/                   # Steering-only impl
│   │   └── trait_coverage/               # Trait coverage impl
│   ├── experiments/
│   │   └── identity_experiments/         # Identity experiment framework
│   ├── tasks/
│   │   ├── boardgame_qa/                 # BoardgameQA benchmark
│   │   └── worfbench/                    # WorfBench benchmark
│   ├── utils/
│   │   └── role_utils/
│   ├── pvx_models/
│   │   └── inspect_modelapi/
│   └── judges/
│
├── scripts/
│   ├── helpers.py
│   ├── prefetch.py
│   ├── analysis/                         # Analysis scripts
│   ├── evaluation/
│   │   ├── empirical/                    # Empirical evaluation (NLP, judge-gold-standard)
│   │   ├── geometry/                     # Geometric analysis (combination, full)
│   │   └── old_psycho/                   # Legacy psychometric evaluation
│   ├── patch_scripts/                    # Data repair / backfill scripts
│   └── cluster/
│       ├── slurm/                        # SLURM batch job scripts (.sh, .sbatch)
│       ├── mats/                         # MATS cluster scripts
│       └── modal_batching/               # Modal cloud compute wrappers
│
├── configs/
│   ├── models.yaml                       # Model presets (names, ids, generation params)
│   ├── runs.yaml                         # Run presets (tasks, models, limits, log dirs)
│   ├── psychometric_runs.yaml
│   ├── riasec.yaml
│   ├── role_list.json
│   ├── trait_coverage_list.json
│   ├── validation_questions.jsonl
│   ├── scoring/
│   ├── role_list_splits/
│   ├── role_list_splits_aa/
│   └── role_list_judge_splits/
│
├── experiment_data/                      # Experiment outputs (CSVs)
│   ├── gold_prompt_experiments/          # Main empirical results
│   ├── pairwise_judge_experiments/
│   ├── pairwise_judge_experiments_gpt/
│   └── experiment_missing_data/          # Backfill tracking (less relevant)
│
├── persona_data/                         # Persona artifacts (vectors, datasets, labels)
│   ├── assistant-axis/
│   ├── gold_labels_prompts_dataset/
│   ├── gold_standard_prompts/
│   ├── model_inits/
│   ├── onet_datasets/
│   ├── riasec_datasets/
│   ├── role_datasets/
│   ├── role_datasets_combination/
│   ├── trait_datasets/
│   └── trait_inits/
│
├── analysis/                             # Analysis results & figures
│   ├── empirical/                        # Empirical results and figures
│   │   ├── comparison_plots/
│   │   ├── gold_prompt_role_graphs/
│   │   ├── nlp_experiments/
│   │   ├── pairwise_graphs/
│   │   └── paper_figures/
│   ├── geometry/                         # Geometry results and figures
│   │   └── figures/
│   └── psycho_graphs/                    # Legacy
│
├── tests/                                # Currently unused
│   ├── unit/
│   └── integration/
│
├── docs/                                 # Claude context docs
├── frontend/
└── logs/
```
