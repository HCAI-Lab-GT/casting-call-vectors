# TODOs

- DONE: swap ollama for HF/vLLM backends (see dataset/PersonaDataset.py)
- weights & bias for telemetry and remote logging
- DONE: create a readme
- tqdm and other logging visuals
- heartbeats and other live reporting when running
- DONE: yaml configs for models/personas etc 
- DONE: outputs of logging/evals/etc are stored in hierarchical folders to organize
- DONE: store metadata with the outputs to ensure we know what outputs have 
- all outputs from language models should be saved like chain of thoughts etc
- need to get a custom score/solver etc as needed for BigBench-Hard and BigBench-ExtraHard (now using inspect-evals/bbh and bbeh tasks; custom scoring still TBD)
- get things running on PACE
- DONE: evaluate BBEH with evalchemy vs inspect-ai (decision: stick with inspect-ai; evalchemy dropped)
