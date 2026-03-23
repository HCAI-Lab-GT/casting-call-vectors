"""
Design document for roles_optimized implementation.

This module describes the architecture, optimization strategy, and implementation
details of the optimized persona vector extraction pipeline.
"""

# Architecture Overview
# ====================
#
# The optimized implementation splits persona vector extraction into two phases:
#
# Phase 1 (CPU): roles_optimized_cpu.py
# - Load role dataset
# - Generate responses using target model
# - Judge responses using LLM judge
# - Save valid Q/A pairs to JSON
# - Skip roles that don't exist
#
# Phase 2 (GPU): roles_optimized_gpu.py
# - Load pre-generated Q/A responses from JSON
# - Pre-tokenize all messages (one-time cost)
# - Randomly shuffle Q/A pairs
# - Extract hidden state activations from target model
# - Compute contrastive persona vectors
# - Save to safetensors format
#
#
# Design Decisions
# ================
#
# 1. Skip Missing Roles (not fail)
#    - Rationale: When processing many roles, some may not have datasets yet
#    - Implementation: Try/except blocks with continue statements
#    - Benefit: Pipeline doesn't halt on missing data
#    - Example: Process 100 roles, skip 5 that don't exist, continue with 95
#
# 2. Two-Phase Architecture (CPU vs GPU)
#    - Rationale: Separates CPU-bound (inference + judging) from GPU-bound (activation extraction)
#    - Benefit: Enables SLURM job optimization with no GPU idle time
#    - Trade-off: Requires intermediate Q/A JSON storage
#    - Storage: ~100KB per 40 Q/A pairs (small price for GPU efficiency)
#
# 3. Pre-tokenization in GPU Phase
#    - Rationale: Tokenization is CPU-bound, do it before GPU inference
#    - Implementation: Cache all tokenized messages in dict
#    - Benefit: Avoids tokenization during GPU loop (reduces transfers)
#    - Cost: ~50MB RAM for 1000 cached messages
#
# 4. Random Shuffling Before Extraction
#    - Rationale: Ensures variance across accumulated samples
#    - Implementation: random.shuffle() before GPU loop
#    - Benefit: More representative persona vectors
#    - Note: Different from sequential processing
#
# 5. Accumulate Instead of Restart
#    - Rationale: If you extract 20 samples, then 20 more, why restart?
#    - Implementation: Store sums and counts, merge before finalizing
#    - Benefit: Supports incremental extraction, resume from interruption
#    - Example: Extract 20 pairs → Extract 20 more → 40-pair average
#
# 6. JSON Format for Q/A Storage
#    - Rationale: Human-readable, easy to inspect/debug, standard format
#    - Structure:
#      {
#        "positive": [
#          {
#            "pos_prompt": "...",
#            "question": "...",
#            "response": "...",
#            "score": 3
#          }
#        ],
#        "base": [
#          {
#            "question": "...",
#            "response": "...",
#            "score": 1
#          }
#        ]
#      }
#    - Benefit: Self-documenting, debuggable, easily filtered/processed
#
#
# Performance Characteristics
# ===========================
#
# CPU Phase (roles_optimized_cpu.py):
#   - Time per role: O(N_pairs * T_inference + N_pairs * T_judge)
#   - Memory: O(model_size) + O(tokenized_cache)
#   - Bottleneck: Model inference speed (limited by CPU/system)
#   - Parallelization: Process multiple roles in parallel on same CPU
#
# GPU Phase (roles_optimized_gpu.py):
#   - Time per role: O(N_pairs * T_activation) where T_activation << T_judge
#   - Memory: O(model_size) + O(activation_accumulator)
#   - Bottleneck: None (GPU at 100% utilization)
#   - Parallelization: Process multiple roles in parallel on multiple GPUs
#
# SLURM Optimization:
#   - Without: GPU waits idle while CPU generates Q/A (~1-2 minutes wasted)
#   - With: GPU immediately extracts while CPU generates next role's Q/A
#   - Savings: ~5% - 15% total wall-clock time per role
#
#
# Comparison to Original Implementation
# ======================================
#
# Original (roles/role_persona_model.py):
# - Single-phase: Generate → Judge → Extract (in same process)
# - GPU idle time: While CPU judges, GPU sits idle
# - Restart on failure: Any interruption means restarting extraction
# - No intermediate storage: Can't inspect Q/A separately
#
# Optimized (roles_optimized_*.py):
# - Two-phase: [Generate → Judge] then [Extract]
# - No GPU idle: Different jobs on CPU and GPU simultaneously
# - Resume capability: Load Q/A JSON and continue
# - Debuggable: Can inspect Q/A before extraction
# - Flexible: Can adjust params per phase independently
#
#
# Extension Points
# ================
#
# 1. Custom Q/A Generator
#    class CustomQAGenerator(RoleQAGenerator):
#        def generate_and_judge_qa(self, role, dataset, ...):
#            # Custom logic here
#            return qa_data
#
# 2. Custom Activation Extractor
#    class CustomExtractor(RoleActivationExtractor):
#        def extract_persona_vector(self, ...):
#            # Custom extraction logic
#            return vectors
#
# 3. Additional Layers
#    # Extract from multiple layers in single pass
#    qa_data = generator.generate_and_judge_qa(...)
#    for layer in [14, 16, 18, 20]:
#        extractor = RoleActivationExtractor(..., layer=layer)
#        extractor.extract_persona_vector()
#
# 4. Batch Processing
#    # Process 100 roles across 10 SLURM jobs
#    for batch in range(0, 100, 10):
#        roles_batch = roles[batch:batch+10]
#        submit_cpu_job(roles_batch)
#        submit_gpu_job(roles_batch)
#
#
# Error Handling Strategy
# =======================
#
# Missing Dataset:
#     try:
#         dataset = RoleDataset.from_json(role)
#     except:
#         logger.error(f"Dataset not found for {role}")
#         continue  # Skip, don't fail
#
# Missing Q/A Responses:
#     if not qa_path.exists():
#         logger.error(f"Q/A responses not found")
#         logger.info("Please run CPU task first")
#         continue  # Skip, don't fail
#
# Inference/Judge Failures:
#     try:
#         response = generator._generate_response(messages)
#     except Exception as e:
#         logger.error(f"Failed to generate: {e}")
#         continue  # Skip this pair, try next
#
# Partial Extraction:
#     # If GPU job crashes after 20/40 pairs extracted:
#     # 1. Sums are already accumulated
#     # 2. Re-run GPU with more pairs
#     # 3. Merges with previous sums naturally
#
#
# Memory Management
# =================
#
# CPU Phase (per batch):
#   Model weights: ~7GB (loaded once)
#   Tokenized cache: ~50-100MB (N_questions * avg_tokens * 2 bytes)
#   Response strings: ~10-50MB (kept in memory during batch)
#   Total: ~7GB + small overhead
#
# GPU Phase (per batch):
#   Model weights: ~7GB (loaded on GPU)
#   Tokenized cache: ~50-100MB (on CPU)
#   Activation accumulators: ~1GB (single-precision float32)
#   Total: ~7GB GPU + ~50MB CPU
#
# Between Phases:
#   Q/A JSON: ~100KB per 40 pairs (disk storage)
#   No model weights kept in memory
#
#
# Testing Strategy
# ================
#
# Unit Tests:
#   - _generate_response(): Generate text from prompts
#   - _top_p_sample(): Nucleus sampling correctness
#   - generate_and_judge_qa(): Q/A generation and filtering
#   - extract_persona_vector(): Activation extraction
#   - contrastive_persona_vectors(): Vector computation
#
# Integration Tests:
#   - Full CPU → JSON pipeline
#   - JSON → Full GPU pipeline
#   - CPU → GPU chained execution
#   - Error handling (missing dataset, missing Q/A, etc.)
#
# Performance Tests:
#   - Time per phase with different role counts
#   - Memory usage during execution
#   - GPU utilization metrics
#   - Q/A quality (score distribution)
#
#
# Future Improvements
# ===================
#
# 1. Parallel Role Processing
#    - Use multiprocessing for multiple roles in CPU phase
#    - Use AsyncIO for multiple GPU devices in GPU phase
#
# 2. Streaming Q/A Generation
#    - Write Q/A pairs to JSON as they're validated (not batch)
#    - Allows GPU phase to start immediately after first valid pair
#
# 3. Adaptive Sampling
#    - Track which prompts/questions are "hard" (low judge scores)
#    - Oversample hard samples for better representation
#
# 4. Partial Extraction Resume
#    - If GPU job crashes, detect and resume from checkpoint
#    - Store intermediate activation sums
#
# 5. Model Quantization
#    - Use 4-bit or 8-bit quantization in inference
#    - Reduce memory usage and increase speed
#
# 6. Distributed Extraction
#    - Extract from multiple layers in parallel
#    - Combine results in single persona vector
#
# 7. Ablation Studies
#    - Compare random shuffle vs sequential
#    - Compare different temperature values
#    - Compare judge thresholds impact
#
#
# References
# ==========
#
# - AbstractPersonaModel: src/pvx/abstraction/pvx_models/abstract_persona_model.py
# - RoleDataset: src/pvx/implementations/roles/role_dataset.py
# - RoleJudge: src/pvx/implementations/judges/role_judge.py
# - Original Implementation: src/pvx/implementations/roles/role_persona_model.py
#
