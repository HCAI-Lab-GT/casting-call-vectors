"""
Optimized roles implementation for persona vector extraction.

This module splits the CPU and GPU intensive work:
- roles_optimized_cpu.py: Generate and judge Q/A responses (CPU)
- roles_optimized_gpu.py: Extract activations and persona vectors (GPU)

This allows SLURM job optimization for maximal GPU usage.
"""

from .roles_optimized_cpu import RoleQAGenerator
from .roles_optimized_gpu import RoleActivationExtractor

__all__ = ["RoleQAGenerator", "RoleActivationExtractor"]
