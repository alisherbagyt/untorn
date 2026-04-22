"""
untorn.benchmark
================
Synthetic tear generator + automated quality metrics for regression
testing the reconstruction pipeline.

Public entry points:
    generate_case()  — create one synthetic torn-paper case
    evaluate_case()  — score a pipeline output against ground truth
    run_suite()      — generate + evaluate a whole benchmark suite
"""

from .generate import generate_case, TearGeneratorConfig
from .evaluate import evaluate_case, CaseMetrics

__all__ = [
    "generate_case",
    "TearGeneratorConfig",
    "evaluate_case",
    "CaseMetrics",
]
