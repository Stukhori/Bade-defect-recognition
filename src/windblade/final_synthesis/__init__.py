"""Deterministic Phase 10 statistical synthesis."""

from windblade.final_synthesis.core import (
    FinalSynthesisError,
    apparatus_check,
    bootstrap_indices,
    holm_adjust,
    run_final_synthesis,
    validate_final_synthesis,
    validate_statistical_plan,
)

__all__ = [
    "FinalSynthesisError",
    "apparatus_check",
    "bootstrap_indices",
    "holm_adjust",
    "run_final_synthesis",
    "validate_final_synthesis",
    "validate_statistical_plan",
]
