"""Phase 11 full-image detection audit and protocol utilities."""

from windblade.detection.core import (
    DetectionAuditError,
    aggregate_seed_metrics,
    apparatus_check,
    box_iou,
    choose_validation_threshold,
    decompose_detection_errors,
    enforce_test_firewall,
    run_audit,
    validate_audit,
    validate_box,
    validate_framework_metrics,
    validate_image_coverage,
    validate_unique_annotations,
    voc_inclusive_to_yolo,
)

__all__ = [
    "DetectionAuditError",
    "aggregate_seed_metrics",
    "apparatus_check",
    "box_iou",
    "choose_validation_threshold",
    "decompose_detection_errors",
    "enforce_test_firewall",
    "run_audit",
    "validate_audit",
    "validate_box",
    "validate_framework_metrics",
    "validate_image_coverage",
    "validate_unique_annotations",
    "voc_inclusive_to_yolo",
]
