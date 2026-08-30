"""Frozen identities and presentation-only labels for the local demo."""

from __future__ import annotations

from pathlib import Path

from windblade.data.processed import LABELS

CLASS_LABELS = tuple(LABELS)
HUMAN_LABELS = {
    "craze": "Craze",
    "corrosion": "Corrosion",
    "surface_injure": "Surface injury",
    "thunderstrike": "Thunderstrike",
    "crack": "Crack",
    "hide_craze": "Hide craze",
}

MODEL_DISPLAY_NAME = "MobileNetV3-Small — frozen Phase 6, full-data seed 17"
DATASET_FINGERPRINT = "4bd754a1015be2ec99c88a57a23586e286b03cc178ee148b298850e5ca848991"
CHECKPOINT_STATE_FINGERPRINT = "3c17629d1b1748e2f3d9046cb9a3d88c6369786acc1381f105974396c0f46757"
CHECKPOINT_FILE_SHA256 = "9c7a5f18e7d05a320e1296c73bbeb9366636e0e55dc7c6ff2bab6d8808a0e5a5"
CHECKPOINT_RELATIVE_PATH = Path(
    "experiments/results/phase6_mobilenet_v3_small_v1/final/seed_17/best_state_dict.pt"
)
EXPECTED_ARCHITECTURE = "torchvision_mobilenet_v3_small"
EXPECTED_SEED = 17
MODEL_INPUT_SIZE = (224, 224)

MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
MAX_IMAGE_DIMENSION = 20_000
SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg"})
SUPPORTED_FORMATS = frozenset({"PNG", "JPEG"})
