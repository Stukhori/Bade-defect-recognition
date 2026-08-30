from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from windblade.config import load_config
from windblade.data.processed import LABELS
from windblade.evaluation.metrics import classification_metrics
from windblade.robustness.aggregation import aggregate_evaluations, prediction_transitions, robustness_derivations
from windblade.robustness.corruptions import (
    apply_corruption,
    brightness_reduction,
    condition_specs,
    gaussian_blur,
    jpeg_round_trip,
    resolution_degradation,
)
from windblade.robustness.dataset import pixel_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs/robustness.yaml")


def synthetic_image() -> Image.Image:
    y, x = np.mgrid[0:224, 0:224]
    pixels = np.stack(((x * 7 + y * 3) % 256, (x * 2 + y * 11) % 256, (x * 13 + y * 5) % 256), axis=-1).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_frozen_conditions_are_one_clean_plus_twelve_degraded() -> None:
    specs = condition_specs(CONFIG.as_dict(), include_clean=True)
    assert len(specs) == 13
    assert specs[0]["condition_id"] == "clean"
    assert len({row["condition_id"] for row in specs}) == 13
    assert len(specs[1:]) * 162 == 1944
    assert [row["parameter"] for row in specs if row["corruption_family"] == "gaussian_blur"] == [0.75, 1.5, 3.0]
    assert [row["parameter"] for row in specs if row["corruption_family"] == "resolution"] == [168, 112, 56]
    assert [row["parameter"] for row in specs if row["corruption_family"] == "brightness"] == [0.75, 0.5, 0.25]
    assert [row["parameter"] for row in specs if row["corruption_family"] == "jpeg"] == [75, 50, 25]


def test_blur_clean_and_configured_radii_are_deterministic_rgb_224() -> None:
    image = synthetic_image()
    clean = gaussian_blur(image, 0)
    assert clean.mode == "RGB" and clean.size == (224, 224)
    assert clean.tobytes() == image.tobytes()
    for radius in (0.75, 1.5, 3.0):
        first = gaussian_blur(image, radius)
        second = gaussian_blur(image, radius)
        assert first.mode == "RGB" and first.size == (224, 224)
        assert first.tobytes() == second.tobytes()


def test_resolution_frozen_bilinear_path_and_clean_untouched() -> None:
    image = synthetic_image()
    assert resolution_degradation(image, 224).tobytes() == image.tobytes()
    for intermediate in (168, 112, 56):
        expected = image.resize((intermediate, intermediate), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
        observed = resolution_degradation(image, intermediate)
        assert observed.mode == "RGB" and observed.size == (224, 224)
        assert observed.tobytes() == expected.tobytes()


@pytest.mark.parametrize("factor", [0.75, 0.5, 0.25])
def test_brightness_exact_rint_clip_uint8(factor: float) -> None:
    pixels = np.zeros((224, 224, 3), dtype=np.uint8)
    pixels[0, 0] = [1, 2, 3]
    pixels[0, 1] = [253, 254, 255]
    image = Image.fromarray(pixels, mode="RGB")
    observed = np.asarray(brightness_reduction(image, factor))
    expected = np.clip(np.rint(pixels.astype(np.float64) * factor), 0, 255).astype(np.uint8)
    assert np.array_equal(observed, expected)
    assert brightness_reduction(image, factor).tobytes() == brightness_reduction(image, factor).tobytes()


def test_brightness_one_is_unchanged() -> None:
    image = synthetic_image()
    assert brightness_reduction(image, 1.0).tobytes() == image.tobytes()


@pytest.mark.parametrize("quality", [75, 50, 25])
def test_jpeg_round_trip_is_deterministic_rgb_and_png_safe(tmp_path: Path, quality: int) -> None:
    image = synthetic_image()
    first = jpeg_round_trip(image, quality, subsampling=2, optimize=False, progressive=False)
    second = jpeg_round_trip(image, quality, subsampling=2, optimize=False, progressive=False)
    assert first.mode == "RGB" and first.size == (224, 224)
    assert first.tobytes() == second.tobytes()
    assert pixel_sha256(first) != pixel_sha256(image)
    output = tmp_path / "decoded.png"
    first.save(output, format="PNG")
    with Image.open(output) as reloaded:
        assert reloaded.mode == "RGB" and reloaded.size == (224, 224)
        assert reloaded.tobytes() == first.tobytes()


def test_each_corruption_is_directly_derived_from_clean_not_cumulative() -> None:
    data = CONFIG.as_dict()
    clean = synthetic_image()
    direct = apply_corruption(clean, "gaussian_blur", "moderate", data)
    cumulative = apply_corruption(apply_corruption(clean, "gaussian_blur", "mild", data), "gaussian_blur", "moderate", data)
    assert direct.tobytes() == gaussian_blur(clean, 1.5).tobytes()
    assert direct.tobytes() != cumulative.tobytes()
    for family in ("resolution", "brightness", "jpeg"):
        assert apply_corruption(clean, family, "moderate", data).tobytes() == apply_corruption(clean.copy(), family, "moderate", data).tobytes()


def test_manifest_design_has_one_method_independent_pixel_identity() -> None:
    specs = condition_specs(CONFIG.as_dict(), include_clean=False)
    assert all("method" not in row for row in specs)
    assert len({row["transformation_config_hash"] for row in specs}) == 12


def test_phase8_scientific_inference_code_has_no_fitting_or_training_calls() -> None:
    for relative in ("src/windblade/robustness/evaluation.py", "src/windblade/robustness/runner.py"):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert ".fit(" not in source
        assert ".backward(" not in source
        assert "optimizer.step(" not in source


def test_metric_derivations_and_flip_transitions() -> None:
    derived = robustness_derivations(0.8, 0.6)
    assert derived["absolute_drop"] == pytest.approx(-0.2)
    assert derived["retention"] == pytest.approx(0.75)
    assert derived["relative_loss"] == pytest.approx(0.25)
    transitions = prediction_transitions([0, 1, 2, 3], [0, 0, 2, 0], [0, 1, 1, 0])
    assert transitions == {
        "prediction_flip_count": 2,
        "prediction_flip_rate": 0.5,
        "harmful_flip_count": 1,
        "beneficial_flip_count": 1,
        "correct_to_correct": 1,
        "correct_to_incorrect": 1,
        "incorrect_to_correct": 1,
        "incorrect_to_incorrect": 1,
    }


def _fake_result(prediction_offset: int = 0) -> dict:
    true = np.repeat(np.arange(6), 27)
    predicted = true.copy()
    if prediction_offset:
        predicted[:prediction_offset] = (predicted[:prediction_offset] + 1) % 6
    metrics = classification_metrics(true, predicted, LABELS)
    rows = []
    for index, (true_id, predicted_id) in enumerate(zip(true, predicted, strict=True)):
        rows.append(
            {
                "instance_id": f"sample_{index:03d}",
                "source_image_id": f"source_{index:03d}",
                "true_class_id": int(true_id),
                "true_label": LABELS[int(true_id)],
                "predicted_class_id": int(predicted_id),
                "predicted_label": LABELS[int(predicted_id)],
                "correct": bool(true_id == predicted_id),
                **{f"logit_{label}": float(class_id == predicted_id) for class_id, label in enumerate(LABELS)},
            }
        )
    return {"metrics": metrics, "predictions": rows, "logits": None}


def test_cnn_aggregation_uses_exact_seeds_sample_sd_and_frozen_class_order() -> None:
    specs = condition_specs(CONFIG.as_dict(), include_clean=True)
    entries = []
    for method in ("hog", "lbp"):
        for spec in specs:
            entries.append({"method": method, "seed": None, "condition_id": spec["condition_id"], "result": _fake_result(0 if spec["condition_id"] == "clean" else 1)})
    for method in ("resnet18", "mobilenet_v3_small"):
        for seed, offset in zip((17, 29, 43), (1, 2, 3), strict=True):
            for spec in specs:
                entries.append({"method": method, "seed": seed, "condition_id": spec["condition_id"], "result": _fake_result(0 if spec["condition_id"] == "clean" else offset)})
    result = aggregate_evaluations(entries, specs)
    assert {row["seed"] for row in result["per_seed_results"] if row["method"] == "resnet18"} == {17, 29, 43}
    cnn_row = next(row for row in result["robustness_summary"] if row["method"] == "resnet18" and row["condition_id"] == "gaussian_blur_mild")
    traditional_row = next(row for row in result["robustness_summary"] if row["method"] == "hog" and row["condition_id"] == "gaussian_blur_mild")
    assert cnn_row["macro_f1_sample_sd"] is not None
    assert cnn_row["standard_deviation"] == "sample (ddof=1)"
    assert traditional_row["macro_f1_sample_sd"] is None
    assert traditional_row["standard_deviation"] == "N/A"
    assert [row["class"] for row in result["per_class_robustness"][:6]] == list(LABELS)
