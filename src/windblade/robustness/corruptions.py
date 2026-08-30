"""Frozen deterministic image corruptions for Phase 8."""

from __future__ import annotations

from io import BytesIO
from importlib import metadata
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageFilter, features

from windblade.features.common import canonical_hash


FAMILIES = ("gaussian_blur", "resolution", "brightness", "jpeg")
SEVERITIES = ("mild", "moderate", "severe")


def pillow_environment() -> dict[str, Any]:
    """Return the Pillow and JPEG implementation identity used for corruption."""

    return {
        "pillow_version": metadata.version("Pillow"),
        "jpeg_support": bool(features.check("jpg")),
        "jpeg_library_version": features.version("jpg"),
    }


def _canonical_rgb(image: Image.Image) -> Image.Image:
    rgb = image.copy() if image.mode == "RGB" else image.convert("RGB")
    if rgb.size != (224, 224):
        raise ValueError(f"Phase 8 requires 224x224 input; received {rgb.size}")
    return rgb


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    rgb = _canonical_rgb(image)
    if radius < 0:
        raise ValueError("Gaussian blur radius cannot be negative")
    return rgb if radius == 0 else rgb.filter(ImageFilter.GaussianBlur(radius=float(radius)))


def resolution_degradation(image: Image.Image, intermediate_size: int, final_size: int = 224) -> Image.Image:
    rgb = _canonical_rgb(image)
    if final_size != 224 or intermediate_size not in {56, 112, 168, 224}:
        raise ValueError("resolution dimensions violate the frozen Phase 8 contract")
    if intermediate_size == final_size:
        return rgb
    reduced = rgb.resize((intermediate_size, intermediate_size), resample=Image.Resampling.BILINEAR)
    return reduced.resize((final_size, final_size), resample=Image.Resampling.BILINEAR)


def brightness_reduction(image: Image.Image, factor: float) -> Image.Image:
    rgb = _canonical_rgb(image)
    if factor not in {0.25, 0.5, 0.75, 1.0}:
        raise ValueError("brightness factor violates the frozen Phase 8 contract")
    if factor == 1.0:
        return rgb
    pixels = np.asarray(rgb, dtype=np.uint8)
    degraded = np.clip(np.rint(pixels.astype(np.float64) * factor), 0, 255).astype(np.uint8)
    return Image.fromarray(degraded, mode="RGB")


def jpeg_round_trip(
    image: Image.Image,
    quality: int,
    *,
    subsampling: int = 2,
    optimize: bool = False,
    progressive: bool = False,
) -> Image.Image:
    rgb = _canonical_rgb(image)
    if quality not in {25, 50, 75}:
        raise ValueError("JPEG quality violates the frozen Phase 8 contract")
    if subsampling != 2 or optimize or progressive:
        raise ValueError("JPEG encoder options violate the frozen Phase 8 contract")
    encoded = BytesIO()
    rgb.save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=optimize,
        progressive=progressive,
    )
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return decoded.convert("RGB").copy()


def condition_specs(config: Mapping[str, Any], *, include_clean: bool = True) -> list[dict[str, Any]]:
    """Return the one clean plus twelve frozen, independently applied conditions."""

    corruptions = config["corruptions"] if "corruptions" in config else config
    if tuple(corruptions["order"]) != FAMILIES or tuple(corruptions["severities"]) != SEVERITIES:
        raise ValueError("corruption order or severity labels changed")
    rows: list[dict[str, Any]] = []
    if include_clean:
        rows.append(
            {
                "condition_id": "clean",
                "corruption_family": "clean",
                "severity": "clean",
                "parameter": "original_phase3_png",
                "transformation_config_hash": canonical_hash({"condition": "clean"}),
            }
        )
    for family in FAMILIES:
        family_config = corruptions[family]
        for severity in SEVERITIES:
            parameter = family_config[severity]
            payload = {
                "family": family,
                "severity": severity,
                "parameter": parameter,
                "configuration": dict(family_config),
                "pillow": pillow_environment(),
            }
            rows.append(
                {
                    "condition_id": f"{family}_{severity}",
                    "corruption_family": family,
                    "severity": severity,
                    "parameter": parameter,
                    "transformation_config_hash": canonical_hash(payload),
                }
            )
    if len(rows) != (13 if include_clean else 12):
        raise AssertionError("unexpected Phase 8 condition count")
    return rows


def apply_corruption(
    image: Image.Image,
    family: str,
    severity: str,
    config: Mapping[str, Any],
) -> Image.Image:
    """Apply exactly one corruption to an independent clean RGB image."""

    corruptions = config["corruptions"] if "corruptions" in config else config
    if family == "clean" and severity == "clean":
        return _canonical_rgb(image)
    if family not in FAMILIES or severity not in SEVERITIES:
        raise ValueError(f"unknown Phase 8 condition: {family}/{severity}")
    family_config = corruptions[family]
    parameter = family_config[severity]
    if family == "gaussian_blur":
        return gaussian_blur(image, float(parameter))
    if family == "resolution":
        return resolution_degradation(image, int(parameter), int(family_config["final_size"]))
    if family == "brightness":
        return brightness_reduction(image, float(parameter))
    return jpeg_round_trip(
        image,
        int(parameter),
        subsampling=int(family_config["subsampling"]),
        optimize=bool(family_config["optimize"]),
        progressive=bool(family_config["progressive"]),
    )
