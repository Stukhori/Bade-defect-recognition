"""Frozen-checkpoint Grad-CAM generation without parameter updates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as functional

from windblade.data.processed import LABELS, sha256_file
from windblade.deep.checkpoints import state_dict_fingerprint
from windblade.deep.dataset import canonical_transform
from windblade.error_analysis.core import ErrorAnalysisError, SEEDS
from windblade.robustness.evaluation import load_frozen_cnn


def validate_target_identity(
    target_role: str,
    target_class_id: int,
    target_label: str,
    prediction: Mapping[str, Any],
) -> None:
    """Require a Grad-CAM target to match its frozen semantic role."""

    if target_role == "true_class":
        expected_id = int(prediction["true_class_id"])
        expected_label = str(prediction["true_label"])
    elif target_role == "predicted_class":
        expected_id = int(prediction["predicted_class_id"])
        expected_label = str(prediction["predicted_label"])
    else:
        raise ErrorAnalysisError(f"unknown Grad-CAM target role: {target_role}")
    if not 0 <= int(target_class_id) < len(LABELS):
        raise ErrorAnalysisError(f"Grad-CAM target index is out of range: {target_class_id}")
    if LABELS[int(target_class_id)] != str(target_label):
        raise ErrorAnalysisError(
            f"Grad-CAM target index/label mismatch: {target_class_id}/{target_label}"
        )
    if int(target_class_id) != expected_id or str(target_label) != expected_label:
        raise ErrorAnalysisError(
            f"Grad-CAM {target_role} target does not match frozen prediction identity: "
            f"observed={target_class_id}/{target_label}, expected={expected_id}/{expected_label}"
        )


def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    modules = dict(model.named_modules())
    if path not in modules:
        raise ErrorAnalysisError(f"Grad-CAM target layer is absent: {path}")
    return modules[path]


def gradcam_map(model: torch.nn.Module, target_layer: torch.nn.Module, tensor: torch.Tensor, target_class: int) -> tuple[np.ndarray, tuple[int, ...], np.ndarray]:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        activations.append(output)
        output.register_hook(lambda grad: gradients.append(grad))

    handle = target_layer.register_forward_hook(capture)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        logits[0, int(target_class)].backward()
    finally:
        handle.remove()
    if len(activations) != 1 or len(gradients) != 1:
        raise ErrorAnalysisError("Grad-CAM hook did not capture exactly one activation/gradient")
    activation, gradient = activations[0].detach(), gradients[0].detach()
    if activation.ndim != 4 or gradient.shape != activation.shape:
        raise ErrorAnalysisError("Grad-CAM activation or gradient shape is invalid")
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    minimum, maximum = cam.min(), cam.max()
    normalized = torch.zeros_like(cam) if float(maximum - minimum) <= 1e-12 else (cam - minimum) / (maximum - minimum)
    if not torch.isfinite(normalized).all() or not torch.isfinite(logits).all():
        raise ErrorAnalysisError("Grad-CAM produced NaN or infinity")
    return normalized.cpu().numpy().astype(np.float32), tuple(activation.shape), logits.detach().cpu().numpy()[0].astype(np.float32)


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def annotation_box(meta: Mapping[str, Any]) -> tuple[int, int, int, int]:
    scale = 224.0 / float(meta["crop_side"])
    values = (
        round((float(meta["bbox_xmin"]) - float(meta["crop_xmin"])) * scale),
        round((float(meta["bbox_ymin"]) - float(meta["crop_ymin"])) * scale),
        round((float(meta["bbox_xmax"]) - float(meta["crop_xmin"])) * scale),
        round((float(meta["bbox_ymax"]) - float(meta["crop_ymin"])) * scale),
    )
    return tuple(max(0, min(223, int(value))) for value in values)  # type: ignore[return-value]


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def render_annotation(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    draw.rectangle(box, outline=(255, 0, 0), width=3)
    return rendered


def render_heatmap(cam: np.ndarray) -> Image.Image:
    rgba = matplotlib.colormaps["magma"](np.clip(cam, 0.0, 1.0), bytes=True)
    return Image.fromarray(np.asarray(rgba[..., :3], dtype=np.uint8), mode="RGB")


def render_overlay(image: Image.Image, heatmap: Image.Image, alpha: float, box: tuple[int, int, int, int]) -> Image.Image:
    blended = Image.blend(image.convert("RGB"), heatmap.convert("RGB"), float(alpha))
    draw = ImageDraw.Draw(blended)
    draw.rectangle(box, outline=(255, 0, 0), width=3)
    return blended


def generate_gradcams(
    config: Mapping[str, Any],
    phase8_config: Mapping[str, Any],
    root: Path,
    selected: Sequence[Mapping[str, Any]],
    error_rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Mapping[str, Any]],
    corrupted_paths: Mapping[tuple[str, str], str],
    figures_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    indexed = {(str(row["method"]), int(row["seed"]), str(row["condition_id"]), str(row["instance_id"])): row for row in error_rows if row["method"] in {"resnet18", "mobilenet_v3_small"}}
    manifest: list[dict[str, Any]] = []
    model_checks: list[dict[str, Any]] = []
    transform = canonical_transform()
    for method in config["gradcam"]["methods"]:
        cases = [row for row in selected if row["method"] == method]
        for seed in SEEDS:
            device = torch.device(config["runtime"]["device"])
            model = load_frozen_cnn(method, seed, phase8_config, root, device)
            target_path = str(config["gradcam"]["target_layers"][method])
            target_layer = resolve_module(model, target_path)
            before = state_dict_fingerprint(model.state_dict())
            checkpoint = root / phase8_config["models"][method]["result_root"] / f"seed_{seed}" / "best_state_dict.pt"
            activation_shapes: set[tuple[int, ...]] = set()
            for case in cases:
                instance_id, condition_id = str(case["instance_id"]), str(case["condition_id"])
                states = [("clean", "clean")]
                if condition_id != "clean": states.append(("degraded", condition_id))
                for input_state, state_condition in states:
                    prediction = indexed[(method, seed, state_condition, instance_id)]
                    path = root / (str(geometry[instance_id]["crop_path"]) if "crop_path" in geometry[instance_id] else f"{config['dataset']['processed_root']}/{geometry[instance_id]['output_relative_path']}")
                    if state_condition != "clean": path = root / corrupted_paths[(instance_id, state_condition)]
                    with Image.open(path) as handle:
                        image = handle.convert("RGB").copy()
                    tensor = transform(image).unsqueeze(0).to(device)
                    true_id = int(prediction["true_class_id"])
                    predicted_id = int(prediction["predicted_class_id"])
                    targets = [("true_class", true_id)]
                    if predicted_id != true_id: targets.append(("predicted_class", predicted_id))
                    state_root = figures_root / "gradcam" / str(case["review_id"]) / method / f"seed_{seed}" / input_state
                    box = annotation_box(geometry[instance_id])
                    _save_png(image, state_root / "input_rgb.png")
                    _save_png(render_annotation(image, box), state_root / "annotation_overlay.png")
                    for target_role, target_id in targets:
                        target_label = LABELS[target_id]
                        validate_target_identity(
                            target_role, target_id, target_label, prediction
                        )
                        first, shape, logits = gradcam_map(model, target_layer, tensor.clone(), target_id)
                        second, shape_second, logits_second = gradcam_map(model, target_layer, tensor.clone(), target_id)
                        if shape != shape_second or not np.array_equal(first, second) or not np.array_equal(logits, logits_second):
                            raise ErrorAnalysisError("Grad-CAM did not regenerate identically in-process")
                        expected_shape = tuple(int(value) for value in config["gradcam"]["expected_activation_shapes"][method])
                        if shape != expected_shape:
                            raise ErrorAnalysisError(f"unexpected {method} Grad-CAM activation shape: {shape}")
                        activation_shapes.add(shape)
                        heatmap = render_heatmap(first)
                        heatmap_path = state_root / f"{target_role}_{target_label}_heatmap.png"
                        overlay_path = state_root / f"{target_role}_{target_label}_overlay.png"
                        _save_png(heatmap, heatmap_path)
                        _save_png(render_overlay(image, heatmap, float(config["gradcam"]["alpha"]), box), overlay_path)
                        manifest.append({
                            "review_id": case["review_id"], "instance_id": instance_id, "method": method, "seed": seed,
                            "condition_id": condition_id, "input_state": input_state, "input_condition_id": state_condition,
                            "target_role": target_role, "target_class_id": target_id, "target_label": target_label,
                            "predicted_label": prediction["predicted_label"], "true_label": prediction["true_label"],
                            "target_module_path": target_path, "activation_shape": "x".join(map(str, shape)),
                            "checkpoint_sha256": sha256_file(checkpoint), "checkpoint_fingerprint": before,
                            "preprocessing": config["gradcam"]["preprocessing"], "normalization": config["gradcam"]["normalization"],
                            "array_sha256": array_sha256(first), "finite": True, "minimum": float(first.min()), "maximum": float(first.max()),
                            "input_path": (state_root / "input_rgb.png").relative_to(root).as_posix(),
                            "annotation_path": (state_root / "annotation_overlay.png").relative_to(root).as_posix(),
                            "heatmap_path": heatmap_path.relative_to(root).as_posix(), "overlay_path": overlay_path.relative_to(root).as_posix(),
                        })
            after = state_dict_fingerprint(model.state_dict())
            if before != after:
                raise ErrorAnalysisError(f"Grad-CAM mutated model parameters: {method}/seed{seed}")
            model_checks.append({"method": method, "seed": seed, "target_module_path": target_path, "activation_shapes": [list(shape) for shape in sorted(activation_shapes)], "checkpoint_sha256": sha256_file(checkpoint), "state_dict_before": before, "state_dict_after": after, "parameters_unchanged": True})
            del model
    return manifest, model_checks
