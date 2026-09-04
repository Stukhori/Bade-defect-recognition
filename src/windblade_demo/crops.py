"""Manual-box mapping and exact Phase 3 contextual-crop reuse."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from PIL import Image, ImageDraw

from windblade.data.crops import CropGeometry, CropValidationError, calculate_square_crop
from windblade_demo.constants import MODEL_INPUT_SIZE


class SelectionValidationError(ValueError):
    """Raised when a manual rectangle is absent or invalid."""


@dataclass(frozen=True)
class PixelBox:
    """Zero-based, half-open image coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass(frozen=True)
class ContextualCrop:
    selected_box: PixelBox
    geometry: CropGeometry
    contextual_pixels: Image.Image
    model_input: Image.Image


def display_image(image: Image.Image, maximum: int = 700) -> Image.Image:
    """Create a display-only image whose coordinate scale is known exactly."""

    if maximum <= 0:
        raise SelectionValidationError("Display maximum must be positive.")
    scale = min(1.0, maximum / image.width, maximum / image.height)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    return image.convert("RGB").resize(size, Image.Resampling.BILINEAR)


def map_display_box(
    rectangle: Mapping[str, Any],
    *,
    display_size: tuple[int, int],
    original_size: tuple[int, int],
) -> PixelBox:
    """Map a cropper rectangle from display pixels to original half-open pixels."""

    try:
        left = float(rectangle["left"])
        top = float(rectangle["top"])
        width = float(rectangle["width"])
        height = float(rectangle["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SelectionValidationError("The rectangle component returned invalid coordinates.") from exc
    if not all(math.isfinite(value) for value in (left, top, width, height)):
        raise SelectionValidationError("The rectangle coordinates must be finite.")
    if width <= 0 or height <= 0:
        raise SelectionValidationError("Draw a rectangle with positive width and height.")
    display_width, display_height = display_size
    original_width, original_height = original_size
    if min(display_width, display_height, original_width, original_height) <= 0:
        raise SelectionValidationError("Image dimensions must be positive.")

    raw_right = left + width
    raw_bottom = top + height
    left = max(0.0, min(left, float(display_width)))
    top = max(0.0, min(top, float(display_height)))
    right = max(0.0, min(raw_right, float(display_width)))
    bottom = max(0.0, min(raw_bottom, float(display_height)))
    x_scale = original_width / display_width
    y_scale = original_height / display_height
    mapped = PixelBox(
        left=max(0, min(original_width - 1, math.floor(left * x_scale))),
        top=max(0, min(original_height - 1, math.floor(top * y_scale))),
        right=max(1, min(original_width, math.ceil(right * x_scale))),
        bottom=max(1, min(original_height, math.ceil(bottom * y_scale))),
    )
    if mapped.width <= 0 or mapped.height <= 0:
        raise SelectionValidationError("The mapped rectangle has no image area.")
    return mapped


def prepare_region(image: Image.Image) -> Image.Image:
    """Apply the frozen Phase 3 RGB/bilinear 224x224 pixel policy."""

    return image.convert("RGB").resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)


def contextual_crop(image: Image.Image, selected: PixelBox) -> ContextualCrop:
    """Build a model input using the frozen Phase 3 contextual geometry."""

    source = image.convert("RGB")
    if not (
        0 <= selected.left < selected.right <= source.width
        and 0 <= selected.top < selected.bottom <= source.height
    ):
        raise SelectionValidationError("The selected rectangle lies outside the original image.")
    try:
        geometry = calculate_square_crop(
            xmin=selected.left + 1,
            ymin=selected.top + 1,
            xmax=selected.right,
            ymax=selected.bottom,
            image_width=source.width,
            image_height=source.height,
            context_multiplier=1.5,
            minimum_side=64,
        )
    except CropValidationError as exc:
        raise SelectionValidationError(str(exc)) from exc
    pixels = source.crop(
        (geometry.crop_xmin, geometry.crop_ymin, geometry.crop_xmax, geometry.crop_ymax)
    )
    model_input = pixels.resize(MODEL_INPUT_SIZE, Image.Resampling.BILINEAR)
    return ContextualCrop(selected, geometry, pixels, model_input)


def annotated_selection(image: Image.Image, crop: ContextualCrop) -> Image.Image:
    """Render the user box and resulting contextual square for presentation."""

    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    contextual = (
        crop.geometry.crop_xmin,
        crop.geometry.crop_ymin,
        crop.geometry.crop_xmax - 1,
        crop.geometry.crop_ymax - 1,
    )
    selected = (
        crop.selected_box.left,
        crop.selected_box.top,
        crop.selected_box.right - 1,
        crop.selected_box.bottom - 1,
    )
    line_width = max(2, max(rendered.size) // 350)
    draw.rectangle(contextual, outline="#38BDF8", width=line_width)
    draw.rectangle(selected, outline="#1D4ED8", width=line_width)
    return rendered
