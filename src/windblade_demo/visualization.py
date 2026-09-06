"""Presentation-only image annotations for user-selected regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from PIL import Image, ImageDraw, ImageFont

from windblade_demo.session import RegionRecord

if TYPE_CHECKING:
    from windblade_demo.detector import RegionProposal


REGION_COLORS = ("#1D4ED8", "#0284C7", "#4338CA", "#7C3AED", "#C2415D", "#0F4C81")


def color_for_region(region_id: str) -> str:
    number = int(region_id[1:]) if region_id.startswith("R") and region_id[1:].isdigit() else 1
    return REGION_COLORS[(number - 1) % len(REGION_COLORS)]


def annotate_regions(
    image: Image.Image,
    records: Iterable[RegionRecord],
    *,
    include_prediction: bool = True,
) -> Image.Image:
    """Draw saved user-selected or user-accepted boxes."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for record in records:
        if record.selected_box is None:
            continue
        color = color_for_region(record.region_id)
        box = tuple(record.selected_box)
        draw.rectangle(box, outline=color, width=max(2, round(min(canvas.size) / 250)))
        label = record.region_id
        if include_prediction:
            label += f" · {record.predicted_label}"
        bounds = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=1)
        width = bounds[2] - bounds[0] + 8
        height = bounds[3] - bounds[1] + 6
        y = max(0, box[1] - height)
        draw.rectangle((box[0], y, min(canvas.width, box[0] + width), y + height), fill=color)
        draw.text((box[0] + 4, y + 3), label, fill="white", font=font, stroke_width=1)
    return canvas


def annotate_proposals(image: Image.Image, proposals: Iterable["RegionProposal"]) -> Image.Image:
    """Draw presentation-only proposal numbers without exposing internal model labels."""

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, round(min(canvas.size) / 250))
    for proposal in proposals:
        box = proposal.box.as_tuple()
        color = "#0B6E99"
        draw.rectangle((box[0], box[1], box[2] - 1, box[3] - 1), outline=color, width=line_width)
        label = proposal.proposal_id
        bounds = draw.textbbox((box[0], box[1]), label, font=font, stroke_width=1)
        label_width = bounds[2] - bounds[0] + 8
        label_height = bounds[3] - bounds[1] + 6
        y = max(0, box[1] - label_height)
        draw.rectangle((box[0], y, min(canvas.width, box[0] + label_width), y + label_height), fill=color)
        draw.text((box[0] + 4, y + 3), label, fill="white", font=font, stroke_width=1)
    return canvas
