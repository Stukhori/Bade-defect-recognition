"""Strict, read-only PASCAL VOC XML parsing and box validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


class VocParseError(ValueError):
    """Raised when a VOC document cannot be interpreted safely."""


@dataclass(frozen=True)
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width_inclusive(self) -> int:
        return self.xmax - self.xmin + 1

    @property
    def height_inclusive(self) -> int:
        return self.ymax - self.ymin + 1

    @property
    def area_inclusive(self) -> int:
        return self.width_inclusive * self.height_inclusive

    @property
    def center_x(self) -> float:
        return (self.xmin + self.xmax) / 2.0

    @property
    def center_y(self) -> float:
        return (self.ymin + self.ymax) / 2.0


@dataclass(frozen=True)
class VocObject:
    raw_label: str
    bbox: BoundingBox
    pose: str | None
    truncated: int | None
    difficult: int | None


@dataclass(frozen=True)
class VocAnnotation:
    source_path: Path
    folder: str | None
    filename: str
    declared_path: str | None
    width: int
    height: int
    depth: int | None
    segmented: int | None
    objects: tuple[VocObject, ...]


def _optional_text(parent: ET.Element, name: str) -> str | None:
    node = parent.find(name)
    if node is None or node.text is None:
        return None
    return node.text


def _required_text(parent: ET.Element, name: str, context: str) -> str:
    value = _optional_text(parent, name)
    if value is None or value == "":
        raise VocParseError(f"missing required {context}.{name}")
    return value


def _required_int(parent: ET.Element, name: str, context: str) -> int:
    raw = _required_text(parent, name, context)
    try:
        return int(raw)
    except ValueError as exc:
        raise VocParseError(f"invalid integer at {context}.{name}: {raw!r}") from exc


def _optional_int(parent: ET.Element, name: str, context: str) -> int | None:
    raw = _optional_text(parent, name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise VocParseError(f"invalid integer at {context}.{name}: {raw!r}") from exc


def parse_voc_xml(path: str | Path) -> VocAnnotation:
    """Parse one VOC XML without altering or normalizing its raw label strings."""

    source = Path(path)
    try:
        tree = ET.parse(source)
    except (ET.ParseError, OSError) as exc:
        raise VocParseError(f"cannot parse VOC XML {source}: {exc}") from exc
    root = tree.getroot()
    if root.tag != "annotation":
        raise VocParseError(f"unexpected XML root {root.tag!r} in {source}")

    filename = _required_text(root, "filename", "annotation")
    size = root.find("size")
    if size is None:
        raise VocParseError("missing required annotation.size")
    width = _required_int(size, "width", "annotation.size")
    height = _required_int(size, "height", "annotation.size")
    depth = _optional_int(size, "depth", "annotation.size")
    if width <= 0 or height <= 0:
        raise VocParseError(f"non-positive declared image size: {width}x{height}")

    objects: list[VocObject] = []
    for index, object_node in enumerate(root.findall("object")):
        context = f"annotation.object[{index}]"
        raw_label = _required_text(object_node, "name", context)
        box_node = object_node.find("bndbox")
        if box_node is None:
            raise VocParseError(f"missing required {context}.bndbox")
        box = BoundingBox(
            xmin=_required_int(box_node, "xmin", f"{context}.bndbox"),
            ymin=_required_int(box_node, "ymin", f"{context}.bndbox"),
            xmax=_required_int(box_node, "xmax", f"{context}.bndbox"),
            ymax=_required_int(box_node, "ymax", f"{context}.bndbox"),
        )
        objects.append(
            VocObject(
                raw_label=raw_label,
                bbox=box,
                pose=_optional_text(object_node, "pose"),
                truncated=_optional_int(object_node, "truncated", context),
                difficult=_optional_int(object_node, "difficult", context),
            )
        )

    return VocAnnotation(
        source_path=source,
        folder=_optional_text(root, "folder"),
        filename=filename,
        declared_path=_optional_text(root, "path"),
        width=width,
        height=height,
        depth=depth,
        segmented=_optional_int(root, "segmented", "annotation"),
        objects=tuple(objects),
    )


def validate_bounding_box(
    box: BoundingBox,
    image_width: int,
    image_height: int,
) -> tuple[str, ...]:
    """Return every validity issue under the release's inclusive VOC convention."""

    issues: list[str] = []
    if box.xmin >= box.xmax:
        issues.append("xmin_not_less_than_xmax")
    if box.ymin >= box.ymax:
        issues.append("ymin_not_less_than_ymax")
    if box.xmin < 0:
        issues.append("xmin_negative")
    if box.ymin < 0:
        issues.append("ymin_negative")
    if box.xmax > image_width:
        issues.append("xmax_exceeds_image_width")
    if box.ymax > image_height:
        issues.append("ymax_exceeds_image_height")
    if box.width_inclusive <= 0:
        issues.append("nonpositive_inclusive_width")
    if box.height_inclusive <= 0:
        issues.append("nonpositive_inclusive_height")
    if box.area_inclusive <= 0:
        issues.append("nonpositive_inclusive_area")
    return tuple(issues)


def inclusive_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Calculate IoU using the inclusive-coordinate convention used upstream."""

    intersection_width = max(0, min(first.xmax, second.xmax) - max(first.xmin, second.xmin) + 1)
    intersection_height = max(0, min(first.ymax, second.ymax) - max(first.ymin, second.ymin) + 1)
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    union = first.area_inclusive + second.area_inclusive - intersection
    return intersection / union
