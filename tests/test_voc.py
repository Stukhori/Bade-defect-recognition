from __future__ import annotations

from pathlib import Path

import pytest

from windblade.data.voc import (
    BoundingBox,
    VocParseError,
    parse_voc_xml,
    validate_bounding_box,
)


def xml_document(objects: str) -> str:
    return f"""<annotation>
  <folder>JPEGImages</folder>
  <filename>7.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <segmented>0</segmented>
  {objects}
</annotation>"""


def object_xml(label: str = "craze", box: tuple[int, int, int, int] = (1, 2, 30, 40)) -> str:
    xmin, ymin, xmax, ymax = box
    return f"""<object>
  <name>{label}</name><pose>Unspecified</pose><truncated>0</truncated><difficult>0</difficult>
  <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
</object>"""


def test_valid_xml_with_multiple_objects_preserves_raw_labels(tmp_path: Path) -> None:
    path = tmp_path / "7.xml"
    path.write_text(
        xml_document(object_xml("craze") + object_xml("Surface_injure", (5, 6, 50, 60))),
        encoding="utf-8",
    )

    annotation = parse_voc_xml(path)

    assert annotation.filename == "7.jpg"
    assert (annotation.width, annotation.height, annotation.depth) == (100, 80, 3)
    assert [item.raw_label for item in annotation.objects] == ["craze", "Surface_injure"]
    assert annotation.objects[0].bbox.width_inclusive == 30


def test_malformed_xml_fails(tmp_path: Path) -> None:
    path = tmp_path / "bad.xml"
    path.write_text("<annotation><filename>bad.jpg", encoding="utf-8")

    with pytest.raises(VocParseError, match="cannot parse VOC XML"):
        parse_voc_xml(path)


def test_missing_label_fails(tmp_path: Path) -> None:
    path = tmp_path / "missing.xml"
    path.write_text(xml_document(object_xml().replace("<name>craze</name>", "")), encoding="utf-8")

    with pytest.raises(VocParseError, match=r"object\[0\]\.name"):
        parse_voc_xml(path)


def test_invalid_coordinate_text_fails(tmp_path: Path) -> None:
    path = tmp_path / "invalid.xml"
    path.write_text(xml_document(object_xml().replace("<xmin>1</xmin>", "<xmin>nan</xmin>")), encoding="utf-8")

    with pytest.raises(VocParseError, match="invalid integer"):
        parse_voc_xml(path)


@pytest.mark.parametrize(
    ("box", "expected_issue"),
    [
        (BoundingBox(10, 5, 10, 20), "xmin_not_less_than_xmax"),
        (BoundingBox(-1, 5, 10, 20), "xmin_negative"),
        (BoundingBox(1, 2, 101, 20), "xmax_exceeds_image_width"),
        (BoundingBox(1, 2, 10, 81), "ymax_exceeds_image_height"),
    ],
)
def test_invalid_boxes_are_reported(box: BoundingBox, expected_issue: str) -> None:
    assert expected_issue in validate_bounding_box(box, 100, 80)


def test_valid_box_has_no_issues() -> None:
    assert validate_bounding_box(BoundingBox(0, 0, 100, 80), 100, 80) == ()
