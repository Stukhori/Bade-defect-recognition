from __future__ import annotations

from io import BytesIO

from PIL import Image
import pytest

from windblade_demo.constants import MAX_UPLOAD_BYTES
from windblade_demo.inputs import UploadValidationError, decode_upload


def encoded(image: Image.Image, format_name: str, **kwargs) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format=format_name, **kwargs)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("image", "filename", "format_name"),
    [
        (Image.new("L", (12, 9), 100), "gray.png", "PNG"),
        (Image.new("RGBA", (12, 9), (10, 20, 30, 40)), "alpha.PNG", "PNG"),
        (Image.new("RGB", (12, 9), (10, 20, 30)), "photo.jpeg", "JPEG"),
    ],
)
def test_supported_modes_decode_to_rgb(image, filename, format_name):
    result = decode_upload(encoded(image, format_name), filename)
    assert result.image.mode == "RGB"
    assert result.image.size == image.size
    assert result.source_format == format_name
    assert len(result.byte_sha256) == 64


def test_exif_orientation_is_applied_before_rgb_conversion():
    image = Image.new("RGB", (8, 5), (10, 20, 30))
    exif = Image.Exif()
    exif[274] = 6
    result = decode_upload(encoded(image, "JPEG", exif=exif), "oriented.jpg")
    assert result.image.size == (5, 8)


@pytest.mark.parametrize(
    ("payload", "filename", "message"),
    [
        (b"", "empty.png", "empty"),
        (b"not an image", "broken.png", "could not be decoded"),
        (encoded(Image.new("RGB", (2, 2)), "PNG"), "wrong.gif", "PNG, JPG, or JPEG"),
    ],
)
def test_invalid_uploads_are_rejected(payload, filename, message):
    with pytest.raises(UploadValidationError, match=message):
        decode_upload(payload, filename)


def test_oversized_payload_is_rejected_before_decode():
    with pytest.raises(UploadValidationError, match="15 MB"):
        decode_upload(b"x" * (MAX_UPLOAD_BYTES + 1), "large.png")


def test_extreme_but_decodable_dimensions_are_safely_rejected():
    payload = encoded(Image.new("RGB", (20_001, 1)), "PNG")
    with pytest.raises(UploadValidationError, match="20,000"):
        decode_upload(payload, "very_wide.png")
