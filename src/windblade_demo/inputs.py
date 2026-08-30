"""In-memory image validation for the local demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import warnings

from PIL import Image, ImageOps, UnidentifiedImageError

from windblade_demo.constants import (
    MAX_IMAGE_DIMENSION,
    MAX_IMAGE_PIXELS,
    MAX_UPLOAD_BYTES,
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS,
)


class UploadValidationError(ValueError):
    """Raised when an upload cannot safely enter the demo workflow."""


@dataclass(frozen=True)
class DecodedUpload:
    image: Image.Image
    filename: str
    source_format: str
    byte_sha256: str
    byte_count: int


def decode_upload(payload: bytes, filename: str) -> DecodedUpload:
    """Validate and decode a PNG/JPEG entirely in memory, applying EXIF orientation."""

    if not isinstance(payload, bytes) or not payload:
        raise UploadValidationError("The uploaded file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("The uploaded file exceeds the 15 MB app limit.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UploadValidationError("Use a PNG, JPG, or JPEG file.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as probe:
                source_format = str(probe.format or "").upper()
                if source_format not in SUPPORTED_FORMATS:
                    raise UploadValidationError("The file contents are not PNG or JPEG.")
                probe.verify()
            with Image.open(BytesIO(payload)) as opened:
                oriented = ImageOps.exif_transpose(opened)
                width, height = oriented.size
                if width <= 0 or height <= 0:
                    raise UploadValidationError("The image dimensions are invalid.")
                if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                    raise UploadValidationError("An image dimension exceeds the 20,000 pixel safety limit.")
                if width * height > MAX_IMAGE_PIXELS:
                    raise UploadValidationError("The image exceeds the 50 megapixel safety limit.")
                image = oriented.convert("RGB").copy()
    except UploadValidationError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UploadValidationError("The image could not be decoded as a valid PNG or JPEG.") from exc
    except Image.DecompressionBombWarning as exc:
        raise UploadValidationError("The image dimensions exceed Pillow's safe decoding limit.") from exc

    return DecodedUpload(
        image=image,
        filename=Path(filename).name,
        source_format=source_format,
        byte_sha256=hashlib.sha256(payload).hexdigest(),
        byte_count=len(payload),
    )
