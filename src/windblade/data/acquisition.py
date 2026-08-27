"""Official-only, immutable acquisition for the WTBD Figshare release."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.error
import urllib.request
import zipfile
import zlib

from windblade.results import read_json, write_json


DATASET_NAME = "WTBD — Wind Turbine Blade Defect dataset"
ARTICLE_TITLE = "Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery"
AUTHORS = ["Lipeng Ji", "Junjie Cheng", "Shilong Wu"]
ARTICLE_DOI = "10.1038/s41597-026-06762-x"
DATASET_DOI = "10.6084/m9.figshare.30210175"
VERSIONED_DATASET_DOI = "10.6084/m9.figshare.30210175.v1"
OFFICIAL_REPOSITORY = "Springer Nature Figshare"
OFFICIAL_PAGE_URL = (
    "https://springernature.figshare.com/articles/dataset/"
    "Multiclass_Dataset_for_Intelligent_Detection_of_Wind_Turbine_Blade_Defects_Using_Drone_Imagery/30210175"
)
OFFICIAL_API_URL = "https://api.figshare.com/v2/articles/30210175"
OFFICIAL_FILE_ID = 61029058
OFFICIAL_FILENAME = "WT blade defect dataset.zip"
OFFICIAL_FILE_SIZE = 78_958_553
OFFICIAL_FILE_MD5 = "14ad7e2cf7161b9100d1d70fb398b0cf"
OFFICIAL_DOWNLOAD_URL = "https://ndownloader.figshare.com/files/61029058"
LICENSE = "CC BY 4.0"
DATASET_VERSION = 1
USER_AGENT = "windblade-phase2/0.1 (official dataset DOI 10.6084/m9.figshare.30210175)"


class AcquisitionError(RuntimeError):
    """Raised when provenance or archive integrity validation fails."""


class AcquisitionBlockedError(AcquisitionError):
    """Raised when official automatic acquisition needs browser fallback."""


@dataclass(frozen=True)
class AcquisitionResult:
    archive_path: Path
    source_record_path: Path
    sha256: str
    size_bytes: int
    acquisition_method: str
    extracted_top_level_entries: tuple[str, ...]


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def hash_file(path: str | Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json, application/zip"},
    )


def fetch_official_metadata(timeout_seconds: int = 30) -> dict:
    try:
        with urllib.request.urlopen(_official_request(OFFICIAL_API_URL), timeout=timeout_seconds) as response:
            metadata = json.load(response)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        raise AcquisitionBlockedError(f"official Figshare metadata request failed: {exc}") from exc
    if not isinstance(metadata, dict):
        raise AcquisitionError("official Figshare API returned non-object metadata")
    return metadata


def _validate_official_metadata(metadata: dict) -> dict:
    if metadata.get("id") != 30210175:
        raise AcquisitionError("official API article ID does not match 30210175")
    if metadata.get("version") != DATASET_VERSION:
        raise AcquisitionError(
            f"official API version changed: expected {DATASET_VERSION}, found {metadata.get('version')!r}"
        )
    files = metadata.get("files")
    if not isinstance(files, list):
        raise AcquisitionError("official API metadata has no file list")
    matches = [item for item in files if item.get("id") == OFFICIAL_FILE_ID]
    if len(matches) != 1:
        raise AcquisitionError("official file ID 61029058 is missing or duplicated")
    source_file = matches[0]
    expected = {
        "name": OFFICIAL_FILENAME,
        "size": OFFICIAL_FILE_SIZE,
        "computed_md5": OFFICIAL_FILE_MD5,
        "download_url": OFFICIAL_DOWNLOAD_URL,
    }
    for field, expected_value in expected.items():
        if source_file.get(field) != expected_value:
            raise AcquisitionError(
                f"official file metadata changed for {field}: "
                f"expected {expected_value!r}, found {source_file.get(field)!r}"
            )
    return source_file


def _validate_archive_identity(path: Path) -> tuple[int, str, str]:
    size = path.stat().st_size
    if size != OFFICIAL_FILE_SIZE:
        raise AcquisitionError(
            f"archive byte size does not match official metadata: {size} != {OFFICIAL_FILE_SIZE}"
        )
    md5 = hash_file(path, "md5")
    if md5 != OFFICIAL_FILE_MD5:
        raise AcquisitionError(
            f"archive MD5 does not match official Figshare metadata: {md5} != {OFFICIAL_FILE_MD5}"
        )
    sha256 = hash_file(path, "sha256")
    return size, md5, sha256


def _download_official_archive(destination: Path, timeout_seconds: int = 120) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(
            _official_request(OFFICIAL_DOWNLOAD_URL), timeout=timeout_seconds
        ) as response, temporary.open("xb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        raise AcquisitionBlockedError(f"official Figshare archive download failed: {exc}") from exc
    temporary.replace(destination)


def _copy_manual_archive(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise AcquisitionError(f"manual archive does not exist: {source}")
    _validate_archive_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    if destination.exists():
        if hash_file(destination) != hash_file(source):
            raise AcquisitionError(f"refusing to overwrite different immutable archive: {destination}")
        return
    shutil.copy2(source, destination)


def _safe_member_path(raw_root: Path, member_name: str) -> Path:
    logical = PurePosixPath(member_name)
    if logical.is_absolute() or ".." in logical.parts:
        raise AcquisitionError(f"unsafe archive member path: {member_name!r}")
    target = raw_root.joinpath(*logical.parts)
    resolved_root = raw_root.resolve()
    resolved_target = target.resolve()
    if resolved_target != resolved_root and resolved_root not in resolved_target.parents:
        raise AcquisitionError(f"archive member escapes raw root: {member_name!r}")
    return target


def _existing_file_matches_zip_member(path: Path, member: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != member.file_size:
        return False
    crc = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            crc = zlib.crc32(chunk, crc)
    return (crc & 0xFFFFFFFF) == member.CRC


def extract_archive_immutably(archive_path: Path, raw_root: Path) -> tuple[str, ...]:
    top_level: set[str] = set()
    try:
        archive = zipfile.ZipFile(archive_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AcquisitionError(f"official archive is not a readable ZIP: {exc}") from exc

    with archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise AcquisitionError(f"ZIP CRC validation failed for: {corrupt_member}")
        for member in archive.infolist():
            logical = PurePosixPath(member.filename)
            if logical.parts:
                top_level.add(logical.parts[0])
            target = _safe_member_path(raw_root, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not _existing_file_matches_zip_member(target, member):
                    raise AcquisitionError(f"refusing to overwrite changed immutable raw file: {target}")
                continue
            with archive.open(member, "r") as source, target.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return tuple(sorted(top_level, key=str.casefold))


def acquire_wtbd(
    *,
    repository_root: str | Path,
    archive: str | Path | None = None,
    timeout_seconds: int = 120,
) -> AcquisitionResult:
    root = Path(repository_root).resolve()
    raw_root = root / "data" / "raw" / "wtbd"
    metadata_root = root / "data" / "metadata" / "wtbd"
    destination = raw_root / OFFICIAL_FILENAME

    if archive is None:
        metadata = fetch_official_metadata(min(timeout_seconds, 30))
        _validate_official_metadata(metadata)
        if not destination.exists():
            _download_official_archive(destination, timeout_seconds)
        method = "automatic official download"
    else:
        _copy_manual_archive(Path(archive).resolve(), destination)
        method = "manual official download"

    size, official_md5, sha256 = _validate_archive_identity(destination)
    top_level_entries = extract_archive_immutably(destination, raw_root)

    source_record = {
        "schema_version": "1.0",
        "dataset_name": DATASET_NAME,
        "article_title": ARTICLE_TITLE,
        "authors": AUTHORS,
        "article_doi": ARTICLE_DOI,
        "dataset_doi": DATASET_DOI,
        "versioned_dataset_doi": VERSIONED_DATASET_DOI,
        "dataset_version": DATASET_VERSION,
        "repository": OFFICIAL_REPOSITORY,
        "license": LICENSE,
        "acquisition_date_utc": _utc_timestamp(),
        "acquisition_method": method,
        "source_url": OFFICIAL_PAGE_URL,
        "official_api_url": OFFICIAL_API_URL,
        "official_file_id": OFFICIAL_FILE_ID,
        "official_supplied_md5": official_md5,
        "original_download_filename": OFFICIAL_FILENAME,
        "archive_sha256": sha256,
        "archive_byte_size": size,
        "archive_relative_path": destination.relative_to(root).as_posix(),
        "extraction_location": raw_root.relative_to(root).as_posix(),
        "extracted_top_level_entries": list(top_level_entries),
    }
    source_path = metadata_root / "source.json"
    if source_path.exists():
        existing = read_json(source_path)
        for field in (
            "dataset_doi",
            "dataset_version",
            "repository",
            "original_download_filename",
            "archive_sha256",
            "archive_byte_size",
        ):
            if existing.get(field) != source_record.get(field):
                raise AcquisitionError(
                    f"existing immutable source record disagrees on {field}: "
                    f"{existing.get(field)!r} != {source_record.get(field)!r}"
                )
    else:
        write_json(source_path, source_record)
    return AcquisitionResult(
        archive_path=destination,
        source_record_path=source_path,
        sha256=sha256,
        size_bytes=size,
        acquisition_method=method,
        extracted_top_level_entries=top_level_entries,
    )
