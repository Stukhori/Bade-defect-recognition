"""Read the existing review HTML without accessing the separate ID mapping."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlsplit


class ReviewPacketError(ValueError):
    """Raised when a frozen packet page is malformed or unsafe."""


@dataclass(frozen=True)
class ReviewAsset:
    path: Path
    caption: str


@dataclass(frozen=True)
class ReviewCase:
    review_id: str
    metadata: str
    true_label: str
    assets: tuple[ReviewAsset, ...]


@dataclass(frozen=True)
class ReviewPacket:
    pass_name: str
    cases: tuple[ReviewCase, ...]

    @property
    def review_ids(self) -> tuple[str, ...]:
        return tuple(case.review_id for case in self.cases)


@dataclass
class _ParsedCase:
    review_id: str = ""
    metadata: str = ""
    images: list[tuple[str, str]] | None = None

    def __post_init__(self) -> None:
        if self.images is None:
            self.images = []


class _CaseParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.cases: list[_ParsedCase] = []
        self.current: _ParsedCase | None = None
        self.capture: str | None = None
        self.buffer: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "section" and "case" in str(attributes.get("class", "")).split():
            if self.current is not None:
                raise ReviewPacketError("nested review case sections are not allowed")
            self.current = _ParsedCase()
        elif self.current is not None and tag in {"h2", "p", "div"}:
            if self.capture is None:
                self.capture = tag
                self.buffer = []
        elif self.current is not None and tag == "img":
            source = attributes.get("src")
            if not source:
                raise ReviewPacketError("review image is missing a source")
            caption = " ".join("".join(self.buffer).split()) if self.capture == "div" else ""
            assert self.current.images is not None
            self.current.images.append((source, caption))

    def handle_data(self, data: str) -> None:
        if self.current is not None and self.capture is not None:
            self.buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self.current is not None and self.capture == tag:
            value = " ".join("".join(self.buffer).split())
            if tag == "h2":
                self.current.review_id = value
            elif tag == "p":
                self.current.metadata = value
            self.capture = None
            self.buffer = []
        if tag == "section" and self.current is not None:
            self.cases.append(self.current)
            self.current = None
            self.capture = None
            self.buffer = []


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _resolve_asset(source: str, page_root: Path, repository_root: Path) -> Path:
    parsed = urlsplit(source)
    if parsed.scheme or parsed.netloc or source.startswith("//"):
        raise ReviewPacketError("external review assets are prohibited")
    resolved = (page_root / source).resolve(strict=True)
    if not _inside(resolved, repository_root.resolve()):
        raise ReviewPacketError("review asset escapes the repository")
    if resolved.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ReviewPacketError("review asset is not a supported local image")
    return resolved


def load_review_packet(
    repository_root: str | Path,
    review_root: str | Path,
    pass_name: str,
) -> ReviewPacket:
    """Load exactly one existing packet page; never read the ID mapping."""

    if pass_name not in {"pass_a", "pass_b"}:
        raise ReviewPacketError(f"unsupported review pass: {pass_name}")
    repository = Path(repository_root).resolve()
    packet_root = Path(review_root).resolve()
    page_root = packet_root / pass_name
    index = page_root / "index.html"
    text = index.read_text(encoding="utf-8")
    parser = _CaseParser()
    parser.feed(text)
    parser.close()
    expected_ids = tuple(f"P9A-{index:03d}" for index in range(1, 61))
    observed_ids = tuple(case.review_id for case in parser.cases)
    if observed_ids != expected_ids:
        raise ReviewPacketError("packet review IDs or row order changed")

    cases: list[ReviewCase] = []
    for parsed_case in parser.cases:
        assert parsed_case.images is not None
        if pass_name == "pass_a":
            match = re.fullmatch(r"Dataset true label: ([a-z_]+)", parsed_case.metadata)
            if match is None:
                raise ReviewPacketError(f"unexpected Pass A metadata: {parsed_case.review_id}")
            true_label = match.group(1)
            allowed_names = {"clean.png", "clean_annotation.png", "degraded.png", "degraded_annotation.png"}
            if len(parsed_case.images) not in {2, 4}:
                raise ReviewPacketError(f"unexpected Pass A image count: {parsed_case.review_id}")
        else:
            match = re.search(r"true label: ([a-z_]+)$", parsed_case.metadata)
            if match is None:
                raise ReviewPacketError(f"unexpected Pass B metadata: {parsed_case.review_id}")
            true_label = match.group(1)
            allowed_names = None
            if not parsed_case.images:
                raise ReviewPacketError(f"Pass B evidence is absent: {parsed_case.review_id}")
        assets: list[ReviewAsset] = []
        for source, caption in parsed_case.images:
            path = _resolve_asset(source, page_root, repository)
            if pass_name == "pass_a":
                expected_asset_root = (page_root / "assets" / parsed_case.review_id).resolve()
                if not _inside(path, expected_asset_root) or path.name not in allowed_names:
                    raise ReviewPacketError(f"unsafe Pass A asset path: {parsed_case.review_id}")
                caption = {
                    "clean.png": "Clean review image",
                    "clean_annotation.png": "Clean image with annotation rectangle",
                    "degraded.png": "Degraded review image",
                    "degraded_annotation.png": "Degraded image with annotation rectangle",
                }[path.name]
            assets.append(ReviewAsset(path=path, caption=caption))
        cases.append(
            ReviewCase(
                review_id=parsed_case.review_id,
                metadata=parsed_case.metadata,
                true_label=true_label,
                assets=tuple(assets),
            )
        )
    return ReviewPacket(pass_name=pass_name, cases=tuple(cases))
