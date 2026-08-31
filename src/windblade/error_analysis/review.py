"""Blinded two-pass human-review packet generation."""

from __future__ import annotations

import html
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from windblade.error_analysis.core import csv_write
from windblade.error_analysis.gradcam import annotation_box, render_annotation


def _save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><title>{html.escape(title)}</title>
<style>body{{font-family:Arial,sans-serif;max-width:1200px;margin:24px auto;padding:0 16px}}.case{{border:1px solid #bbb;padding:16px;margin:18px 0}}img{{max-width:224px;margin:4px;vertical-align:top}}table{{border-collapse:collapse}}td,th{{border:1px solid #aaa;padding:6px}}code{{white-space:pre-wrap}}</style></head><body><h1>{html.escape(title)}</h1>{body}</body></html>\n"""


def _contact_sheets(review_root: Path, selected: Sequence[Mapping[str, Any]], assets: Mapping[str, list[Path]], pass_name: str) -> list[str]:
    outputs: list[str] = []
    page_size = 12
    for page_index, start in enumerate(range(0, len(selected), page_size), start=1):
        page_rows = selected[start:start + page_size]
        width, cell_height = 700, 250
        sheet = Image.new("RGB", (width, cell_height * len(page_rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for row_index, row in enumerate(page_rows):
            y = row_index * cell_height
            draw.text((8, y + 8), f"{row['review_id']} | label: {row['true_label']}", fill="black")
            for image_index, path in enumerate(assets.get(str(row["review_id"]), [])[:3]):
                with Image.open(path) as source:
                    thumb = source.convert("RGB").copy(); thumb.thumbnail((210, 210))
                sheet.paste(thumb, (8 + image_index * 225, y + 32))
        path = review_root / pass_name / "contact_sheets" / f"page_{page_index:02d}.png"
        _save(sheet, path); outputs.append(path.relative_to(review_root).as_posix())
    return outputs


def render_pass_b_index(
    mapping_rows: Sequence[Mapping[str, Any]],
    gradcam_rows: Sequence[Mapping[str, Any]],
    root: Path,
    review_root: Path,
) -> tuple[str, dict[str, list[Path]]]:
    """Render Pass B captions from the explicit Grad-CAM target identity."""

    grad_by_review: dict[str, list[Mapping[str, Any]]] = {}
    for row in gradcam_rows:
        grad_by_review.setdefault(str(row["review_id"]), []).append(row)
    body = "<p>Open only after Pass A is complete. Maps are independently normalized: color intensity is not quantitatively comparable across maps. Grad-CAM is descriptive and does not prove causal reasoning.</p>"
    pass_b_assets: dict[str, list[Path]] = {}
    for row in mapping_rows:
        review_id = str(row["review_id"])
        evidence = sorted(
            grad_by_review.get(review_id, []),
            key=lambda item: (
                int(item["seed"]),
                str(item["input_state"]),
                str(item["target_role"]),
            ),
        )
        links: list[str] = []
        assets: list[Path] = []
        for item in evidence:
            path = root / str(item["overlay_path"])
            assets.append(path)
            relative = Path(os.path.relpath(path, review_root / "pass_b"))
            links.append(
                f'<div>seed {item["seed"]}, {html.escape(str(item["input_state"]))}, '
                f'{html.escape(str(item["target_role"]))}: {html.escape(str(item["target_label"]))}'
                f'<br><img src="{html.escape(relative.as_posix())}"></div>'
            )
        pass_b_assets[review_id] = assets
        body += (
            f'<section class="case"><h2>{review_id}</h2><p>Model: '
            f'{html.escape(str(row["method"]))}; condition: {html.escape(str(row["condition_id"]))}; '
            f'event: {html.escape(str(row["selection_event"]))}; rule: '
            f'{html.escape(str(row["eligibility_rule"]))}; true label: '
            f'{html.escape(str(row["true_label"]))}</p>{"".join(links)}</section>'
        )
    return _page("Phase 9A Pass B — model-evidence review", body), pass_b_assets


def pass_b_caption_mismatches(
    page_text: str,
    gradcam_rows: Sequence[Mapping[str, Any]],
    root: Path,
    review_root: Path,
) -> list[dict[str, str]]:
    """Return any evidence row whose rendered caption is not its target label."""

    mismatches: list[dict[str, str]] = []
    for item in gradcam_rows:
        review_id = str(item["review_id"])
        marker = f'<section class="case"><h2>{review_id}</h2>'
        start = page_text.find(marker)
        end = page_text.find("</section>", start)
        if start < 0 or end < 0:
            mismatches.append({"review_id": review_id, "reason": "case_missing"})
            continue
        section = page_text[start:end]
        path = root / str(item["overlay_path"])
        relative = Path(os.path.relpath(path, review_root / "pass_b")).as_posix()
        expected = (
            f'{html.escape(str(item["target_role"]))}: '
            f'{html.escape(str(item["target_label"]))}<br><img src="{html.escape(relative)}">'
        )
        if expected not in section:
            mismatches.append(
                {
                    "review_id": review_id,
                    "seed": str(item["seed"]),
                    "input_state": str(item["input_state"]),
                    "target_role": str(item["target_role"]),
                    "target_label": str(item["target_label"]),
                    "reason": "caption_or_asset_mismatch",
                }
            )
    return mismatches


def create_review_packet(
    config: Mapping[str, Any], root: Path, selected: Sequence[Mapping[str, Any]], geometry: Mapping[str, Mapping[str, Any]],
    corrupted_paths: Mapping[tuple[str, str], str], gradcam_rows: Sequence[Mapping[str, Any]], figures_root: Path, summary_root: Path,
) -> dict[str, Any]:
    review_root = summary_root / "human_review_packet"
    pass_a_assets: dict[str, list[Path]] = {}
    mapping_rows: list[dict[str, Any]] = []
    for case in selected:
        review_id, instance_id, condition = str(case["review_id"]), str(case["instance_id"]), str(case["condition_id"])
        meta = geometry[instance_id]
        clean_path = root / config["dataset"]["processed_root"] / meta["output_relative_path"]
        with Image.open(clean_path) as handle: clean = handle.convert("RGB").copy()
        box = annotation_box(meta)
        asset_root = review_root / "pass_a" / "assets" / review_id
        clean_out, annotation_out = asset_root / "clean.png", asset_root / "clean_annotation.png"
        _save(clean, clean_out); _save(render_annotation(clean, box), annotation_out)
        assets = [clean_out, annotation_out]
        degraded_out = ""
        if condition != "clean":
            with Image.open(root / corrupted_paths[(instance_id, condition)]) as handle: degraded = handle.convert("RGB").copy()
            degraded_path, degraded_annotation = asset_root / "degraded.png", asset_root / "degraded_annotation.png"
            _save(degraded, degraded_path); _save(render_annotation(degraded, box), degraded_annotation)
            assets.extend([degraded_path, degraded_annotation]); degraded_out = degraded_path.relative_to(review_root).as_posix()
        pass_a_assets[review_id] = assets
        mapping_rows.append({"review_id": review_id, "instance_id": instance_id, "source_image_id": case["source_image_id"], "method": case["method"], "condition_id": condition, "selection_event": case["selection_event"], "eligibility_rule": case["eligibility_rule"], "true_label": case["true_label"], "clean_asset": clean_out.relative_to(review_root).as_posix(), "degraded_asset": degraded_out})

    pass_a_body = "<p>Complete Pass A before opening Pass B. Model identity, predictions, correctness, event type, and Grad-CAM are intentionally hidden. Do not open the separate ID mapping.</p>"
    for row in mapping_rows:
        rid = row["review_id"]
        images = "".join(f'<img src="{html.escape(path.relative_to(review_root / "pass_a").as_posix())}" alt="{rid}">' for path in pass_a_assets[rid])
        pass_a_body += f'<section class="case"><h2>{rid}</h2><p>Dataset true label: <b>{html.escape(str(row["true_label"]))}</b></p>{images}</section>'
    (review_root / "pass_a").mkdir(parents=True, exist_ok=True)
    (review_root / "pass_a" / "index.html").write_text(_page("Phase 9A Pass A — visual-quality review", pass_a_body), encoding="utf-8")

    pass_b_page, pass_b_assets = render_pass_b_index(
        mapping_rows, gradcam_rows, root, review_root
    )
    (review_root / "pass_b").mkdir(parents=True, exist_ok=True)
    (review_root / "pass_b" / "index.html").write_text(pass_b_page, encoding="utf-8")

    pass_a_fields = ["review_id", *config["review_packet"]["pass_a_fields"].keys()]
    pass_b_fields = ["review_id", *config["review_packet"]["pass_b_fields"].keys()]
    csv_write(review_root / "pass_a" / "pass_a_review_form.csv", [{field: row["review_id"] if field == "review_id" else "" for field in pass_a_fields} for row in mapping_rows], pass_a_fields)
    csv_write(review_root / "pass_b" / "pass_b_review_form.csv", [{field: row["review_id"] if field == "review_id" else "" for field in pass_b_fields} for row in mapping_rows], pass_b_fields)
    csv_write(review_root / "id_mapping" / "review_id_mapping.csv", mapping_rows)
    pass_a_sheets = _contact_sheets(review_root, selected, pass_a_assets, "pass_a")
    pass_b_sheets = _contact_sheets(review_root, selected, pass_b_assets, "pass_b")
    instructions = """# Phase 9A human review instructions\n\n1. Open `pass_a/index.html` and complete `pass_a/pass_a_review_form.csv` without opening Pass B or the ID mapping.\n2. After saving every Pass A judgment, open `pass_b/index.html` and complete `pass_b/pass_b_review_form.csv`.\n3. Do not edit review IDs or the separate mapping. Use only the declared response choices; notes are optional free text.\n4. Return both completed CSV files. Phase 9B will validate and incorporate them.\n\nGrad-CAM maps are independently normalized visualizations. Their colors cannot be compared quantitatively across maps, and they do not prove what caused a prediction.\n"""
    (review_root / "README.md").write_text(instructions, encoding="utf-8")
    return {"review_case_count": len(selected), "pass_a_form": "human_review_packet/pass_a/pass_a_review_form.csv", "pass_b_form": "human_review_packet/pass_b/pass_b_review_form.csv", "mapping": "human_review_packet/id_mapping/review_id_mapping.csv", "pass_a_contact_sheets": pass_a_sheets, "pass_b_contact_sheets": pass_b_sheets}
