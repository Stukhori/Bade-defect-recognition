"""Audit and, when explicitly requested, repair Phase 9A Grad-CAM captions."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image
import torch

from windblade.config import load_config
from windblade.data.processed import LABELS, read_csv, sha256_file
from windblade.deep.checkpoints import state_dict_fingerprint
from windblade.deep.dataset import canonical_transform
from windblade.error_analysis.core import (
    ErrorAnalysisError,
    fingerprint_mapping,
    json_write,
)
from windblade.error_analysis.gradcam import (
    array_sha256,
    gradcam_map,
    resolve_module,
    validate_target_identity,
)
from windblade.error_analysis.review import (
    pass_b_caption_mismatches,
    render_pass_b_index,
)
from windblade.robustness.evaluation import load_frozen_cnn


SUMMARY = Path("experiments/summaries/phase9_error_analysis_v1")
BLANK_PASS_B_SHA256 = "7ae98fa0cb8c05edd57632460b1a08339c96f4f24b0a991fc1b7ac64ccdfa9e8"


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _blank_form_bytes(headers: list[str], review_ids: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
    writer.writeheader()
    writer.writerows(
        {field: review_id if field == "review_id" else "" for field in headers}
        for review_id in review_ids
    )
    return stream.getvalue().encode("utf-8")


def _file_fingerprint(paths: list[Path], repository: Path) -> str:
    hashes = {
        path.relative_to(repository).as_posix(): sha256_file(path)
        for path in sorted(set(paths))
    }
    return fingerprint_mapping(hashes)


def _verify_arrays(
    repository: Path, manifest: list[dict[str, str]]
) -> dict[str, Any]:
    config = load_config(repository / "configs/error_analysis.yaml").as_dict()
    phase8 = load_config(repository / config["inputs"]["phase8_config"]).as_dict()
    transform = canonical_transform()
    mismatches: list[dict[str, str]] = []
    role_counts: Counter[str] = Counter()
    checkpoint_checks: list[dict[str, Any]] = []
    for method in config["gradcam"]["methods"]:
        method_rows = [row for row in manifest if row["method"] == method]
        for seed in config["evaluation"]["seeds"]:
            rows = [row for row in method_rows if int(row["seed"]) == int(seed)]
            device = torch.device(config["runtime"]["device"])
            model = load_frozen_cnn(method, int(seed), phase8, repository, device)
            layer = resolve_module(model, str(config["gradcam"]["target_layers"][method]))
            before = state_dict_fingerprint(model.state_dict())
            for row in rows:
                with Image.open(repository / row["input_path"]) as handle:
                    image = handle.convert("RGB").copy()
                tensor = transform(image).unsqueeze(0).to(device)
                target_index = int(row["target_class_id"])
                array, shape, logits = gradcam_map(model, layer, tensor, target_index)
                role_counts[row["target_role"]] += 1
                reasons: list[str] = []
                if array_sha256(array) != row["array_sha256"]:
                    reasons.append("array_sha256")
                if LABELS[int(np.argmax(logits))] != row["predicted_label"]:
                    reasons.append("logit_argmax")
                if "x".join(map(str, shape)) != row["activation_shape"]:
                    reasons.append("activation_shape")
                if reasons:
                    mismatches.append(
                        {
                            "review_id": row["review_id"],
                            "method": method,
                            "seed": str(seed),
                            "input_state": row["input_state"],
                            "target_role": row["target_role"],
                            "target_class_id": row["target_class_id"],
                            "reasons": "|".join(reasons),
                        }
                    )
            after = state_dict_fingerprint(model.state_dict())
            checkpoint_checks.append(
                {
                    "method": method,
                    "seed": int(seed),
                    "state_before": before,
                    "state_after": after,
                    "parameters_unchanged": before == after,
                }
            )
            del model
    if any(not row["parameters_unchanged"] for row in checkpoint_checks):
        raise ErrorAnalysisError("read-only Grad-CAM verification changed a checkpoint state")
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "recomputed_records": sum(role_counts.values()),
        "recomputed_role_counts": dict(sorted(role_counts.items())),
        "array_or_logit_mismatches": mismatches,
        "checkpoint_checks": checkpoint_checks,
        "arrays_or_figures_written": 0,
    }


def audit(
    repository: Path, *, repair_caption_only: bool, verify_arrays: bool = False
) -> dict[str, Any]:
    summary = repository / SUMMARY
    review_root = summary / "human_review_packet"
    pass_a = review_root / "pass_a/pass_a_review_form.csv"
    pass_b = review_root / "pass_b/pass_b_review_form.csv"
    pass_b_page = review_root / "pass_b/index.html"
    reproduction_path = summary / "reproducibility.json"
    mapping = read_csv(review_root / "id_mapping/review_id_mapping.csv")
    manifest = read_csv(summary / "gradcam/gradcam_manifest.csv")
    errors = read_csv(summary / "error_manifest.csv")
    frozen = {
        (row["method"], row["seed"], row["condition_id"], row["instance_id"]): row
        for row in errors
    }

    target_mismatches: list[dict[str, str]] = []
    incorrect_frozen_identities: set[tuple[str, str, str, str]] = set()
    incorrect_display_identities: set[tuple[str, str, str, str, str, str]] = set()
    incorrect_review_ids: set[str] = set()
    incorrect_prediction_target_rows = 0
    role_counts: Counter[str] = Counter()
    target_index_counts: Counter[str] = Counter()
    evidence_paths: list[Path] = []
    for row in manifest:
        key = (row["method"], row["seed"], row["input_condition_id"], row["instance_id"])
        prediction = frozen.get(key)
        if prediction is None:
            target_mismatches.append({"review_id": row["review_id"], "reason": "missing_frozen_prediction"})
            continue
        try:
            validate_target_identity(
                row["target_role"], int(row["target_class_id"]), row["target_label"], prediction
            )
        except ErrorAnalysisError as exc:
            target_mismatches.append({"review_id": row["review_id"], "reason": str(exc)})
        role_counts[row["target_role"]] += 1
        target_index_counts[f'{row["target_class_id"]}:{row["target_label"]}'] += 1
        evidence_paths.extend(
            repository / row[field]
            for field in ("heatmap_path", "overlay_path")
        )
        if row["true_label"] != row["predicted_label"]:
            incorrect_frozen_identities.add(key)
            incorrect_display_identities.add(
                (
                    row["review_id"], row["method"], row["seed"],
                    row["input_condition_id"], row["instance_id"], row["input_state"],
                )
            )
            incorrect_review_ids.add(row["review_id"])
            if row["target_role"] == "predicted_class":
                incorrect_prediction_target_rows += 1

    page_before = pass_b_page.read_text(encoding="utf-8")
    caption_before = pass_b_caption_mismatches(
        page_before, manifest, repository, review_root
    )
    pass_a_before = pass_a.read_bytes()
    pass_b_before = pass_b.read_bytes()
    evidence_before = _file_fingerprint(evidence_paths, repository)
    reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
    old_output_fingerprint = reproduction["output_fingerprint"]

    record: dict[str, Any] = {
        "status": "PASS" if not target_mismatches else "FAIL",
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 9A Grad-CAM target identity and Pass B reporting",
        "gradcam_records": len(manifest),
        "target_role_counts": dict(sorted(role_counts.items())),
        "target_index_label_counts": dict(sorted(target_index_counts.items())),
        "target_identity_mismatches": target_mismatches,
        "incorrect_prediction_unique_frozen_identities": len(incorrect_frozen_identities),
        "incorrect_prediction_display_identities": len(incorrect_display_identities),
        "incorrect_prediction_target_rows": incorrect_prediction_target_rows,
        "incorrect_prediction_review_cases": len(incorrect_review_ids),
        "affected_review_ids": sorted(incorrect_review_ids),
        "caption_mismatches_before": len(caption_before),
        "caption_mismatch_role_counts_before": dict(
            sorted(Counter(row.get("target_role", "unknown") for row in caption_before).items())
        ),
        "caption_only": not target_mismatches and bool(caption_before),
        "underlying_arrays_or_figures_regenerated": False,
        "pass_a_sha256_before": hashlib.sha256(pass_a_before).hexdigest(),
        "pass_b_sha256_before": hashlib.sha256(pass_b_before).hexdigest(),
        "pass_b_required_answers_invalidated": sum(
            bool(value)
            for row in read_csv(pass_b)
            for field, value in row.items()
            if field != "review_id"
        ),
        "gradcam_evidence_fingerprint_before": evidence_before,
        "phase9b_started": False,
        "visual_interpretations_made": 0,
    }
    if verify_arrays:
        record["independent_array_recomputation"] = _verify_arrays(repository, manifest)
        if record["independent_array_recomputation"]["status"] != "PASS":
            record["status"] = "FAIL"
    if target_mismatches:
        if repair_caption_only:
            raise ErrorAnalysisError("caption-only repair refused because target identities are invalid")
        return record
    if not repair_caption_only:
        return record

    rendered_once, _ = render_pass_b_index(mapping, manifest, repository, review_root)
    rendered_twice, _ = render_pass_b_index(mapping, manifest, repository, review_root)
    if rendered_once != rendered_twice:
        raise ErrorAnalysisError("corrected Pass B page did not render deterministically")
    _atomic_write(pass_b_page, rendered_once.encode("utf-8"))

    with pass_b.open("r", encoding="utf-8", newline="") as handle:
        headers = list(csv.DictReader(handle).fieldnames or ())
    review_ids = [row["review_id"] for row in mapping]
    _atomic_write(pass_b, _blank_form_bytes(headers, review_ids))
    if sha256_file(pass_b) != BLANK_PASS_B_SHA256:
        raise ErrorAnalysisError("regenerated Pass B form does not match the frozen blank template")
    if pass_a.read_bytes() != pass_a_before:
        raise ErrorAnalysisError("Pass A changed during caption repair")

    corrected_page = pass_b_page.read_text(encoding="utf-8")
    caption_after = pass_b_caption_mismatches(
        corrected_page, manifest, repository, review_root
    )
    if caption_after:
        raise ErrorAnalysisError(f"corrected Pass B captions remain invalid: {caption_after[0]}")
    evidence_after = _file_fingerprint(evidence_paths, repository)
    if evidence_after != evidence_before:
        raise ErrorAnalysisError("caption-only repair changed Grad-CAM arrays or figures")

    page_key = f"{SUMMARY.as_posix()}/human_review_packet/pass_b/index.html"
    form_key = f"{SUMMARY.as_posix()}/human_review_packet/pass_b/pass_b_review_form.csv"
    reproduction["file_hashes"][page_key] = sha256_file(pass_b_page)
    reproduction["file_hashes"][form_key] = sha256_file(pass_b)
    reproduction["output_fingerprint"] = fingerprint_mapping(reproduction["file_hashes"])
    json_write(reproduction_path, reproduction)

    record.update(
        {
            "status": "PASS",
            "repair": "caption_only",
            "caption_mismatches_after": 0,
            "corrected_page_two_render_equality": True,
            "pass_a_sha256_after": sha256_file(pass_a),
            "pass_a_preserved_byte_for_byte": True,
            "pass_b_sha256_after": sha256_file(pass_b),
            "pass_b_required_answers_after": 0,
            "pass_b_page_sha256_before": hashlib.sha256(page_before.encode("utf-8")).hexdigest(),
            "pass_b_page_sha256_after": sha256_file(pass_b_page),
            "gradcam_evidence_fingerprint_after": evidence_after,
            "gradcam_arrays_and_figures_unchanged": True,
            "output_fingerprint_before": old_output_fingerprint,
            "output_fingerprint_after": reproduction["output_fingerprint"],
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-caption-only", action="store_true")
    parser.add_argument("--verify-arrays", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/audits/phase9a_gradcam_reporting_audit.json"),
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    record = audit(
        repository,
        repair_caption_only=args.repair_caption_only,
        verify_arrays=args.verify_arrays,
    )
    output = args.output if args.output.is_absolute() else repository / args.output
    json_write(output, record)
    print(json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
