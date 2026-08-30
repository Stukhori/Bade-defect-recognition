"""Validate the local review interface without writing canonical judgments."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from windblade_review.packet import ReviewPacket, load_review_packet
from windblade_review.schema import PassSchema, load_pass_schema
from windblade_review.store import ReviewStore
from windblade_review.workflow import GateError, pass_b_access_allowed, validate_pass_a_lock


BLANK_HASHES = {
    "pass_a": "44a5200e8b921b65f55cd391943abfbd4ca600e9723fcd8c70a36eb6cf2b7d58",
    "pass_b": "7ae98fa0cb8c05edd57632460b1a08339c96f4f24b0a991fc1b7ac64ccdfa9e8",
}
MAPPING_HASH = "46b19248797997e8aa7236b9c3fbf17f972977f2cdb2c3a958bd12513f25210b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _form_path(root: Path, pass_name: str) -> Path:
    name = "pass_a_review_form.csv" if pass_name == "pass_a" else "pass_b_review_form.csv"
    return root / pass_name / name


def _schema_record(schema: PassSchema) -> dict[str, Any]:
    return {
        "headers": list(schema.headers),
        "required_fields": list(schema.required_fields),
        "choices": {
            field.name: list(field.choices)
            for field in schema.fields
            if field.required
        },
        "optional_fields": [field.name for field in schema.fields if not field.required],
    }


def _form_record(store: ReviewStore, repository_root: Path) -> dict[str, Any]:
    snapshot = store.load()
    return {
        "path": store.path.resolve().relative_to(repository_root.resolve()).as_posix(),
        "sha256": store.sha256(),
        "rows": len(snapshot.rows),
        "answered_required": snapshot.answered_required,
        "total_required": snapshot.total_required,
        "complete_cases": snapshot.completed_cases,
        "complete": snapshot.complete,
    }


def _first_choices(schema: PassSchema) -> dict[str, str]:
    return {
        field.name: (field.choices[0] if field.required else "")
        for field in schema.fields
    }


def _complete_copy(store: ReviewStore) -> None:
    values = _first_choices(store.schema)
    for review_id in store.expected_ids:
        store.save_case(review_id, values)


def _packet_fingerprint(
    packets: tuple[ReviewPacket, ...], mapping: Path, repository_root: Path
) -> dict[str, Any]:
    paths = {mapping}
    for packet in packets:
        for case in packet.cases:
            paths.update(asset.path for asset in case.assets)
    entries = [
        (path.resolve().relative_to(repository_root.resolve()).as_posix(), _sha256(path))
        for path in sorted(paths)
    ]
    aggregate = hashlib.sha256(
        "".join(f"{path}\0{digest}\n" for path, digest in entries).encode("utf-8")
    ).hexdigest()
    return {"file_count": len(entries), "aggregate_sha256": aggregate}


def validate(root: Path, *, require_blank: bool, forms_only: bool) -> dict[str, Any]:
    packet_root = root / "experiments/summaries/phase9_error_analysis_v1/human_review_packet"
    config = root / "configs/error_analysis.yaml"
    schemas = {
        name: load_pass_schema(config, name)
        for name in ("pass_a", "pass_b")
    }
    packets = {
        name: load_review_packet(root, packet_root, name)
        for name in ("pass_a", "pass_b")
    }
    stores = {
        name: ReviewStore(_form_path(packet_root, name), schemas[name], packets[name].review_ids)
        for name in ("pass_a", "pass_b")
    }
    forms = {name: _form_record(store, root) for name, store in stores.items()}
    if forms["pass_a"]["total_required"] != 300 or forms["pass_b"]["total_required"] != 240:
        raise RuntimeError("canonical required-answer counts changed")
    if require_blank:
        for name in ("pass_a", "pass_b"):
            if forms[name]["answered_required"] != 0 or forms[name]["sha256"] != BLANK_HASHES[name]:
                raise RuntimeError(f"canonical {name} is not the frozen blank form")

    mapping = packet_root / "id_mapping/review_id_mapping.csv"
    if _sha256(mapping) != MAPPING_HASH:
        raise RuntimeError("separate review-ID mapping changed")
    packet_before = _packet_fingerprint(tuple(packets.values()), mapping, root)
    record: dict[str, Any] = {
        "status": "PASS",
        "scope": "non-scientific local human-review interface",
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "schemas": {name: _schema_record(schema) for name, schema in schemas.items()},
        "canonical_forms": forms,
        "mapping_sha256": MAPPING_HASH,
        "packet_and_mapping": packet_before,
        "prohibited_actions": {
            "canonical_judgments_written": 0,
            "mapping_reads_by_interface": 0,
            "model_or_llm_inference": 0,
            "external_service_calls": 0,
            "scientific_artifacts_modified": 0,
        },
    }
    if forms_only:
        record["mode"] = "forms_only"
        return record

    cache = root / "experiments/cache"
    cache.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="review_interface_validation_", dir=cache))
    try:
        copied = temporary / "forms"
        for name in ("pass_a", "pass_b"):
            destination = _form_path(copied, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(stores[name].path, destination)
        copied_a = ReviewStore(_form_path(copied, "pass_a"), schemas["pass_a"], packets["pass_a"].review_ids)
        copied_b = ReviewStore(_form_path(copied, "pass_b"), schemas["pass_b"], packets["pass_b"].review_ids)

        if copied_a.load().answered_required != 0 or copied_b.load().answered_required != 0:
            raise RuntimeError("full temporary smoke validation requires blank copied forms")
        if pass_b_access_allowed(copied_a.load(), pass_a_locked=False, pass_b_started=True):
            raise RuntimeError("Pass B gate opened for incomplete Pass A")
        copied_a.save_case(packets["pass_a"].review_ids[0], {"defect_visible": "uncertain"})
        if copied_a.load().first_incomplete_index != 0:
            raise RuntimeError("partial Pass A did not resume at its first incomplete case")
        try:
            validate_pass_a_lock(copied_a, True)
        except GateError:
            pass
        else:
            raise RuntimeError("incomplete Pass A unexpectedly locked")

        _complete_copy(copied_a)
        try:
            validate_pass_a_lock(copied_a, False)
        except GateError:
            pass
        else:
            raise RuntimeError("Pass A locked without attestation")
        locked_hash = validate_pass_a_lock(copied_a, True)
        if pass_b_access_allowed(copied_a.load(), pass_a_locked=True, pass_b_started=False):
            raise RuntimeError("locking Pass A automatically opened Pass B")
        if not pass_b_access_allowed(copied_a.load(), pass_a_locked=True, pass_b_started=True):
            raise RuntimeError("deliberate Pass B gate did not open")

        copied_b.save_case(packets["pass_b"].review_ids[0], _first_choices(schemas["pass_b"]))
        if copied_b.load().first_incomplete_index != 1:
            raise RuntimeError("partial Pass B did not resume at its first incomplete case")
        _complete_copy(copied_b)
        completed_b = copied_b.load(require_complete=True)
        if completed_b.answered_required != 240:
            raise RuntimeError("temporary Pass B completion count changed")
        record["temporary_copy_smoke"] = {
            "status": "PASS",
            "pass_a_required_answers": 300,
            "pass_a_locked_sha256": locked_hash,
            "pass_b_required_answers": 240,
            "pass_b_opened_only_after_deliberate_action": True,
            "resume_checks": "PASS",
            "temporary_data_removed": True,
        }
    finally:
        resolved_temp = temporary.resolve()
        resolved_cache = cache.resolve()
        if resolved_temp.parent != resolved_cache:
            raise RuntimeError("refusing temporary cleanup outside the review validation cache")
        shutil.rmtree(resolved_temp)

    for name, store in stores.items():
        after = _form_record(store, root)
        if after != forms[name]:
            raise RuntimeError(f"canonical {name} changed during temporary validation")
    if _packet_fingerprint(tuple(packets.values()), mapping, root) != packet_before:
        raise RuntimeError("packet assets or mapping changed during validation")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-blank", action="store_true")
    parser.add_argument("--check-forms-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    record = validate(root, require_blank=args.require_blank, forms_only=args.check_forms_only)
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
